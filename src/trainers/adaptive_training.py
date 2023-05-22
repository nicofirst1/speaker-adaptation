import copy
import datetime
from typing import Dict

import lovely_tensors as lt
import numpy as np
import rich
import torch
import wandb
from rich.progress import Progress
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.commons import (
    SPEAKER_CHK,
    AccuracyEstimator,
    get_dataloaders,
    get_listener_check,
    load_wandb_checkpoint,
    mask_oov_embeds,
    parse_args,
    set_seed,
    EarlyStopping,
)
from src.commons.Translator import Translator
from src.commons.model_utils import get_mask_from_utts, save_model, merge_dict, compare_model_weights, LeakFinder
from src.data.dataloaders import Vocab
from src.models import ListenerModel
from src.models.simulator.SimulatorModel import SimulatorModel
from src.models.speaker.SpeakerModelEC import SpeakerModelEC
from src.wandb_logging import WandbLogger

global cudaper


def normalize_aux(aux, data_length):
    aux["s_loss"] = np.mean(aux["s_loss"]).mean()
    aux["sim_list_loss"] = np.mean(aux["sim_list_loss"])

    aux["sim_list_acc"] = np.mean([x[-1] for x in aux["sim_list_acc"]])
    aux["adapted_list_target_acc"] = np.mean([x[-1] for x in aux["adapted_list_target_acc"]])
    aux["sim_target_acc"] = np.mean([x[-1] for x in aux["sim_accuracy"]]).mean()
    aux["original_acc"] = np.mean(aux["original_acc"]) / data_length

    del aux["sim_accuracy"]



def predict(
        data: Dict,
        speak_model: SpeakerModelEC,
        list_model: ListenerModel,
        sim_model: SimulatorModel,
        criterion,
        translator: Translator,
        adapt_lr: float,
        s: int,
):


    ## extract data
    context_separate = data["separate_images"]
    target_img_feats = data["target_img_feats"]
    targets = data["target"]



    ################################################
    #   Get results with original hypo
    ################################################
    set_seed(seed)
    # generate hypothesis
    with torch.no_grad():
        utterance_s, logs, decoder_hid = speak_model.generate_hypothesis(
            context_separate,
            target_img_feats,
        )

        utterance = translator.s2l(utterance_s)

        history_att = logs["history_att"]

        # translate utt to ids and feed to listener

        masks = get_mask_from_utts(utterance, translator.list_vocab, device)

        list_out = list_model(utterance, context_separate, masks)

        # # get  accuracy
        list_preds = torch.argmax(list_out, dim=1)
        list_target_accuracy = torch.eq(list_preds.squeeze(dim=-1), targets.squeeze()).sum().double().item()
        original_acc = list_target_accuracy




    ################################################
    #   Get results with adapted hypo
    ################################################
    # decoder_hid = normalize(decoder_hid)
    h0 = decoder_hid.clone().requires_grad_(True)
    h0_optimizer = torch.optim.Adam([h0], lr=adapt_lr)
    h0_optimizer.zero_grad()

    # repeat for s interations
    s_accs = []
    s_loss = []
    sim_accuracy = []
    sim_list_acc = []

    sim_list_loss = 0

    # perform loop
    i = 0
    while i < s:
        set_seed(seed)
        h0_optimizer.zero_grad()

        sim_out = sim_model(
            separate_images=context_separate,
            utterance=utterance,
            masks=masks,
            speaker_embeds=h0,
        )
        sim_list_loss += criterion(sim_out, list_preds)

        # compute loss and perform backprop
        loss = criterion(sim_out, targets)
        aux = acc_estimator(sim_out, targets, list_out, data["domain"])
        loss.backward(retain_graph=True)
        h0_optimizer.step()

        s_loss.append(loss.detach().item())

        with torch.no_grad():
            # get modified hypo
            utterance_s, dec_logit = speak_model.nucleus_sampling(h0, history_att)
            utterance = translator.s2l(utterance_s)

            # generate utt for list
            # translate utt to ids and feed to listener
            masks = get_mask_from_utts(utterance, translator.list_vocab, device)

            list_out = list_model(utterance, context_separate, masks)
            list_preds = torch.argmax(list_out, dim=1)

            sim_accuracy.append(aux["sim_target_accuracy"])
            sim_list_acc.append(aux["sim_list_accuracy"])
            s_accs.append(aux["list_target_accuracy"])


        # break if listener gets it right
        if aux["sim_target_accuracy"]:
            break
        i += 1

    res = dict(
        original_acc=original_acc,
        adapted_list_target_acc=s_accs,
        sim_accuracy=sim_accuracy,
        sim_list_acc=sim_list_acc,
        domains=data["domain"],
        sim_list_loss=sim_list_loss.detach().item(),

        s_loss=s_loss,
    )

    return res, sim_list_loss


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain
    lt.monkey_patch()

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.autograd.set_detect_anomaly(True)

    ###################################
    ##  LOGGER
    ###################################

    list_vocab = Vocab(common_p.vocab_file, is_speaker=False)

    flag = common_p.type_of_int

    logger = WandbLogger(
        vocab=list_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        project=f"adaptive-training",
        tags=common_p.tags,
    )

    metric = common_p.metric
    sweep_config = wandb.config

    is_sweep = sweep_config._check_locked("adapt_lr")

    ##########################
    # LISTENER
    ##########################

    list_checkpoint = get_listener_check(common_p.train_domain)
    list_checkpoint, _ = load_wandb_checkpoint(list_checkpoint, device)
    list_args = list_checkpoint["args"]

    # update paths
    list_args.reset_paths()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    list_model = ListenerModel(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        domain,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    # mask OOV words in the vocab

    with torch.no_grad():
        list_model.embeddings = mask_oov_embeds(
            list_model.embeddings,
            list_vocab,
            domain,
            replace_token=common_p.mask_oov_embed,
            data_path=common_p.data_path,
        )

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
        deterministic=True,
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)
    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################

    sim_model = SimulatorModel(
        len(list_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.sim_domain,
        device,
        embed_temp=common_p.embed_temp,
    ).to(device)

    ###################################
    ##  Get speaker dataloader
    ###################################

    train_dl_all, test_dl_all, val_dl_all = get_dataloaders(
        common_p, speak_vocab, domain="all", unary_val_bs=False
    )

    translator = Translator(speak_vocab, list_vocab, device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.AdamW(
        sim_model.parameters(), lr=common_p.learning_rate, weight_decay=0.0001
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=3, factor=0.25, verbose=True, threshold=0.5
    )
    loss_f = nn.CrossEntropyLoss(reduction=common_p.reduction)
    acc_estimator = AccuracyEstimator(domain, all_domains=logger.domains)

    ###################################
    ## RESUME AND EARLYSTOPPING
    ###################################

    metric = common_p.metric

    if metric == "loss":
        es = EarlyStopping(common_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(common_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model], )

    ###################################
    ##  ADAPTIVE TRAINING
    ###################################
    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )


    for epoch in range(common_p.epochs):
        print("Epoch : ", epoch)

        auxs = []

        sim_model.train()

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.progress.track(
                enumerate(train_dl_all),
                total=len(train_dl_all),
                description=f"Training epoch {epoch}",
        ):

            optimizer.zero_grad()
            # get datapoints
            aux, loss = predict(
                data,
                speak_model=speaker_model,
                list_model=list_model,
                sim_model=sim_model,
                criterion=loss_f,
                translator=translator,
                adapt_lr=common_p.adapt_lr,
                s=common_p.s_iter,
            )

            auxs.append(aux)

            # # optimizer
            loss.backward()
            # nn.utils.clip_grad_value_(sim_model.parameters(), clip_value=1.0)
            optimizer.step()

        aux = merge_dict(auxs)
        normalize_aux(aux, len(train_dl_all.dataset.data))
        logger.on_eval_end(
            aux, list_domain=train_dl_all.dataset.domain, modality="train"
        )

        print(
            f"Train loss {aux['sim_list_loss']:.6f}, accuracy {aux['sim_list_acc'] * 100:.2f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        sim_model.eval()
        sim_model_copy = copy.deepcopy(sim_model)

        print(f"\nEvaluation")
        auxs = []

        for ii, data in rich.progress.track(
                enumerate(val_dl_all),
                total=len(val_dl_all),
                description=f"evaluating...",
        ):
            aux, loss = predict(
                data,
                speak_model=speaker_model,
                list_model=list_model,
                sim_model=sim_model,
                criterion=loss_f,
                translator=translator,
                adapt_lr=common_p.adapt_lr,
                s=common_p.s_iter, )

            auxs.append(aux)

        compare_model_weights(sim_model, sim_model_copy)

        aux = merge_dict(auxs)
        normalize_aux(
            aux, len(val_dl_all.dataset.data)
        )

        eval_accuracy, eval_loss = aux["sim_list_acc"], aux["sim_list_loss"]

        scheduler.step(eval_accuracy)

        print(
            f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f} "
        )
        logger.on_eval_end(
            aux, list_domain=val_dl_all.dataset.domain, modality="eval"
        )

        ###################################
        ##  Saving and early stopping
        ###################################

        if epoch > 0 and epoch % 2 == 0:
            save_model(
                model=sim_model,
                model_type=f"Simulator{domain.capitalize()}",
                epoch=epoch,
                accuracy=eval_accuracy,
                args=common_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if common_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
        print("\n\n")
