import copy
import datetime
import operator
from typing import Dict, Optional, Tuple

import numpy as np
import rich.progress
import spacy as spacy
import torch
from numpy import mean
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from src.commons import (LISTENER_CHK_DICT,
                         SPEAKER_CHK, EarlyStopping, get_dataloaders,
                         load_wandb_checkpoint, load_wandb_dataset, mask_attn,
                         merge_dict, parse_args, save_model, hypo2utterance, get_sim_chk, SimLoss, get_domain_accuracy,
                         set_seed)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.wandb_logging import ListenerLogger


def normalize_aux(aux, data_length, epoch=""):
    mean_s = mean([len(x) for x in aux['sim_list_accuracy']])
    accs = mean([x[-1] == 1 for x in aux['sim_list_accuracy']])
    aux['loss'] = mean([x[-1] for x in aux.pop('losses')])

    aux['accs'] = accs
    aux['mean_s'] = mean_s

    # best attempt to tie accuracy to mean_s
    s_norm = 1 - mean_s / common_p.s_iter
    aux['s_acc'] = s_norm * accs

    # add hypo table
    nlp = spacy.load("en_core_web_lg")
    hypos = aux.pop('hypos')
    for idx in range(len(hypos)):
        h = hypos[idx]
        # estimate dissimilarity between first and last hypo
        if len(h) > 1:

            doc1 = nlp(str(h[0]))
            doc2 = nlp(str(h[-1]))
            sim = 1 - doc1.similarity(doc2)
        else:
            sim = 1

        if len(h) < common_p.s_iter + 1:
            hypos[idx] += ["/"] * (common_p.s_iter + 1 - len(hypos[idx]))
        hypos[idx] += [sim]

    columns = ["original_hypo"]
    columns += [f"hypo_{i}" for i in range(common_p.s_iter)]
    columns += ["similarity"]

    aux[f"hypo_table_epoch{epoch}"] = wandb.Table(columns, data=hypos)

    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["sim_list_loss"] = np.mean(aux["sim_list_loss"])

    # aux["sim_loss"] = np.mean(aux["sim_loss"])

    # a function to recursively flatten a list of lists into a single list
    def flatten(l):
        def inner():
            for el in l:
                if isinstance(el, list):
                    yield from flatten(el)
                else:
                    yield el

        return list(inner())

    aux["sim_list_accuracy"] = np.sum(flatten(aux["sim_list_accuracy"])) / data_length
    aux["list_target_accuracy"] = np.sum(flatten(aux["list_target_accuracy"])) / data_length
    aux["sim_target_accuracy"] = np.sum(flatten(aux["sim_target_accuracy"])) / data_length
    aux["sim_list_neg_accuracy"] = np.sum(flatten(aux["sim_list_neg_accuracy"])) / np.sum(flatten(aux["neg_pred_len"]))
    aux["sim_list_pos_accuracy"] = np.sum(flatten(aux["sim_list_pos_accuracy"])) / np.sum(flatten(aux["pos_pred_len"]))

    domains = flatten(aux.pop("domains"))
    aux["domain/list_target_acc"] = get_domain_accuracy(flatten(aux.pop('list_target_accuracy_dom')), domains,
                                                        logger.domains)
    aux["domain/sim_list_acc"] = get_domain_accuracy(flatten(aux.pop('sim_list_accuracy_dom')), domains, logger.domains)
    aux["domain/sim_target_acc"] = get_domain_accuracy(flatten(aux.pop('sim_target_accuracy_dom')), domains,
                                                       logger.domains)

    # flatten nested lists
    aux["sim_preds"] = flatten(aux['sim_preds'])
    aux["list_preds"] = flatten(aux['list_preds'])


def get_predictions(
        data: Dict,
        sim_model: torch.nn.Module,
        speak_model: torch.nn.Module,
        list_model: torch.nn.Module,
        optimizer: optim.Optimizer,
        criterion: SimLoss,
        adapt_lr: float,
        s_iter: int,
        list_vocab: Vocab,
) -> Tuple[torch.Tensor, int, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    targets = data["target"]
    prev_hist = data["prev_histories"]
    prev_utterance = data["prev_utterance"]
    prev_utt_lengths = data["prev_length"]
    target_img_feats = data["target_img_feats"]

    device = list_model.device

    max_length_tensor = prev_utterance.shape[1]
    speak_masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

    ################################################
    #   Get results with original hypo
    ################################################
    # generate hypothesis
    hypo, logs, decoder_hid = speak_model.generate_hypothesis(
        prev_utterance,
        prev_utt_lengths,
        context_concat,
        target_img_feats,
    )

    history_att = logs["history_att"]

    ################################################
    #   Get results with adapted hypo
    ################################################
    h0 = decoder_hid.clone().detach().requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.Adam([h0], lr=adapt_lr)
    else:
        optimizer.add_param_group({"params": h0, "lr": adapt_lr})

    losses = []
    hypos = []
    accs = []

    # perform loop
    i = 0
    infos = []
    while i < s_iter:
        set_seed(seed)

        # get modified hypo
        hypo = speak_model.nucleus_sampling(h0, history_att, speak_masks)
        hypos.append(hypo)

        # generate utt for list
        # translate utt to ids and feed to listener
        utterance = hypo2utterance(hypo, list_vocab)
        lengths = [utterance.shape[1]]
        max_length_tensor = utterance.shape[1]
        masks = mask_attn(lengths, max_length_tensor, device)

        list_out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks
        )

        sim_out = sim_model(h0, context_separate, context_concat, prev_hist, masks)

        # compute loss and perform backprop
        loss, info = criterion(sim_out, targets, list_out, data["domain"])
        losses.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

        # get  accuracy
        infos.append(info)

        # break if sim gets it right
        if info['sim_target_accuracy']:
            break
        i += 1

    aux = dict(
        hypos=hypos,
        losses=losses,
        accs=accs,

    )
    infos = merge_dict(infos)
    aux.update(infos)
    return mean(losses), info['sim_target_accuracy'], aux


def process_epoch(
        data_loader: DataLoader,
        sim_model: torch.nn.Module,
        speaker_model: torch.nn.Module,
        list_model: torch.nn.Module,
        optimizer: Optional[optim.Optimizer],
        list_vocab: Vocab,
        loss_f: torch.nn.Module,
        split: str,
        common_p,
) -> Dict:
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """

    for i, data in rich.progress.track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"{split} epoch {epoch}",
    ):
        if split == "train":
            sim_model.zero_grad()

        # get datapoints
        loss, accuracy, aux = get_predictions(
            data, sim_model, speaker_model, list_model, optimizer, loss_f, common_p.adapt_lr, common_p.s_iter,
            list_vocab
        )

        auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux, len(data_loader.dataset.data), epoch)

    return aux


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]

    speak_vocab = Vocab(parse_args("speak").vocab_file, is_speaker=True)

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="learning2stir",
    )

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(
        LISTENER_CHK_DICT[domain],
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

    # for reproducibility
    seed = list_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    model = get_model("list", list_args.model_type)
    list_model = model(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK,
        device,
    )  # datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1]))
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    # init speak model and load state

    model = get_model("speak", speak_p.model_type)
    speaker_model = model(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.beam_size,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
        use_beam=common_speak_p.use_beam,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################
    check = get_sim_chk(common_p.type_of_sim, common_p.model_type, common_p.pretrain_loss, domain)
    sim_check, _ = load_wandb_checkpoint(check, device)

    # load args
    sim_p = sim_check["args"]
    sim_p.train_domain = domain
    sim_p.device = device
    sim_p.resume_train = common_p.resume_train

    # for debug
    sim_p.subset_size = common_p.subset_size
    sim_p.debug = common_p.debug
    sim_p.pretrain_loss = common_p.pretrain_loss

    sim_p.reset_paths()

    model = get_model("sim", sim_p.model_type)
    sim_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
        sim_p.train_domain,
        sim_p.device,
    ).to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################
    optimizer = optim.Adam(sim_model.parameters(), lr=sim_p.learning_rate)
    cel = nn.CrossEntropyLoss(reduction=sim_p.reduction)
    kl_loss = nn.KLDivLoss(reduction=sim_p.reduction, log_target=True)
    loss_f = SimLoss(common_p.pretrain_loss, common_p.reduction, common_p.model_type,
                     alpha=common_p.focal_alpha, gamma=common_p.focal_gamma,
                     list_domain=domain, all_domains=logger.domains)

    ###################################
    ## RESUME
    ###################################

    if common_p.resume_train:
        sim_model.load_state_dict(sim_check["model_state_dict"])
        optimizer.load_state_dict(sim_check["optimizer_state_dict"])
        sim_model = sim_model.to(device)

    metric = sim_p.metric

    if metric == "loss":

        es = EarlyStopping(sim_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(sim_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model])

    ###################################
    ##  Get speaker dataloader
    ###################################
    sim_p.batch_size = 1

    training_loader, test_loader, val_loader = get_dataloaders(
        sim_p, speak_vocab, domain
    )

    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(sim_p.epochs):

        print("Epoch : ", epoch)

        auxs = []
        data = {}

        sim_model.train()
        sim_model.use_batchnorm(False)

        # torch.enable_grad()

        i = 0

        ###################################
        ##  TRAIN LOOP
        ###################################
        with torch.set_grad_enabled(True):
            aux = process_epoch(training_loader, sim_model, speaker_model, list_model, optimizer, list_vocab, loss_f,
                                "train",
                                common_p)

        logger.on_eval_end(
            aux, list_domain=training_loader.dataset.domain, modality="train"
        )

        print(
            f"Train loss {mean(aux['losses']):.6f}, accuracy {mean(aux['accs']):.4f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        sim_model.eval()

        for param in sim_model.parameters():
            param.requires_grad = False

        print(f"\nEvaluation")

        aux = process_epoch(val_loader, sim_model, speaker_model, list_model, None, list_vocab, loss_f, "eval",
                            common_p)

        logger.on_eval_end(
            aux, list_domain=val_loader.dataset.domain, modality="eval"
        )
        eval_loss = mean(aux['losses'])
        eval_accuracy = mean(aux['accs'])
        print(
            f"Eval loss {eval_loss:.6f}, accuracy {eval_accuracy:.4f} "
        )

        print(f"\nTest")
        aux = process_epoch(test_loader, sim_model, speaker_model, list_model, None, list_vocab, loss_f, "test",
                            common_p)

        logger.on_eval_end(
            aux, list_domain=test_loader.dataset.domain, modality="test"
        )

        print(
            f"Test loss {mean(aux['losses']):.6f}, accuracy {mean(aux['accs']):.4f} "
        )

        if not common_p.is_test:
            save_model(
                model=sim_model,
                model_type="Simulator",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
                args=sim_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if sim_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)