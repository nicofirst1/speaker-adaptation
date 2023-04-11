import datetime
from typing import Dict, Tuple

import lovely_tensors as lt
import numpy as np
import rich.progress
import torch
import wandb
from torch import nn, optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.commons import (SPEAKER_CHK, EarlyStopping,
                         get_listener_check, load_wandb_checkpoint,
                         mask_attn, mask_oov_embeds,
                         merge_dict, parse_args, save_model, set_seed,
                         speak2list_vocab, translate_utterance)
from src.commons.Baseline import MeanBaseline
from src.data.dataloaders import Vocab
from src.data.dataloaders.EcDataset import EcDataset
from src.models import ListenerModel, SpeakerModelEC
from src.wandb_logging import ListenerLogger

global common_p
global list_vocab


def normalize_aux(aux, logger, all_domains, max_targets=3):
    batch_size = len(aux["target_id"][0])
    aux["loss"] = np.mean(aux["loss"])
    aux["policy_loss"] = np.mean(aux["policy_loss"])
    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["entropy_loss"] = np.mean(aux["entropy_loss"])
    list_acc=aux["list_acc"]
    aux["list_acc"] = np.mean(aux["list_acc"]) / batch_size
    aux["baseline"] = np.mean(aux["baseline"])
    aux['enc_log_probs'] = torch.stack(aux['enc_log_probs']).mean(dim=0).mean(dim=0).numpy()
    aux['dec_log_probs'] = torch.stack(aux['dec_log_probs']).mean(dim=0).mean(dim=0).numpy()

    # remove nans from log probs
    aux['enc_log_probs'] = np.nan_to_num(aux['enc_log_probs'])
    aux['dec_log_probs'] = np.nan_to_num(aux['dec_log_probs'])

    # get max targets random ids in range of targets
    target_ids = np.random.choice(range(len(aux["target_id"])), max_targets)

    idx = 0
    for i in target_ids:
        t_id = aux["target_id"][i]
        utt = aux["utterance"][i]

        jdx = np.random.choice(range(len(t_id)))
        t_id = t_id[jdx]
        utt = utt[jdx]

        t_id = logger.img_id2path[t_id]

        img = wandb.Image(t_id, caption=utt)

        aux[f"img_{idx}"] = img
        idx += 1

    del aux["target_id"]
    del aux["utterance"]


def get_predictions(
        data: Dict,
        list_model: ListenerModel,
        speak_model: SpeakerModelEC,
        loss_f: nn.CrossEntropyLoss,
        translator,
        baseline: MeanBaseline
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """
    global common_p
    global list_vocab

    # get datapoints
    context_separate = data["image_set"]
    target = data["target_index"]
    image_ids = data["image_ids"]
    target_id = data["target_id"]
    target_img_feat = data["target_img_feat"]

    hypos, model_params, _ = speak_model.generate_hypothesis(context_separate, target_img_feat)

    utterance = hypos
    translator(utterance)
    lengths = torch.tensor([len(x) for x in utterance])
    max_length_tensor = torch.max(lengths).item()
    # get mask and translate utterance
    masks = mask_attn(lengths, max_length_tensor, list_model.device)

    # get outputs
    list_out = list_model(utterance, context_separate, masks)
    list_out = list_out.squeeze(-1)

    # Losses and preds
    list_preds = torch.argmax(list_out, dim=1)
    list_acc = list_preds.eq(target).sum().item()
    list_loss = loss_f(list_out, target).mean()

    bs = baseline.predict(list_loss.detach())
    enc_logits = model_params['encoder_logits']
    dec_logits = model_params['decoder_logits']
    enc_log_probs = torch.log_softmax(enc_logits, dim=-1)
    dec_log_probs = torch.log_softmax(dec_logits, dim=-1)

    # try to implement entropy loss here
    # https://github.com/facebookresearch/EGG/blob/18d72d86cf9706e7ad82f94719b56accd288e59a/egg/zoo/compo_vs_generalization_ood/archs.py#L144

    if common_p.use_enc_logits:
        policy_loss = (list_loss.detach() - bs) * enc_log_probs
        distr= Categorical(logits=enc_logits)
    else:
        policy_loss = (list_loss.detach() - bs) * dec_log_probs
        distr= Categorical(logits=dec_logits)

    policy_loss = policy_loss.mean()

    entropy=distr.entropy()
    entropy_loss=- entropy.mean()

    loss = list_loss + policy_loss + entropy_loss * common_p.entropy_loss_weight

    baseline.update(list_loss.detach())
    dec_utt = list_vocab.batch_decode(utterance.squeeze())

    aux = dict(
        loss=loss.detach().cpu().item(),
        policy_loss=policy_loss.detach().cpu().item(),
        list_loss=list_loss.detach().cpu().item(),
        entropy_loss=entropy_loss.detach().cpu().item(),
        baseline=bs.detach().cpu().item(),

        utterance=dec_utt,
        target_id=target_id,
        list_acc=list_acc,
        enc_log_probs=enc_log_probs.detach().cpu().squeeze(),
        dec_log_probs=dec_log_probs.detach().cpu().squeeze(),


    )

    return loss, aux


def evaluate(
        data_loader: DataLoader,
        speak_model: SpeakerModelEC,
        list_model: ListenerModel,
        translator,
        baseline: MeanBaseline,
        loss_f: torch.nn.Module,
        split: str,
) -> Dict:
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """

    auxs = []
    episodes = common_p.episodes

    for idx in rich.progress.track(
            range(episodes),
            total=episodes,
            description=f"evaluating '{split}' split...",
    ):
        data = next(iter(data_loader))
        loss, aux = get_predictions(
            data, list_model, speak_model, loss_f, translator, baseline
        )

        auxs.append(aux)

    aux = merge_dict(auxs)

    return aux


def get_kwargs(split, common_p):
    kwargs = {
        "device": common_p.device,
        "episodes": common_p.episodes,
        "domain": common_p.train_domain,
        "utterances_file": f"{split}_{common_p.utterances_file}",
        "vectors_file": common_p.vectors_file,
        "chain_file": f"{split}_{common_p.chains_file}",
        "orig_ref_file": f"{split}_{common_p.orig_ref_file}",
        "split": split,
        "subset_size": common_p.subset_size,
        "image_size": common_p.image_size,
        "img2dom_file": common_p.img2dom_file,
        "data_dir": common_p.data_path,
        "batch_size": common_p.batch_size,
    }
    return kwargs


def main():
    lt.monkey_patch()



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048
    global common_p
    global list_vocab

    common_p = parse_args("list")
    domain = common_p.train_domain

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        project="ec_pretrain",
    )

    ##########################
    # LISTENER
    ##########################

    list_check = get_listener_check(domain, common_p.golden_data_perc)

    list_checkpoint, _ = load_wandb_checkpoint(
        list_check,
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.device = device
    list_args.reset_paths()

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    list_model = ListenerModel(
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

    speaker_model = SpeakerModelEC(
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

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.AdamW(
        speaker_model.parameters(), lr=common_p.learning_rate, weight_decay=0.0001
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=common_p.epochs // 10, factor=0.5, verbose=True, threshold=0.5
    )
    loss_f = nn.CrossEntropyLoss(reduction="none")

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

    logger.watch_model([speaker_model], log_freq=100)

    ###################################
    ##  Get  dataloader
    ###################################
    # need batchsize =1 for generating the new dataloaders
    data_domain = common_p.data_domain

    kwargs = get_kwargs("train", common_p)
    dataset = EcDataset(**kwargs)
    dataloader_train = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    kwargs = get_kwargs("val", common_p)
    dataset = EcDataset(**kwargs)
    dataloader_eval = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(common_p.epochs):

        print("Epoch : ", epoch)

        auxs = []

        speaker_model.train()
        dataloader_train.dataset.randomize_data()
        dataloader_eval.dataset.randomize_data()

        # torch.enable_grad()
        ###################################
        ##  TRAIN LOOP
        ###################################
        baseline = MeanBaseline()

        for idx in rich.progress.track(
                range(common_p.episodes),
                total=common_p.episodes,
                description=f"Training epoch {epoch}",
        ):
            optimizer.zero_grad()

            data = next(iter(dataloader_train))

            # get datapoints
            loss, aux = get_predictions(
                data,
                list_model,
                speaker_model,
                loss_f,
                translator,
                baseline
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            nn.utils.clip_grad_value_(speaker_model.parameters(), clip_value=1.0)
            optimizer.step()

        aux = merge_dict(auxs)
        aux['lr'] = optimizer.param_groups[0]['lr']

        normalize_aux(aux, logger, all_domains=logger.domains)
        logger.on_eval_end(
            aux, list_domain=data_domain, modality="train"
        )

        print(
            f"Train loss {aux['loss']:.6f}, accuracy {aux['list_acc'] * 100:.2f}% "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            speaker_model.eval()

            print(f"\nEvaluation")
            aux = evaluate(
                dataloader_eval,
                speaker_model,
                list_model,
                translator,
                baseline,
                loss_f,
                split="eval",
            )
            normalize_aux(aux, logger, all_domains=logger.domains)

            eval_accuracy, eval_loss = aux["list_acc"], aux["loss"]

            scheduler.step(eval_accuracy)

            print(
                f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% "
            )
            logger.on_eval_end(
                aux, list_domain=data_domain, modality="eval"
            )

        if epoch > 0 and epoch % 50 == 0:
            save_model(
                model=speaker_model,
                model_type="speaker_ec",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
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


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, finishing run")
        wandb.finish()
