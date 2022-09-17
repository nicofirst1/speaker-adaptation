import datetime
from typing import Dict, Tuple

import numpy as np
import rich.progress
import torch
from torch import nn

import wandb
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, get_dataloaders,
                         get_domain_accuracy, hsja, load_wandb_checkpoint,
                         mask_attn, merge_dict, parse_args, speak2list_vocab,
                         translate_utterance, set_seed)
from src.data.dataloaders import Vocab
from src.models import get_model
from src.wandb_logging import ListenerLogger
from torch.utils.data import DataLoader


def get_mask(utts):
    lengths = []
    batch_size = utts.shape[0]
    for idx in range(batch_size):
        nz = torch.nonzero(utts[idx])
        if len(nz) > 1:
            nz = torch.max(nz) + 1
        else:
            nz = 1
        lengths.append(nz)

    max_len = max(lengths)
    masks = mask_attn(lengths, max_len, device)

    return masks


def normalize_aux(aux):
    diffs_hypo = [
        (a, b, c, d)
        for a, b, c, d in zip(
            aux["origin_h"], aux["mod_h"], aux["origin_acc"], aux["mod_acc"]
        )
        if a != b
    ]

    columns = ["origin_h", "mod_h", "origin_acc", "mod_acc"]
    table = wandb.Table(columns=columns, data=diffs_hypo)
    aux["modified_hypos"] = table

    domains = aux.pop("domain")

    aux["domain/mod_accs"] = get_domain_accuracy(
        aux.pop("mod_acc"), domains, logger.domains
    )
    aux["domain/origin_accs"] = get_domain_accuracy(
        aux.pop("origin_acc"), domains, logger.domains
    )


def get_predictions(
    data: Dict,
    speak_model: torch.nn.Module,
    list_model: torch.nn.Module,
    list_vocab: Vocab,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/int out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    targets = data["target"]
    prev_hist = data["prev_histories"]
    prev_utterance = data["prev_utterance"]
    prev_utt_lengths = data["prev_length"]
    target_img_feats = data["target_img_feats"]
    batch_size = context_concat.size(0)
    device = target_img_feats.device

    max_length_tensor = prev_utterance.shape[1]
    speak_masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

    ################################################
    #   Get results with original hypo
    ################################################
    # generate hypothesis
    utts, logs, decoder_hid = speak_model.generate_hypothesis(
        prev_utterance,
        prev_utt_lengths,
        context_concat,
        target_img_feats,
    )

    #translator(utts)
    origin_h = [list_vocab.decode(sent) for sent in utts]
    mod_utts = origin_h

    history_att = logs["history_att"]

    ################################################
    #   Get results with adapted hypo
    ################################################
    h0 = decoder_hid.clone().detach()

    # get lengths
    masks = get_mask(utts)

    if batch_size == 1:
        masks = masks.squeeze(1)

    list_out = list_model(utts, context_separate, context_concat, prev_hist, masks)

    preds = torch.argmax(list_out, dim=1)
    correct = torch.eq(preds, targets).sum()
    mod_correct = correct

    if not correct:


        def hsja_model(context_separate, context_concat, prev_hist):
            def inner(h0):

                # get modified hypo
                utts = speak_model.nucleus_sampling(h0, history_att, speak_masks)
                translator(utts)
                tmp = [list_vocab.decode(sent) for sent in utts]

                masks = get_mask(utts)
                batch_size = utts.shape[0]

                if batch_size == 1:
                    masks = masks.squeeze(1)[0]
                    ncs = context_separate
                    ncc = context_concat
                else:
                    ncs = context_separate.repeat(batch_size, 1, 1)
                    ncc = context_concat.repeat(batch_size, 1)

                return list_model(utts, ncs, ncc, prev_hist, masks)

            return inner

        model_fn = hsja_model(context_separate, context_concat, prev_hist)
        h0 = hsja(
            model_fn,
            h0,
            clip_max=1,
            clip_min=-1,
            constraint="l2",
            num_iterations=5,
            gamma=1.0,
            target_label=targets,
            target_image=None,
            stepsize_search="geometric_progression",
            max_num_evals=70,
            init_num_evals=100,
            verbose=True,
            device=device,
        )

        utts = speak_model.nucleus_sampling(h0, history_att, speak_masks)
        translator(utts)

        mod_utts = [list_vocab.decode(sent) for sent in utts]

        masks = get_mask(utts)

        list_out = list_model(utts, context_separate, context_concat, prev_hist, masks)
        preds = torch.argmax(list_out, dim=1)
        mod_correct = torch.eq(preds, targets).sum()

    aux = dict(
        origin_h=origin_h[0],
        mod_h=mod_utts[0],
        origin_acc=correct.detach().item(),
        mod_acc=mod_correct.detach().item(),
        domain=data["domain"][0],
    )

    return aux


def process_epoch(
    data_loader: DataLoader,
    speaker_model: torch.nn.Module,
    list_model: torch.nn.Module,
    list_vocab: Vocab,
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
    auxs = []
    with torch.no_grad():
        for i, data in rich.progress.track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"{split}",
        ):
            # get datapoints
            aux = get_predictions(data, speaker_model, list_model, list_vocab)

            auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux)

    return aux


if __name__ == "__main__":

    img_dim = 2048

    common_p = parse_args("int")
    domain = common_p.train_domain
    device = common_p.device

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
        project="adversarial_adapt",
    )

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(
        LISTENER_CHK_DICT[domain],
        device,
    )
    list_args = list_checkpoint["args"]

    # update list args
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

    # for reproducibility
    seed = list_args.seed
    set_seed(seed)

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


    ###################################
    ##  Get speaker dataloader
    ###################################

    data_domain = common_p.data_domain
    common_p.batch_size = 1

    training_loader, test_loader, val_loader = get_dataloaders(
        common_p, speak_vocab, "all"
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

    ###################################
    ##  TRAIN LOOP
    ###################################

    # aux = process_epoch(
    #     training_loader, speaker_model, list_model, list_vocab, "train", common_p
    # )
    #
    # logger.on_eval_end(
    #     aux,
    #     list_domain=training_loader.dataset.domain,
    #     modality="train",
    #     commit=True,
    # )

    ###################################
    ##  EVAL LOOP
    ###################################

    print(f"\nEvaluation")

    aux = process_epoch(
        val_loader, speaker_model, list_model, list_vocab, "eval", common_p
    )

    logger.on_eval_end(
        aux,
        list_domain=val_loader.dataset.domain,
        modality="eval",
        commit=True,
    )

    print(f"\nTEST")

    aux = process_epoch(
        test_loader, speaker_model, list_model, list_vocab, "test", common_p
    )

    logger.on_eval_end(
        aux,
        list_domain=test_loader.dataset.domain,
        modality="test",
        commit=True,
    )

    logger.on_train_end({}, epoch_id=0)
