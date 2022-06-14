import datetime

import numpy as np
import rich.progress
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from src.commons import (LISTENER_CHK_DICT, SIM_DOMAIN_CHK, SPEAKER_CHK,
                         get_dataloaders, hypo2utterance,
                         load_wandb_checkpoint, mask_attn, parse_args)
from src.data.dataloaders import Vocab
from src.models import ListenerModel_hist, SimulatorModel
from src.models.speaker.model_speaker_hist_att import SpeakerModel
from src.wandb_logging import ListenerLogger


def evaluate(
    data_loader: DataLoader,
    speak_model: SpeakerModel,
    sim_model: SimulatorModel,
    list_model: ListenerModel_hist,
    criterion,
    split:str,
    lr:float=0.1
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """
    original_accs = []
    original_hypos = []

    modified_accs = []
    modified_hypos = []

    for ii, data in rich.progress.track(
        enumerate(data_loader),
        total=len(data_loader),
        description=f"Evaluating on split {split}",
    ):

        ## extract data
        context_separate = data["separate_images"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        context_concat = data["concat_context"]
        target_img_feats = data["target_img_feats"]
        targets = data["target"]
        prev_hist = data["prev_histories"]

        # generate hypothesis
        hypo, logs, decoder_hid = speak_model.generate_hypothesis(
            prev_utterance,
            prev_utt_lengths,
            context_concat,
            target_img_feats,
        )

        original_hypos.append(hypo)

        ################################################
        #   Get results with original hypo
        ################################################
        # translate utt to ids and feed to listener
        utterance = hypo2utterance(hypo, speak_model.vocab)
        lengths = [utterance.shape[1]]
        max_length_tensor = utterance.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        list_out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks
        )

        # get  accuracy
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
        original_accs.append(list_target_accuracy)

        ################################################
        #   Get results with adapted hypo
        ################################################
        h0 = decoder_hid.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([h0], lr=lr)

        sim_out = sim_model(h0, context_separate, context_concat, prev_hist, masks)

        # compute loss and perform backprop
        loss = criterion(sim_out, targets)
        loss.backward()
        optimizer.step()

        # get modified hypo
        history_att = logs["history_att"]
        max_length_tensor = prev_utterance.shape[1]
        masks = mask_attn(prev_utt_lengths, max_length_tensor, device)
        hypo = speak_model.beam_serach(h0, history_att, masks)
        modified_hypos.append(hypo)

        # translate utt to ids and feed to listener
        utterance = hypo2utterance(hypo, speak_model.vocab)
        lengths = [utterance.shape[1]]
        max_length_tensor = utterance.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        list_out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks
        )

        # get  accuracy
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
        modified_accs.append(list_target_accuracy)

    # make wandb table
    data = list(
        list(
            zip(
                original_hypos,
                modified_hypos,
                original_accs,
                modified_accs,
                [x != y for x, y in zip(original_accs, modified_accs)],
            )
        )
    )
    table_columns = [
        "original hypo",
        "adapted_hypo",
        "original guess",
        "adapted guess",
        "diff",
    ]
    talbe = wandb.Table(columns=table_columns, data=data)

    original_accs = np.mean(original_accs)
    modified_accs = np.mean(modified_accs)

    metrics = dict(
        original_accs=original_accs,
        modified_accs=modified_accs,
        hypo_table=talbe,
    )

    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=split)

    return original_accs


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(LISTENER_CHK_DICT[domain], device)
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

    list_model = ListenerModel_hist(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)

    # init speak model and load state
    speaker_model = SpeakerModel(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        speak_p.beam_size,
        speak_p.max_len,
        device=device,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################
    sim_check, _ = load_wandb_checkpoint(SIM_DOMAIN_CHK[domain], device)
    # load args
    sim_p = sim_check["args"]
    sim_p.train_domain = domain
    sim_p.device = device

    # for debug
    sim_p.subset_size = common_p.subset_size
    sim_p.debug = common_p.debug

    sim_p.reset_paths()

    sim_model = SimulatorModel(
        len(list_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
        sim_p.device,
    ).to(device)

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(sim_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="speaker-influence",
    )

    metric = sim_p.metric

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = sim_p.batch_size
    # need batchsize =1 for generating hypothesis
    sim_p.batch_size = 1
    training_loader, _, val_loader = get_dataloaders(sim_p, speak_vocab, domain)

    ###################################
    ##  LOSS
    ###################################

    cel = nn.CrossEntropyLoss(reduction=sim_p.reduction)

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    ###################################
    ##  EVAL LOOP
    ###################################

    sim_model.eval()

    print(f"\nEvaluation on train")
    evaluate(
        training_loader,
        speaker_model,
        sim_model,
        list_model,
        criterion=cel,
        split="train",
        lr=common_p.learning_rate
    )

    print(f"\nEvaluation on eval")
    evaluate(
        val_loader,
        speaker_model,
        sim_model,
        list_model,
        criterion=cel,
        split="val",
        lr=common_p.learning_rate

    )

    logger.on_train_end({},epoch_id=0)
