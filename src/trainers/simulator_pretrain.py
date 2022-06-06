import datetime
import operator
import os
from typing import Dict, Tuple

import numpy as np
import rich.progress
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.commons import (
    LISTENER_CHK_DICT,
    SPEAKER_CHK,
    EarlyStopping,
    get_dataloaders,
    load_wandb_checkpoint,
    mask_attn,
    parse_args,
    save_model, SIM_CHK,
)
from src.commons.data_utils import speaker_augmented_dataloader
from src.data.dataloaders import Vocab
from src.models import ListenerModel, SimulatorModel
from src.models.speaker.model_speaker_hist_att import SpeakerModel
from src.wandb_logging import ListenerLogger


def get_predictions(
        data: DataLoader, list_model: ListenerModel, sim_model: SimulatorModel, criterion, cel
) -> Tuple[torch.Tensor, int, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    utterance = data["speak_utterance"]
    lengths = [utterance.shape[1]]
    targets = data["target"]
    prev_hist = data["prev_histories"]
    speak_embds = data["speak_h1embed"]
    max_length_tensor = utterance.shape[1]
    masks = mask_attn(lengths, max_length_tensor, device)

    # get outputs
    list_out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

    sim_out = sim_model(speak_embds, context_separate, context_concat, prev_hist, masks)

    targets = targets.to(device)

    # Losses and preds
    list_loss = cel(list_out, targets)
    sim_list_loss = criterion(list_out, sim_out)
    sim_loss = cel(sim_out, targets)
    loss = sim_list_loss

    list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
    sim_preds = torch.argmax(sim_out.squeeze(dim=-1), dim=1)

    # accuracy
    targets=targets.squeeze()
    sim_list_accuracy = torch.eq(list_preds, sim_preds).sum()/ sim_preds.shape[0]
    list_target_accuracy=torch.eq(list_preds, targets).sum()/ list_preds.shape[0]
    sim_target_accuracy=torch.eq(sim_preds, targets).sum()/ sim_preds.shape[0]

    sim_list_accuracy=sim_list_accuracy.item()
    list_target_accuracy=list_target_accuracy.item()
    sim_target_accuracy=sim_target_accuracy.item()

    list_preds = list_preds.tolist()
    sim_preds = sim_preds.tolist()

    aux = dict(
        sim_preds=sim_preds,
        list_preds=list_preds,

        sim_list_accuracy=sim_list_accuracy,
        list_target_accuracy=list_target_accuracy,
        sim_target_accuracy=sim_target_accuracy,

        list_loss=list_loss,
        sim_list_loss=sim_list_loss,
        sim_loss=sim_loss,
    )

    return loss, sim_list_accuracy, aux


def evaluate(
        data_loader: DataLoader,
        sim_model: torch.nn.Module,
        list_model: torch.nn.Module,
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """
    losses = []
    accuracies = []

    flag = "eval"

    for ii, data in enumerate(data_loader):
        loss, accuracy, aux = get_predictions(data, list_model, sim_model, criterion, cel)

        losses.append(loss.item())
        accuracies.append(accuracy)

        logger.on_batch_end(loss, data, aux, batch_id=ii, modality=flag)

    losses = np.mean(losses)
    accuracies = np.mean(accuracies)

    metrics = dict(accuracy=accuracies, loss=losses)

    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=flag)

    return accuracies, losses


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
    list_args.vocab_file = "vocab.csv"
    list_args.vectors_file = os.path.basename(list_args.vectors_file)
    list_args.device = device

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
    vocab = Vocab(list_args.vocab_file)

    list_model = ListenerModel(
        len(vocab),
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
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    # init speak model and load state
    speaker_model = SpeakerModel(
        vocab,
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
    if common_p.resume_train:
        sim_check, _ = load_wandb_checkpoint(SIM_CHK, device)
        # load args

        sim_p = sim_check["args"]
        sim_p.vocab_file = "vocab.csv"
        sim_p.train_domain = domain
        sim_p.__post_init__()
    else:
        sim_p=common_p


    sim_model = SimulatorModel(
        len(vocab),
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
        vocab=vocab,
        opts=vars(sim_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="simulator-pretrain",
    )

    metric = sim_p.metric

    if metric == "loss":

        es = EarlyStopping(sim_p.atience, operator.le)
    elif metric == "accs":
        es = EarlyStopping(sim_p.patience, operator.ge)
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model])

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(sim_model.parameters(), lr=sim_p.learning_rate)
    cel = nn.CrossEntropyLoss(reduction=sim_p.reduction)
    criterion= nn.KLDivLoss(reduction=sim_p.reduction)

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = sim_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    sim_p.batch_size = 1
    training_loader, _, val_loader, _ = get_dataloaders(sim_p, vocab, domain)

    speak_train_dl = speaker_augmented_dataloader(
        training_loader, vocab, speaker_model, batch_size=bs, split_name="train"
    )
    speak_val_dl = speaker_augmented_dataloader(
        val_loader, vocab, speaker_model, batch_size=bs, split_name="val"
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

        losses = []
        accuracies = []

        sim_model.train()
        torch.enable_grad()

        count = 0

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.progress.track(
                enumerate(speak_train_dl),
                total=len(speak_train_dl),
                description=f"Training epoch {epoch}",
        ):
            # get datapoints
            loss, accuracy, aux = get_predictions(data, list_model, sim_model, criterion, cel)

            losses.append(loss.item())
            accuracies.append(accuracy)

            # optimizer
            loss.backward()
            optimizer.step()

            # logs
            logger.on_batch_end(
                loss,
                data,
                aux=aux,
                batch_id=i,
                modality="train",
            )

        losses = np.mean(losses)
        accuracies = np.mean(accuracies)

        logger.on_batch_end(
            losses,
            data,
            aux=dict(accuracy=accuracies),
            batch_id=i,
            modality="train",
        )

        print(f"Train loss {losses:.6f}, accuracy {accuracies:.3f} ")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            isValidation = True
            print(f'\nEvaluation')
            current_accuracy, current_loss = evaluate(
                speak_val_dl, sim_model, list_model
            )

            print(f"Evaluation loss {current_loss:.6f}, accuracy {current_accuracy:.3f} ")

            save_model(
                model=sim_model,
                model_type="Simulator",
                epoch=epoch,
                accuracy=current_accuracy,
                optimizer=optimizer,
                args=sim_p,
                timestamp=timestamp,
                logger=logger,
                loss=current_loss,
            )

            # check for early stopping
            metric_val = current_loss if sim_p.metric == "loss" else current_accuracy
            if es.should_stop(metric_val):
                break

        logger.on_train_end({"loss": losses}, epoch_id=epoch)
