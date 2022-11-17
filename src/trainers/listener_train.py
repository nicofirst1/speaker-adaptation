import sys
from os.path import abspath, dirname, join
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import wandb
from rich.progress import track
from src.commons import (LISTENER_CHK_DICT, EarlyStopping, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint, mask_attn,
                         parse_args, save_model, SPEAKER_CHK)
from src.commons.data_utils import MixedDataTrainer, load_wandb_dataset
from src.data.dataloaders import Vocab, AbstractDataset
from src.models import get_model
from src.models.listener.ListenerModel import ListenerModel
from src.models.speaker.SpeakerModel import SpeakerModel
from src.wandb_logging import DataLogger, ListenerLogger
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

global logger


def evaluate(
    data_loader: DataLoader, model: torch.nn.Module, in_domain: bool, split: str, use_golden:Dict
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model: listener model
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """
    losses_eval = []
    accuracies = []
    ranks = []
    corrects = []
    domains = []
    count = 0

    domain_accuracy = {}
    flag = f"{split}/"
    flag += "in_domain" if in_domain else "out_domain"

    for ii, data in enumerate(data_loader):

        count += 1

        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        targets = data["target"]
        prev_hist = data["prev_histories"]


        if use_golden[ii]:
            utterances = data["utterance"]
        else:
            utterances = data["speak_utterance"]

        max_length_tensor = utterances.shape[1]
        masks = mask_attn(data["length"], max_length_tensor, list_args.device)

        out = model(utterances, context_separate, context_concat, prev_hist, masks)

        targets = targets.to(list_args.device)
        loss = criterion(out, targets)
        losses_eval.append(loss.item())

        preds = torch.argmax(out, dim=1).squeeze(dim=-1)
        targets = targets.squeeze(dim=-1)

        correct = torch.eq(targets, preds)
        corrects += correct.tolist()
        accuracy = correct.sum() / preds.shape[0]
        accuracies.append(accuracy.cpu())

        scores_ranked, images_ranked = torch.sort(out.squeeze(), descending=True)

        if out.shape[0] > 1:
            for s in range(out.shape[0]):
                # WARNING - assumes batch size > 1
                rank_target = images_ranked[s].tolist().index(targets[s].item())
                ranks.append(rank_target + 1)  # no 0

        else:
            rank_target = images_ranked.tolist().index(targets.item())
            ranks.append(rank_target + 1)  # no 0

        aux = dict(
            preds=preds.squeeze(),
            ranks=ranks,
            scores_ranked=scores_ranked,
            images_ranked=images_ranked,
            accuracy=accuracy,
        )

        logger.on_batch_end(loss, data, aux, batch_id=ii, modality=flag)
        domains += data["domain"]

    if not in_domain:
        domain_accuracy = get_domain_accuracy(corrects, domains, logger.domains)

    # normalize based on batches
    # domain_accuracy = {k: v / ii for k, v in domain_accuracy.items()}
    loss = np.mean(losses_eval)
    accuracy = np.mean(accuracies)

    MRR = np.sum([1 / r for r in ranks]) / len(ranks)

    print(
        "Acc",
        round(accuracy, 5),
        "Loss",
        round(loss, 5),
        "MRR",
        round(MRR, 5),
    )

    metrics = dict(mrr=MRR, loss=loss)
    if len(domain_accuracy) > 0:
        metrics["domain_accuracy"] = domain_accuracy

    logger.log_datapoint(data, preds, modality="eval")
    # logger.log_viz_embeddings(data, modality="eval")
    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=flag)

    return accuracy, loss, MRR


if __name__ == "__main__":

    list_args = parse_args("list")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    domain = list_args.train_domain


    # for reproducibilty
    torch.manual_seed(list_args.seed)
    np.random.seed(list_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # print("Loading the vocab...")
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)
    vocab_size = len(list_vocab)
    # print(f'vocab size {vocab_size}')


    logger = ListenerLogger(
        vocab=list_vocab,
        opts=vars(list_args),
        group=list_args.train_domain,
        train_logging_step=20,
        val_logging_step=1,
    )

    if "vectors.json" in list_args.vectors_file:  # from resnet
        img_dim = 2048
    elif "clip.json" in list_args.vectors_file:
        img_dim = 512
    else:
        raise KeyError(f"No valid image vector for file '{list_args.vectors_file}'")

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK,
        device,
     datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1]))
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    # init speak model and load state

    speaker_model = SpeakerModel(
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
    ##  LISTENER MODEL
    ###################################


    listener_model = ListenerModel(
        vocab_size,
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device=list_args.device,
    ).to(list_args.device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    learning_rate = list_args.learning_rate
    optimizer = optim.Adam(listener_model.parameters(), lr=learning_rate)

    reduction_method = list_args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)

    ###################################
    ##  RESTORE MODEL
    ###################################

    if list_args.resume_train:
        checkpoint, file = load_wandb_checkpoint(
            LISTENER_CHK_DICT[list_args.train_domain], list_args.device
        )

        listener_model.load_state_dict(checkpoint["model_state_dict"])
        listener_model = listener_model.to(list_args.device)
        epoch = checkpoint["epoch"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed run at epoch {epoch}")

    if list_args.is_test:
        training_loader = []
        list_args.epochs = 1

    logger.watch_model([listener_model])

    ###################################
    ##  DATALOADERS
    ###################################


    training_loader, test_loader, val_loader = get_dataloaders(
        list_args, list_vocab, domain, unary_val_bs=False
    )
    _, test_loader_speaker, val_loader_speaker = get_dataloaders(
        list_args, list_vocab, domain="all", unary_val_bs=False
    )
    bs = list_args.batch_size
    data_domain = list_args.data_domain


    load_params = {
        "batch_size": bs,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": AbstractDataset.get_collate_fn(
            speaker_model.device,
            list_vocab["<sos>"],
            list_vocab["<eos>"],
            list_vocab["<nohs>"],
        ),
    }

    speak_train_dl = load_wandb_dataset(
        "train",
        data_domain,
        load_params,
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        subset_size=list_args.subset_size,
    )

    load_params = {
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": AbstractDataset.get_collate_fn(
            speaker_model.device,
            list_vocab["<sos>"],
            list_vocab["<eos>"],
            list_vocab["<nohs>"],
        ),
    }
    speak_val_dl = load_wandb_dataset(
        "val",
        data_domain,
        load_params,
        list_vocab,
        speaker_model,
        val_loader,
        logger,
        subset_size=list_args.subset_size,
    )

    use_golden_train = {k: torch.randn((1,)) for k in range(len(speak_train_dl))}
    use_golden_train = {k: (v > list_args.golden_data_perc).item() for k, v in use_golden_train.items()}

    use_golden_val = {k: torch.randn((1,)) for k in range(len(speak_train_dl))}
    use_golden_val = {k: (v > list_args.golden_data_perc).item() for k, v in use_golden_val.items()}


    if list_args.log_data:
        # log dataset once
        data_logger = DataLogger(
            vocab=list_vocab,
            opts=vars(list_args),
        )
        data_logger.log_dataset(training_loader.dataset, "train")
        data_logger.log_dataset(val_loader.dataset, "val")
        print("Dataset logged")

    ###################################
    ##  EPOCHS START
    ###################################

    metric = list_args.metric

    if metric == "loss":

        es = EarlyStopping(list_args.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(list_args.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    print("training starts", timestamp)

    for epoch in range(list_args.epochs):


        losses = []
        data = {}
        preds = []

        listener_model.train()
        torch.enable_grad()

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in track(
            enumerate(speak_train_dl),
            total=len(speak_train_dl),
            description=f"Training epoch {epoch}",
        ):
            # collect info from datapoint
            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            lengths = data["length"]
            targets = data["target"]
            prev_hist = data["prev_histories"]


            # get speaker/golden data
            if use_golden_train[i]:

                utterances = data["utterance"]

            else:
                utterances = data["speak_utterance"]

            max_length_tensor = utterances.shape[1]
            masks = mask_attn(lengths, max_length_tensor, list_args.device)
            out = listener_model(utterances, context_separate, context_concat, prev_hist, masks)

            listener_model.zero_grad()

            # TARGETS SUITABLE FOR CROSS-ENTROPY LOSS
            targets = targets.to(list_args.device)
            loss = criterion(out, targets)

            targets = targets.squeeze()
            preds = torch.argmax(out, dim=1).squeeze(dim=-1)
            accuracy = torch.eq(targets, preds).sum() / preds.shape[0]

            aux = dict(preds=preds, accuracy=accuracy)
            logger.on_batch_end(loss, data, aux=aux, batch_id=i, modality="train")

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        losses = np.mean(losses)
        print("Train loss sum", round(losses, 5))  # sum all the batches for this epoch
        logger.log_datapoint(data, preds, modality="train")
        # logger.log_viz_embeddings(data, modality="train")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            listener_model.eval()

            print(f'\nEval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                speak_val_dl, listener_model, in_domain=True, split="eval",use_golden=use_golden_val )

            print(f"\nEval on all domains")
            evaluate(speak_val_dl, listener_model, in_domain=False, split="eval")


            if not list_args.is_test:
                save_model(
                    model=listener_model,
                    model_type="listener",
                    epoch=epoch,
                    accuracy=current_accuracy,
                    optimizer=optimizer,
                    args=list_args,
                    timestamp=timestamp,
                    logger=logger,
                    loss=current_loss,
                    mrr=current_MRR,
                )

        logger.on_train_end({"loss": losses}, epoch_id=epoch)

        # check for early stopping
        metric_val = current_loss if metric == "loss" else current_accuracy
        if es.should_stop(metric_val):
            break

    wandb.finish()
