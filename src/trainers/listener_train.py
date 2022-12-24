import sys
from os.path import abspath, dirname, join
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import wandb
from rich.progress import track
from torch import nn, optim
from torch.utils.data import DataLoader

from src.commons import (LISTENER_CHK_DICT, EarlyStopping, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint, mask_attn,
                         parse_args, save_model, SPEAKER_CHK, speak2list_vocab, translate_utterance, merge_dict)
from src.commons.data_utils import load_wandb_dataset
from src.data.dataloaders import Vocab, AbstractDataset
from src.models import ListenerModel, SpeakerModel
from src.wandb_logging import DataLogger, ListenerLogger

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

global logger


def get_prediction(data, use_golden, list_model, translator):
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    targets = data["target"]
    prev_hist = data["prev_histories"]

    if use_golden:
        utterances = data["utterance"]
        lengths = data["length"]

    else:
        utterances = data["speak_utterance"]
        translator(utterances)
        lengths = data["speak_length"]

    max_length_tensor = utterances.shape[1]
    masks = mask_attn(lengths, max_length_tensor, list_args.device)
    out = list_model(utterances, context_separate, context_concat, prev_hist, masks)

    targets = targets.to(list_args.device)
    loss = criterion(out, targets)

    preds = torch.argmax(out, dim=1).squeeze(dim=-1)
    targets = targets.squeeze(dim=-1)

    correct = torch.eq(targets, preds)
    accuracy = correct.sum() / preds.shape[0]

    scores_ranked, images_ranked = torch.sort(out.squeeze(), descending=True)

    if out.shape[0] > 1:
        for s in range(out.shape[0]):
            # WARNING - assumes batch size > 1
            rank_target = images_ranked[s].tolist().index(targets[s].item()) + 1

    else:
        rank_target = images_ranked.tolist().index(targets.item()) + 1

    aux = dict(
        preds=preds.squeeze(),
        ranks=rank_target,
        scores_ranked=scores_ranked,
        images_ranked=images_ranked,
        accuracy=accuracy,
        correct=correct,

    )

    # logger.on_batch_end(loss, data, aux, batch_id=ii, modality=flag)

    return loss, aux


def evaluate(
        data_loader: DataLoader, list_model: torch.nn.Module, in_domain: bool, split: str, use_golden: Dict, translator
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model: listener model
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """
    losses_eval = []
    domains = []
    auxs = []
    domain_accuracy = {}
    flag = f"{split}/"
    flag += "in_domain" if in_domain else "out_domain"

    for ii, data in track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"{flag}"):
        loss, aux = get_prediction(data, use_golden[ii], list_model, translator)
        losses_eval.append(loss.item())
        domains += data["domain"]
        auxs.append(aux)

    aux = merge_dict(auxs)

    if not in_domain:
        domain_accuracy = get_domain_accuracy(aux['correct'], domains, logger.domains)
        ood_ranks=[aux['ranks'][idx] for idx, d in enumerate(domains) if d!=list_model.domain]
        print(domain_accuracy)

    # normalize based on batches
    # domain_accuracy = {k: v / ii for k, v in domain_accuracy.items()}
    loss = np.mean(losses_eval)
    accuracy = np.mean(aux["accuracy"])
    MRR = np.sum([1 / r for r in aux["ranks"]]) / len(aux["ranks"])

    metrics = dict(mrr=MRR, loss=loss, accuracy=accuracy, preds=torch.stack(aux["preds"]))
    if len(domain_accuracy) > 0:
        metrics["domain_accuracy"] = domain_accuracy
        metrics['ood_mrr'] = np.sum([1 / r for r in ood_ranks]) / len(ood_ranks)

    # logger.log_datapoint(data, preds, modality="eval")
    # logger.log_viz_embeddings(data, modality="eval")
    print(f"{flag} loss {loss:.3f}, accuracy {accuracy:.3f}")

    logger.on_eval_end(metrics, list_domain=list_model.domain, modality=flag)

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

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    # load golden dataloader

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

    # load spekaer augmented dataloaders

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
    speak_val_dl_domain = load_wandb_dataset(
        "val",
        data_domain,
        load_params,
        list_vocab,
        speaker_model,
        val_loader,
        logger,
        subset_size=list_args.subset_size,
    )

    speak_val_dl_all = load_wandb_dataset(
        "val",
        "all",
        load_params,
        list_vocab,
        speaker_model,
        val_loader_speaker,
        logger,
        subset_size=list_args.subset_size,
    )


    def rnd():
        return torch.FloatTensor(1).uniform_(0, 1)


    # define percentage of data to be used
    use_golden_train = {k: rnd() for k in range(len(speak_train_dl))}
    use_golden_train = {k: (v < list_args.golden_data_perc).item() for k, v in use_golden_train.items()}

    use_golden_val_domain = {k: rnd() for k in range(len(speak_val_dl_domain))}
    use_golden_val_domain = {k: (v < list_args.golden_data_perc).item() for k, v in use_golden_val_domain.items()}

    use_golden_val_all = {k: rnd() for k in range(len(speak_val_dl_all))}
    use_golden_val_all = {k: (v < list_args.golden_data_perc).item() for k, v in use_golden_val_all.items()}

    if list_args.log_data:
        # log dataset once
        data_logger = DataLogger(
            vocab=list_vocab,
            opts=vars(list_args),
        )
        data_logger.log_dataset(speak_train_dl.dataset, "train")
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
        accuracies = []
        ranks = []
        auxs = []

        listener_model.train()
        torch.enable_grad()

        # randomize order of data
        speak_train_dl.dataset.randomize_target_location()
        speak_val_dl_domain.dataset.randomize_target_location()
        speak_val_dl_all.dataset.randomize_target_location()


        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in track(
                enumerate(speak_train_dl),
                total=len(speak_train_dl),
                description=f"Training epoch {epoch}",
        ):
            optimizer.zero_grad()

            loss, aux = get_prediction(data, use_golden_train[i], listener_model, translator)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accuracies.append(aux["accuracy"])
            ranks.append(aux["ranks"])
            auxs.append(aux)

        aux = merge_dict(auxs)
        losses = np.mean(losses)
        accuracies = np.mean(aux["accuracy"])
        ranks = np.mean(aux["ranks"])
        aux = dict(
            loss=losses,
            accuracy=accuracies,
            ranks=ranks,
            preds=torch.cat(aux["preds"]),
        )
        print(f"Train loss {losses:.3f}, accuracy {accuracies:.3f}")
        # logger.log_datapoint(data, preds, modality="train")
        # logger.log_viz_embeddings(data, modality="train")
        logger.on_eval_end(aux, listener_model.domain, "train")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            listener_model.eval()

            print(f'\nEval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                speak_val_dl_domain, listener_model, in_domain=True, split="eval", use_golden=use_golden_val_domain,
                translator=translator)

            print(f"\nEval on all domains")
            evaluate(speak_val_dl_all, listener_model, in_domain=False, split="eval", use_golden=use_golden_val_all,
                     translator=translator)
            if epoch % 2 == 0:
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
            print("\n\n")

        logger.on_train_end({"loss": losses}, epoch_id=epoch)

        # check for early stopping
        metric_val = current_loss if metric == "loss" else current_accuracy
        if es.should_stop(metric_val):
            break

    wandb.finish()
