import operator
import sys
from os.path import abspath, dirname

import numpy as np
import torch
import torch.utils.data
from rich.progress import track
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from src.commons import (LISTENER_CHK_DICT, EarlyStopping, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint, mask_attn,
                         parse_args, save_model)
from src.data.dataloaders import Vocab
from src.models import ListenerModel_hist, ListenerModel_no_hist, get_model
from src.wandb_logging import DataLogger, ListenerLogger

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

global logger


def evaluate(
    data_loader: DataLoader, model: torch.nn.Module, in_domain: bool, split: str
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
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
        # print(i)

        count += 1

        utterances = data["utterance"]

        context_separate = data["separate_images"]
        context_concat = data["concat_context"]

        lengths = data["length"]
        targets = data["target"]

        max_length_tensor = utterances.shape[1]

        masks = mask_attn(data["length"], max_length_tensor, args.device)

        prev_hist = data["prev_histories"]

        out = model(utterances, context_separate, context_concat, prev_hist, masks)

        targets = targets.to(args.device)
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

    args = parse_args("list")

    print(args)

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )
    print("code starts", timestamp)

    domain = args.train_domain

    model_type = args.model_type

    # for reproducibilty
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # print("Loading the vocab...")
    vocab = Vocab(args.vocab_file, is_speaker=False)
    vocab_size = len(vocab)
    # print(f'vocab size {vocab_size}')

    # add debug label
    tags = []
    if args.debug or args.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=vocab,
        opts=vars(args),
        group=args.train_domain,
        train_logging_step=20,
        val_logging_step=1,
        tags=tags,
    )

    ###################################
    ##  DATALOADERS
    ###################################

    if "vectors.json" in args.vectors_file:  # from resnet
        img_dim = 2048
    elif "clip.json" in args.vectors_file:
        img_dim = 512
    else:
        raise KeyError(f"No valid image vector for file '{args.vectors_file}'")

    training_loader, test_loader, val_loader = get_dataloaders(
        args, vocab, domain, unary_val_bs=False
    )
    _, test_loader_speaker, val_loader_speaker = get_dataloaders(
        args, vocab, domain="all", unary_val_bs=False
    )

    if args.log_data:
        # log dataset once
        data_logger = DataLogger(
            vocab=vocab,
            opts=vars(args),
            tags=tags,
        )
        data_logger.log_dataset(training_loader.dataset, "train")
        data_logger.log_dataset(val_loader.dataset, "val")
        print("Dataset logged")

    ###################################
    ##  MODEL
    ###################################

    if args.embed_type == "scratch":
        embedding_dim = args.embed_dim  # gave 768, like BERT

    hidden_dim = args.hidden_dim
    att_dim = args.attention_dim

    dropout_prob = args.dropout_prob

    # depending on the selected model type, we will have a different architecture

    model = get_model("list", model_type)
    model = model(
        vocab_size,
        embedding_dim,
        hidden_dim,
        img_dim,
        att_dim,
        dropout_prob,
        args.train_domain,
        device=args.device,
    ).to(args.device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reduction_method = args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)

    ###################################
    ##  RESTORE MODEL
    ###################################

    if args.resume_train:
        checkpoint, file = load_wandb_checkpoint(
            LISTENER_CHK_DICT[args.train_domain], args.device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        epoch = checkpoint["epoch"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed run at epoch {epoch}")

    if args.is_test:
        training_loader = []
        args.epochs = 1

    logger.watch_model([model])

    ###################################
    ##  EPOCHS START
    ###################################

    metric = args.metric

    if metric == "loss":

        es = EarlyStopping(args.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(args.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    print("training starts", timestamp)

    for epoch in range(args.epochs):

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases
            training_loader, test_loader, val_loader = get_dataloaders(
                args, vocab, domain, unary_val_bs=False
            )

            if args.is_test:
                training_loader = []

        losses = []
        data = {}
        preds = []

        model.train()
        torch.enable_grad()

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in track(
            enumerate(training_loader),
            total=len(training_loader),
            description=f"Training epoch {epoch}",
        ):
            # collect info from datapoint
            utterances = data["utterance"]
            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            lengths = data["length"]
            targets = data["target"]
            prev_hist = data["prev_histories"]

            max_length_tensor = utterances.shape[1]
            masks = mask_attn(lengths, max_length_tensor, args.device)

            out = model(utterances, context_separate, context_concat, prev_hist, masks)

            model.zero_grad()

            # TARGETS SUITABLE FOR CROSS-ENTROPY LOSS
            targets = targets.to(args.device)
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
            model.eval()

            print(f'\nEval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                val_loader, model, in_domain=True, split="eval"
            )

            print(f"\nEval on all domains")
            evaluate(val_loader_speaker, model, in_domain=False, split="eval")

            print(f'\nTest on domain "{domain}"')
            evaluate(test_loader, model, in_domain=True, split="test")

            print(f"\nTest on all domains")
            evaluate(test_loader_speaker, model, in_domain=False, split="test")

            if not args.is_test:
                save_model(
                    model=model,
                    model_type=model_type,
                    epoch=epoch,
                    accuracy=current_accuracy,
                    optimizer=optimizer,
                    args=args,
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
