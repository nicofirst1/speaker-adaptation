import sys
from os.path import abspath, dirname

import numpy as np
import torch
import torch.utils.data
import wandb
from rich.progress import track
from torch import nn, optim
from torch.utils.data import DataLoader

from data.dataloaders import Vocab, get_dataloaders
from models.listener.model_listener import ListenerModel
from trainers.parsers import parse_args
from trainers.utils import mask_attn
from wandb_logging import DataLogger, ListenerLogger, save_model

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime
import os

global logger

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")


def evaluate(
        data_loader: DataLoader, model: torch.nn.Module, in_domain: bool
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

    count = 0

    domain_accuracy = {}
    flag = "eval/"
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

        masks = mask_attn(data["length"], max_length_tensor, device)

        prev_hist = data["prev_histories"]

        out = model(
            utterances, context_separate, context_concat, prev_hist, masks, device
        )

        targets = targets.to(device)
        loss = criterion(out, targets)
        losses_eval.append(loss.item())

        preds = torch.argmax(out, dim=1)

        correct = torch.eq(preds, targets).sum()
        accuracies.append(float(correct) / len(preds))

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
            correct=correct / preds.shape[0],
        )

        logger.on_batch_end(loss, data, aux, batch_id=ii, modality=flag)

        if not in_domain:
            # estimate per domain accuracy
            tmp = logger.get_domain_accuracy(data, preds)
            for k in tmp.keys():
                if k not in domain_accuracy.keys():
                    domain_accuracy[k] = 0

                domain_accuracy[k] += tmp[k]

    # normalize based on batches
    domain_accuracy = {k: v / ii for k, v in domain_accuracy.items()}
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

    metrics = dict(mrr=MRR, domain_accuracy=domain_accuracy, loss=loss)

    logger.log_datapoint(data, preds, modality="eval")
    logger.log_viz_embeddings(data, modality="eval")
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
    vocab = Vocab(args.vocab_file)
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

    training_loader, test_loader, val_loader, _ = get_dataloaders(args, vocab, domain)
    _, _, val_loader_speaker, _ = get_dataloaders(args, vocab, domain="speaker")

    if args.log_data:
        # log dataset once
        data_logger = DataLogger(
            vocab=vocab,
            opts=vars(args),
            tags=tags,
        )
        data_logger.log_dataset(training_loader.dataset, "train")
        data_logger.log_dataset(test_loader.dataset, "test")
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

    if model_type == "scratch_rrr":  # embeds from scratch, visual context + hist
        model = ListenerModel(
            vocab_size, embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob
        ).to(device)

    logger.watch_model([model])

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reduction_method = args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)

    ###################################
    ##  EPOCHS START
    ###################################

    batch_size = args.batch_size
    metric = args.metric
    shuffle = args.shuffle

    epochs = 100
    patience = 50  # when to stop if there is no improvement
    patience_counter = 0

    best_loss = float("inf")
    best_accuracy = -1
    best_mrr = -1

    prev_loss = float("inf")
    prev_accuracy = -1
    prev_mrr = -1

    best_epoch = -1

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    print("training starts", timestamp)

    for epoch in range(epochs):

        print("Epoch : ", epoch)

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases
            training_loader, _, val_loader, _ = get_dataloaders(args, vocab, domain)

        losses = []

        model.train()
        torch.enable_grad()

        count = 0

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in track(
                enumerate(training_loader),
                total=len(training_loader),
                description="Training",
        ):



            # print(count)
            count += 1

            utterances = data["utterance"]

            context_separate = data["separate_images"]
            context_concat = data["concat_context"]

            lengths = data["length"]
            targets = data["target"]

            max_length_tensor = utterances.shape[1]

            masks = mask_attn(data["length"], max_length_tensor, device)

            prev_hist = data["prev_histories"]

            out = model(
                utterances, context_separate, context_concat, prev_hist, masks, device
            )

            model.zero_grad()

            # targets = torch.tensor([[torch.argmax(tg)] for tg in targets]).to(device)
            # TARGETS SUITABLE FOR CROSS-ENTROPY LOSS
            targets = targets.to(device)
            loss = criterion(out, targets)

            preds = torch.argmax(out.squeeze(dim=-1), dim=1)
            logger.on_batch_end(
                loss, data, aux={"preds": preds}, batch_id=i, modality="train"
            )

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        losses = np.mean(losses)
        print("Train loss sum", round(losses, 5))  # sum all the batches for this epoch
        logger.log_datapoint(data, preds, modality="train")
        logger.log_viz_embeddings(data, modality="train")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            model.eval()

            isValidation = True
            print(f'\nVal Eval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                val_loader, model, in_domain=True
            )

            print(f"\nVal Eval on all domains")
            evaluate(val_loader_speaker, model, in_domain=False)

            if metric == "loss":

                if best_loss <= current_loss:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_loss = current_loss
                    best_epoch = epoch

                    save_model(
                        model,
                        model_type,
                        best_epoch,
                        current_accuracy,
                        current_loss,
                        current_MRR,
                        optimizer,
                        args,
                        "loss",
                        timestamp,
                        args.seed,
                        t,
                    )

                print("patience", patience_counter, "\n")

                print("\nBest", best_epoch, round(best_loss, 5), metric)  # validset
                print()

            elif metric == "accs":

                if best_accuracy >= current_accuracy:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_accuracy = current_accuracy
                    best_epoch = epoch

                    save_model(
                        model=model,
                        model_type=model_type,
                        epoch=best_epoch,
                        accuracy=current_accuracy,
                        optimizer=optimizer,
                        args=args,
                        timestamp=timestamp,
                        logger=logger,
                        loss=current_loss,
                        mrr=current_MRR,
                    )

                print("patience", patience_counter)

                print(
                    "\nBest", best_epoch, round(best_accuracy, 5), metric, "\n"
                )  # validset

            elif metric == "mrr":

                if best_mrr >= current_MRR:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_mrr = current_MRR
                    best_epoch = epoch

                    save_model(
                        model,
                        model_type,
                        best_epoch,
                        current_accuracy,
                        current_loss,
                        current_MRR,
                        optimizer,
                        args,
                        "mrr",
                        timestamp,
                        args.seed,
                        t,
                    )

                print("patience", patience_counter)

                print(
                    "\nBest", best_epoch, round(best_mrr, 5), metric, "\n"
                )  # validset

            prev_accuracy = current_accuracy
            prev_loss = current_loss
            prev_mrr = current_MRR

        logger.on_train_end({"loss": losses}, epoch_id=epoch)

    wandb.finish()
