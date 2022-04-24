import argparse
import sys
from os.path import abspath, dirname
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
import wandb
from rich.progress import track
from torch import nn, optim
from torch.utils.data import DataLoader

from models.model_listener import ListenerModel
from utils.Vocab import Vocab
from wandb_logging.DataLogger import DataLogger
from wandb_logging.ListenerLogger import ListenerLogger

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime
import os

from utils.ListenerDataset import ListenerDataset

global logger

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")


def mask_attn(actual_num_tokens, max_num_tokens, device):
    masks = []

    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (
                max_num_tokens - actual_num_tokens[n]
        )
        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def save_model(
        model,
        model_type,
        epoch,
        accuracy,
        loss,
        mrr,
        optimizer,
        args,
        metric,
        timestamp,
        seed,
        t,
):
    file_name = (
            "saved_models/model_listener_adp_"
            + model_type
            + "_"
            + args.embed_type
            + "_CE_"
            + str(seed)
            + "_"
            + metric
            + "_"
            + timestamp
            + ".pkl"
    )

    print(file_name)

    duration = datetime.datetime.now() - t

    print("model saving duration", duration)

    torch.save(
        {
            "accuracy": accuracy,
            "mrr": mrr,
            "args": args,  # more detailed info, metric, model_type etc
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "seed": seed,
        },
        file_name,
    )

    wandb.save(file_name)


def evaluate(
        data_loader: DataLoader, breaking: bool, model: torch.nn.Module, in_domain: bool
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param breaking:
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

        if breaking and count == 5:
            break

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
            tmp = logger.log_domain_accuracy(data, preds)
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
        int(accuracy),
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


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train_domain",
        type=str,
        default="food",
        choices=[
            "appliances",
            "food",
            "indoor",
            "outdoor",
            "speaker",
            "vehicles",
            "all",
        ],
    )  # domain to train the listener on
    parser.add_argument(
        "-embed_type", type=str, default="scratch"
    )  # type of embeds to use
    parser.add_argument(
        "-embed_dim", type=int, default=768
    )  # if from scratch, 768 as BERT
    parser.add_argument("-data_path", type=str, default="../../data")
    parser.add_argument(
        "-vectors_file",
        type=str,
        default="vectors.json",
        choices=["vectors.json", "clip.json"],
    )  # or clip.json
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-model_type", type=str, default="scratch_rrr")
    parser.add_argument(
        "-subset_size", type=int, default=-1
    )  # -1 is the full dataset, if you put 10, it will only use 10 chains
    parser.add_argument("-shuffle", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-log_data", action="store_true")
    parser.add_argument("-breaking", action="store_true")
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-attention_dim", type=int, default=512)
    parser.add_argument("-hidden_dim", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-metric", type=str, default="accs", choices=["accs", "loss"])  # accs or loss
    parser.add_argument("-dropout_prob", type=float, default=0.0)
    parser.add_argument(
        "-reduction", type=str, default="sum", choices=['sum']
    )  # reduction for crossentropy loss
    parser.add_argument(
        "-wandb_dir", type=str, default="wandb_out"
    )

    return parser


def get_data_loaders(
        args: argparse.Namespace, domain: str, img_dim: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare the dataset and dataloader
    :param args: argparse args
    :param domain:
    :param img_dim:
    :return: train, test and eval dataloader
    """

    if domain == "all":
        domain = "speaker"

    trainset = ListenerDataset(
        data_dir=args.data_path,
        domain=domain,
        utterances_file="train_ids_utterances.pickle",
        vectors_file=args.vectors_file,
        chain_file="train_text_chains.json",
        split="train",
        subset_size=args.subset_size,
        image_size=img_dim,
    )

    testset = ListenerDataset(
        data_dir=args.data_path,
        domain=domain,
        utterances_file="test_ids_utterances.pickle",
        vectors_file=args.vectors_file,
        chain_file="test_text_chains.json",
        split="test",
        subset_size=args.subset_size,
        image_size=img_dim,
    )

    valset = ListenerDataset(
        data_dir=args.data_path,
        domain=domain,
        utterances_file="val_ids_utterances.pickle",
        vectors_file=args.vectors_file,
        chain_file="val_text_chains.json",
        split="val",
        subset_size=args.subset_size,
        image_size=img_dim,
    )

    # print('train len', len(trainset))
    # print('test len', len(testset))
    # print('val len', len(valset))

    load_params = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "collate_fn": ListenerDataset.get_collate_fn(device),
    }

    load_params_test = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": ListenerDataset.get_collate_fn(device),
    }

    training_loader = torch.utils.data.DataLoader(trainset, **load_params)

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

    if args.log_data:
        # log dataset once
        data_logger =DataLogger(
            vocab=vocab,
            opts=vars(args),
            tags=tags,
        )
        data_logger.log_dataset(trainset,"train")
        data_logger.log_dataset(testset,"test")
        data_logger.log_dataset(valset,"val")
        print("Dataset logged")


    return training_loader, test_loader, val_loader


if __name__ == "__main__":

    parser = arg_parse()
    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )
    print("code starts", timestamp)

    args = parser.parse_args()

    domain = args.train_domain

    model_type = args.model_type

    # for reproducibilty
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # print("Loading the vocab...")
    vocab = Vocab("../../data/" + args.vocab_file)
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

    if args.vectors_file == "vectors.json":  # from resnet
        img_dim = 2048
    elif args.vectors_file == "clip.json":
        img_dim = 512
    else:
        raise KeyError(f"No valid image vector for file '{args.vectors_file}'")

    training_loader, test_loader, val_loader = get_data_loaders(args, domain, img_dim)
    _, _, val_loader_speaker = get_data_loaders(args, "speaker", img_dim)

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
    breaking = args.breaking

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
            training_loader, _, val_loader = get_data_loaders(args, domain, img_dim)

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

            if breaking and count == 5:
                break

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
                val_loader, breaking, model, in_domain=True
            )

            print(f"\nVal Eval on all domains")
            evaluate(val_loader_speaker, breaking, model, in_domain=False)

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
                        model,
                        model_type,
                        best_epoch,
                        current_accuracy,
                        current_loss,
                        current_MRR,
                        optimizer,
                        args,
                        "accs",
                        timestamp,
                        args.seed,
                        t,
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
