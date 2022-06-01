import datetime
import os

import numpy as np
import rich.progress
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from commons import (get_dataloaders, get_domain_accuracy,
                     load_wandb_checkpoint, mask_attn, save_model)
from data.dataloaders import Vocab
from models import ListenerModel, SimulatorModel
from wandb_logging import ListenerLogger


def evaluate(
    data_loader: DataLoader,
    sim_model: torch.nn.Module,
    list_model: torch.nn.Module,
    in_domain: bool,
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
    domains = []
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

        targets = data["target"]

        max_length_tensor = utterances.shape[1]

        masks = mask_attn(data["length"], max_length_tensor, device)

        prev_hist = data["prev_histories"]

        sim_out = sim_model(
            utterances, context_separate, context_concat, prev_hist, masks, device
        )
        list_out = list_model(
            utterances, context_separate, context_concat, prev_hist, masks, device
        )

        targets = targets.to(device)

        list_loss = criterion(list_out, targets)
        sim_loss = criterion(list_out, sim_out)
        loss = sim_loss

        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        sim_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)

        # accuracy
        correct = torch.eq(list_preds, sim_preds).sum()
        accuracies.append(float(correct))

        aux = dict(
            sim_preds=sim_preds.squeeze(),
            list_preds=list_preds.squeeze(),
            ranks=ranks,
            correct=correct / sim_preds.shape[0],
        )

        logger.on_batch_end(loss, data, aux, batch_id=ii, modality=flag)
        domains += data["domain"]

    if not in_domain:
        domain_accuracy = get_domain_accuracy(accuracies, domains, logger.domains)

    # normalize based on batches
    # domain_accuracy = {k: v / ii for k, v in domain_accuracy.items()}
    loss = np.mean(losses_eval)
    accuracy = np.mean(accuracies)

    metrics = dict(domain_accuracy=domain_accuracy, loss=loss)

    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=flag)

    return accuracy, loss


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_dim = 2048

    # listener dict
    listener_dict = dict(
        all="adaptive-speaker/listener/ListenerModel_all:v20",
        appliances="adaptive-speaker/listener/ListenerModel_appliances:v20",
        food="adaptive-speaker/listener/ListenerModel_food:v20",
        indoor="adaptive-speaker/listener/ListenerModel_indoor:v20",
        outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v20",
        vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v20",
    )

    domain = "all"

    list_checkpoint, _ = load_wandb_checkpoint(listener_dict[domain], device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.vocab_file = "vocab.csv"
    list_args.vectors_file = os.path.basename(list_args.vectors_file)
    list_args.device = device

    # for debug
    # list_args.subset_size = 10

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
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    sim_model = SimulatorModel(
        len(vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
    ).to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    learning_rate = list_args.learning_rate
    optimizer = optim.Adam(sim_model.parameters(), lr=learning_rate)

    reduction_method = list_args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)

    # add debug label
    tags = []
    if list_args.debug or list_args.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=vocab,
        opts=vars(list_args),
        group=list_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="speaker-list-dom",
    )
    t = datetime.datetime.now()

    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(list_args["epochs"]):

        print("Epoch : ", epoch)

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases
            training_loader, _, val_loader, _ = get_dataloaders(
                list_args, vocab, domain
            )

        losses = []
        accuracies = []

        sim_model.train()
        torch.enable_grad()

        count = 0

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.track(
            enumerate(training_loader),
            total=len(training_loader),
            description="Training",
        ):
            # get datapoints
            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            utterance = data["utterance"]
            lengths = [utterance.shape[1]]
            targets = data["target"]
            prev_hist = data["prev_histories"]

            max_length_tensor = utterance.shape[1]
            masks = mask_attn(lengths, max_length_tensor, device)

            # get outputs
            list_out = list_model(
                utterance, context_separate, context_concat, prev_hist, masks, device
            )

            sim_out = sim_model(
                utterance, context_separate, context_concat, prev_hist, masks, device
            )

            targets = targets.to(device)

            # Losses and preds

            list_loss = criterion(list_out, targets)
            sim_loss = criterion(list_out, sim_out)
            loss = sim_loss

            list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
            sim_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)

            # accuracy
            correct = torch.eq(list_preds, sim_preds).sum()
            accuracies.append(float(correct))

            # optimizer
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # logs
            logger.on_batch_end(
                loss,
                data,
                aux={
                    "list_preds": list_preds,
                    "sim_preds": sim_preds,
                    "list_loss": list_loss,
                },
                batch_id=i,
                modality="train",
            )

        losses = np.mean(losses)
        print("Train loss sum", round(losses, 5))  # sum all the batches for this epoch

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            isValidation = True
            print(f'\nVal Eval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                val_loader, sim_model, in_domain=True
            )

            save_model(
                model=sim_model,
                model_type="Simulator",
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
