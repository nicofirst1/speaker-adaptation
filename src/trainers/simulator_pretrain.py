import datetime
import operator
import os

import numpy as np
import rich.progress
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.commons import (get_dataloaders, get_domain_accuracy,
                         load_wandb_checkpoint, mask_attn, save_model, LISTENER_CHK_DICT, SPEAKER_CHK, EarlyStopping,
                         parse_args)
from src.data.dataloaders import Vocab
from src.models import ListenerModel, SimulatorModel
from src.models.speaker.model_speaker_hist_att import SpeakerModelHistAtt
from src.wandb_logging import ListenerLogger


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
    domain = "all"

    sim_p = parse_args("sim")


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
    list_args.subset_size = sim_p.subset_size
    list_args.debug = sim_p.debug

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
        device=device
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
    speaker_model = SpeakerModelHistAtt(
        vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        device=device
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################

    sim_model = SimulatorModel(
        len(vocab),
        sim_p.embed_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
    ).to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(sim_model.parameters(), lr=sim_p.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction=sim_p.reduction_method)

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if sim_p.debug or sim_p.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=vocab,
        opts=vars(sim_p),
        group=sim_p.train_domain,
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

    t = datetime.datetime.now()

    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    for epoch in range(list_args["epochs"]):

        print("Epoch : ", epoch)

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases
            training_loader, _, val_loader, _ = get_dataloaders(
                sim_p, vocab, domain
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
            prev_utterance_ids = data["prev_utterance"]
            prev_lengths = data["prev_length"]
            target_img_feats = data["target_img_feats"]

            max_length_tensor = utterance.shape[1]
            masks = mask_attn(lengths, max_length_tensor, device)

            # get outputs
            list_out = list_model(
                utterance, context_separate, context_concat, prev_hist, masks, device
            )

            _, speak_embds = speaker_model.simulator_forward(
                utterance,
                lengths,
                prev_utterance_ids,
                prev_lengths,
                context_separate,
                context_concat,
                target_img_feats,
                targets,
                prev_hist,
                prev_lengths,
                normalize=False,
                masks=masks,
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
                args=sim_p,
                timestamp=timestamp,
                logger=logger,
                loss=current_loss,
                mrr=current_MRR,
            )

            # check for early stopping
            metric_val = current_loss if sim_p.metric == "loss" else current_accuracy
            if es.should_stop(metric_val): break

        logger.on_train_end({"loss": losses}, epoch_id=epoch)
