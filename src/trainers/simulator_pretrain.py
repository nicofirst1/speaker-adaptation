import datetime
import operator
from typing import Dict, Tuple

import numpy as np
import rich.progress
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from src.commons import (DATASET_CHK, LISTENER_CHK_DICT, SIM_ALL_CHK,
                         SPEAKER_CHK, EarlyStopping, get_dataloaders,
                         load_wandb_checkpoint, load_wandb_dataset, mask_attn,
                         merge_dict, parse_args, save_model)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.wandb_logging import ListenerLogger


def normalize_aux(aux, data_length, max_targets=3):
    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["sim_list_loss"] = np.mean(aux["sim_list_loss"])
    aux["sim_loss"] = np.mean(aux["sim_loss"])

    aux["sim_list_accuracy"] = np.sum(aux["sim_list_accuracy"]) / data_length

    aux["list_target_accuracy"] = np.sum(aux["list_target_accuracy"]) / data_length

    aux["sim_target_accuracy"] = np.sum(aux["sim_target_accuracy"]) / data_length


    # flatten nested lists
    aux["sim_preds"] = [x for xs in aux["sim_preds"] for x in xs]
    aux["list_preds"] = [x for xs in aux["list_preds"] for x in xs]

    if len(aux["target"]) > max_targets:
        aux["target"] = np.random.choice(aux["target"], size=max_targets, replace=False).tolist()


def get_predictions(
    data: Dict,
    list_model: torch.nn.Module,
    sim_model: torch.nn.Module,
    criterion: torch.nn.Module,
    cel: torch.nn.Module,
    list_vocab: Vocab,
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
    batch_size = utterance.shape[0]

    masks = mask_attn(lengths, max_length_tensor, device)

    # get outputs
    list_out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

    sim_out = sim_model(speak_embds, context_separate, context_concat, prev_hist, masks)

    targets = targets.to(device)

    # Losses and preds
    list_loss = cel(list_out, targets)
    sim_list_loss = cel(sim_out, list_out)
    sim_loss = cel(sim_out, targets)
    loss = sim_list_loss

    list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
    sim_preds = torch.argmax(sim_out.squeeze(dim=-1), dim=1)

    # accuracy
    targets = targets.squeeze()
    sim_list_accuracy = torch.eq(list_preds, sim_preds).sum()
    list_target_accuracy = torch.eq(list_preds, targets).sum()
    sim_target_accuracy = torch.eq(sim_preds, targets).sum()

    sim_list_accuracy = sim_list_accuracy.tolist()
    list_target_accuracy = list_target_accuracy.tolist()
    sim_target_accuracy = sim_target_accuracy.tolist()

    list_preds = list_preds.tolist()
    sim_preds = sim_preds.tolist()

    # logging
    rnd_idx = np.random.randint(0, batch_size)
    hypo = list_vocab.decode(utterance[rnd_idx])
    caption = data["orig_utterance"][rnd_idx]
    target = data["image_set"][rnd_idx][data["target"][rnd_idx]]
    target = logger.img_id2path[str(target)]
    target = wandb.Image(target, caption=f"Hypo:{hypo}\nCaption : {caption}")

    # if split=="eval":
    #     for rnd_idx in range(batch_size):
    #         hypo=list_vocab.decode(utterance[rnd_idx])
    #         caption=data['orig_utterance'][rnd_idx]
    #         target=data['image_set'][rnd_idx][data['target'][rnd_idx]]
    #         target=logger.img_id2path[str(target)]
    #         target=wandb.Image(target, caption=f"Hypo:{hypo}\nCaption : {caption}")
    #         d={k:v[rnd_idx:rnd_idx+1] for k,v in data.items()}
    #         show_img(d, logger.img_id2path,f"modified_train", hypo=hypo)
    #         a=1

    aux = dict(
        sim_preds=sim_preds,
        list_preds=list_preds,
        sim_list_accuracy=sim_list_accuracy,
        list_target_accuracy=list_target_accuracy,
        sim_target_accuracy=sim_target_accuracy,
        list_loss=list_loss.detach().cpu().item(),
        sim_list_loss=sim_list_loss.detach().cpu().item(),
        sim_loss=sim_loss.detach().cpu().item(),
        target=target,
    )

    return loss, sim_list_accuracy, aux


def evaluate(
    data_loader: DataLoader,
    sim_model: torch.nn.Module,
    list_model: torch.nn.Module,
    list_vocab: Vocab,
    split: str,
):
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """

    auxs = []

    flag = f"{split}"

    for ii, data in rich.progress.track(
        enumerate(data_loader),
        total=len(data_loader),
        description=f"{split} epoch {epoch}",
    ):
        loss, accuracy, aux = get_predictions(
            data, list_model, sim_model, criterion, cel, list_vocab=list_vocab
        )

        auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux, len(data_loader.dataset.data))

    logger.on_eval_end(aux, list_domain=data_loader.dataset.domain, modality=flag)

    return aux["sim_list_accuracy"], aux["sim_list_loss"]


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(
        LISTENER_CHK_DICT[domain],
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
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

    ##########################
    # SIMULATOR
    ##########################
    if common_p.resume_train:
        sim_check, _ = load_wandb_checkpoint(SIM_ALL_CHK, device)
        # load args
        sim_p = sim_check["args"]
        sim_p.train_domain = domain
        sim_p.device = device
        sim_p.resume_train = common_p.resume_train

        # for debug
        sim_p.subset_size = common_p.subset_size
        sim_p.debug = common_p.debug

        sim_p.reset_paths()

    else:
        sim_p = common_p

    model = get_model("sim", sim_p.model_type)
    sim_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
        sim_p.train_domain,
        sim_p.device,
    ).to(device)

    if common_p.resume_train:
        sim_model.load_state_dict(sim_check["model_state_dict"])
        sim_model = sim_model.to(device)

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
        project="simulator-pretrain",
    )

    metric = sim_p.metric

    if metric == "loss":

        es = EarlyStopping(sim_p.patience, operator.ge)
    elif metric == "accs":
        es = EarlyStopping(sim_p.patience, operator.le)
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model])

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(sim_model.parameters(), lr=sim_p.learning_rate)
    cel = nn.CrossEntropyLoss(reduction=sim_p.reduction)
    criterion = nn.KLDivLoss(reduction=sim_p.reduction)

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    sim_p.batch_size = 1
    sim_p.shuffle = False

    shuffle = common_p.shuffle
    training_loader, test_loader, val_loader = get_dataloaders(
        sim_p, speak_vocab, domain
    )

    if common_p.is_test:
        training_loader = []
        sim_p.epochs = 1

    load_params = {
        "batch_size": bs,
        "shuffle": shuffle,
        "collate_fn": AbstractDataset.get_collate_fn(
            speaker_model.device,
            list_vocab["<sos>"],
            list_vocab["<eos>"],
            list_vocab["<nohs>"],
        ),
    }

    speak_train_dl = load_wandb_dataset(
        "train",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        DATASET_CHK,
        subset_size=common_p.subset_size,
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
        domain,
        load_params,
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        DATASET_CHK,
        subset_size=common_p.subset_size,
    )
    speak_test_dl = load_wandb_dataset(
        "test",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        DATASET_CHK,
        subset_size=common_p.subset_size,
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

        auxs = []
        data = {}

        sim_model.train()
        # torch.enable_grad()

        i = 0

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.progress.track(
            enumerate(speak_train_dl),
            total=len(speak_train_dl),
            description=f"Training epoch {epoch}",
        ):
            # get datapoints
            loss, accuracy, aux = get_predictions(
                data, list_model, sim_model, criterion, cel, list_vocab
            )

            auxs.append(aux)

            # optimizer
            sim_model.zero_grad()
            loss.backward()
            optimizer.step()

        aux = merge_dict(auxs)
        normalize_aux(aux,len(speak_train_dl.dataset.data))
        logger.on_eval_end(aux, list_domain=speak_train_dl.dataset.domain, modality="train")


        print(
            f"Train loss {aux['sim_list_loss']:.6f}, accuracy {aux['sim_list_accuracy']:.3f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            isValidation = True
            print(f"\nEvaluation")
            eval_accuracy, eval_loss = evaluate(
                speak_val_dl, sim_model, list_model, list_vocab, split="eval"
            )

            print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy:.3f} ")

            print(f"\nTest")
            test_accuracy, test_loss = evaluate(
                speak_test_dl, sim_model, list_model, list_vocab, split="test"
            )
            print(f"Test loss {test_loss:.6f}, accuracy {test_accuracy:.3f} ")

        if not common_p.is_test:
            save_model(
                model=sim_model,
                model_type="Simulator",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
                args=sim_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if sim_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
