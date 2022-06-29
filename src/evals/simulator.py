import datetime
import operator
from typing import Dict, Tuple

import numpy as np
import rich.progress
import torch
from rich.pretty import pprint
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from src.commons import (DATASET_CHK, LISTENER_CHK_DICT, SIM_ALL_CHK,
                         SIM_DOMAIN_CHK, SPEAKER_CHK, EarlyStopping,
                         get_dataloaders, load_wandb_checkpoint,
                         load_wandb_dataset, mask_attn, merge_dict, parse_args,
                         save_model)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.trainers.simulator_pretrain import (evaluate, get_predictions,
                                             normalize_aux)
from src.wandb_logging import ListenerLogger

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
    else:
        sim_check, _ = load_wandb_checkpoint(SIM_DOMAIN_CHK[domain], device)

    # load args
    sim_p = sim_check["args"]
    sim_p.train_domain = domain
    sim_p.device = device
    sim_p.resume_train = common_p.resume_train
    sim_p.test_split = common_p.test_split

    # for debug
    sim_p.subset_size = common_p.subset_size
    sim_p.debug = common_p.debug

    sim_p.reset_paths()

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
        project="simulator-eval",
    )

    ###################################
    ##  LOSS
    ###################################

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
        val_loader,
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
        test_loader,
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

    def print_aux(split: str, aux: Dict):
        """
        Print stats about the results contained in aux
        """
        from rich import print

        accuracy, loss = aux["sim_list_accuracy"], aux["sim_list_loss"]
        split_len = len(aux.pop("list_preds"))
        aux.pop("sim_preds")
        print(
            f"\n\n{split} loss {loss:.6f}, accuracy {accuracy:.3f}, split-len {split_len}\nAux: {aux}\n"
        )

    with torch.no_grad():
        sim_model.eval()

        ###########################
        #   TRAIN SPLIT
        ###########################
        split = "train"
        aux = evaluate(
            speak_train_dl,
            sim_model,
            list_model,
            list_vocab,
            split=split,
            cel=cel,
            kl=criterion,
        )
        logger.on_eval_end(aux, list_domain=speak_val_dl.dataset.domain, modality=split)
        print_aux(split, aux)

        ###########################
        #   EVAL SPLIT
        ###########################
        split = "eval"

        aux = evaluate(
            speak_val_dl,
            sim_model,
            list_model,
            list_vocab,
            split=split,
            cel=cel,
            kl=criterion,
        )
        logger.on_eval_end(aux, list_domain=speak_val_dl.dataset.domain, modality=split)
        print_aux(split, aux)

        ###########################
        #   TEST SPLIT
        ###########################
        split = "test"

        aux = evaluate(
            speak_test_dl,
            sim_model,
            list_model,
            list_vocab,
            split=split,
            cel=cel,
            kl=criterion,
        )

        logger.on_eval_end(
            aux, list_domain=speak_test_dl.dataset.domain, modality=split
        )
        print_aux(split, aux)

        logger.on_train_end({}, epoch_id=0)
