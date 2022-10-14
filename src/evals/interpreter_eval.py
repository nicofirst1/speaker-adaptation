import datetime
from typing import Dict

import numpy as np
import torch
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, AccuracyEstimator,
                         IntLossPretrain, get_dataloaders, get_int_chk,
                         load_wandb_checkpoint, load_wandb_dataset, parse_args)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.trainers.interpreter_pretrain import evaluate
from src.wandb_logging import ListenerLogger

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("int")
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
    # INTERPRETER
    ##########################

    check = get_int_chk(common_p.model_type, common_p.pretrain_loss, domain)
    int_check, _ = load_wandb_checkpoint(check, device)

    # load args
    int_p = int_check["args"]
    int_p.train_domain = domain
    int_p.device = device
    int_p.resume_train = common_p.resume_train
    int_p.test_split = common_p.test_split

    # for debug
    int_p.subset_size = common_p.subset_size
    int_p.debug = common_p.debug

    int_p.reset_paths()

    model = get_model("int", int_p.model_type)
    int_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        int_p.hidden_dim,
        img_dim,
        int_p.attention_dim,
        int_p.dropout_prob,
        int_p.train_domain,
        int_p.device,
    ).to(device)

    int_model.load_state_dict(int_check["model_state_dict"])
    int_model = int_model.to(device)

    ###################################
    ##  LOGGER
    ###################################

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(int_p),
        train_logging_step=1,
        project=f"speaker-eval-{common_p.type_of_int}",
        val_logging_step=1,
        tags=common_p.tags,
    )

    ###################################
    ##  LOSS
    ###################################

    loss_f = IntLossPretrain(
        common_p.pretrain_loss,
        common_p.reduction,
        common_p.model_type,
        alpha=common_p.focal_alpha,
        gamma=common_p.focal_gamma,
        list_domain=domain,
        all_domains=logger.domains,
    )
    acc_estimator = AccuracyEstimator(
        domain, common_p.model_type, all_domains=logger.domains
    )

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    int_p.batch_size = 1
    int_p.shuffle = False

    shuffle = common_p.shuffle
    training_loader, test_loader, val_loader = get_dataloaders(
        int_p, speak_vocab, domain
    )

    if common_p.is_test:
        training_loader = []
        int_p.epochs = 1

    # train
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
        subset_size=common_p.subset_size,
    )

    # eval

    speak_val_dl = load_wandb_dataset(
        "val",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        val_loader,
        logger,
        subset_size=common_p.subset_size,
    )

    # test
    speak_test_dl = load_wandb_dataset(
        "test",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        test_loader,
        logger,
        subset_size=common_p.subset_size,
        test_split=common_p.test_split,
    )

    ###################################
    ##  START OF EVALUATION
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

        accuracy, loss = aux["int_list_accuracy"], aux["loss"]
        split_len = len(aux.pop("list_preds"))
        aux.pop("int_preds")
        print(
            f"\n\n{split} loss {loss:.6f}, accuracy {accuracy:.3f}, split-len {split_len}\nAux: {aux}\n"
        )

    with torch.no_grad():
        int_model.eval()

        ###########################
        #   TRAIN SPLIT
        ###########################
        split = "train"
        aux = evaluate(
            speak_train_dl,
            int_model,
            list_model,
            list_vocab,
            loss_f=loss_f,
            acc_estimator=acc_estimator,
            all_domains=logger.domains,
            split=split,
        )
        logger.on_eval_end(aux, list_domain=speak_val_dl.dataset.domain, modality=split)
        print_aux(split, aux)

        ###########################
        #   EVAL SPLIT
        ###########################
        split = "eval"

        aux = evaluate(
            speak_val_dl,
            int_model,
            list_model,
            list_vocab,
            loss_f=loss_f,
            acc_estimator=acc_estimator,
            all_domains=logger.domains,
            split=split,
        )
        logger.on_eval_end(aux, list_domain=speak_val_dl.dataset.domain, modality=split)
        print_aux(split, aux)

        ###########################
        #   TEST SPLIT
        ###########################
        split = "test"

        aux = evaluate(
            speak_test_dl,
            int_model,
            list_model,
            list_vocab,
            loss_f=loss_f,
            acc_estimator=acc_estimator,
            all_domains=logger.domains,
            split=split,
        )

        logger.on_eval_end(
            aux, list_domain=speak_test_dl.dataset.domain, modality=split
        )
        print_aux(split, aux)

        logger.on_train_end({}, epoch_id=0)
