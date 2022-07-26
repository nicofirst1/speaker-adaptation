import datetime
from typing import Dict, Tuple

import numpy as np
import rich.progress
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.commons import (LISTENER_CHK_DICT, SIM_ALL_CE_CHK,
                         SPEAKER_CHK, EarlyStopping, get_dataloaders,
                         load_wandb_checkpoint, load_wandb_dataset, mask_attn,
                         merge_dict, parse_args, save_model, SimLoss, get_domain_accuracy)
from src.commons.Params import SpeakerArguments
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
    aux["sim_list_neg_accuracy"] = np.sum(aux["sim_list_neg_accuracy"]) / np.sum(aux["neg_pred_len"])
    aux["sim_list_pos_accuracy"] = np.sum(aux["sim_list_pos_accuracy"]) / np.sum(aux["pos_pred_len"])

    def flatten(xss):
        return [x for xs in xss for x in xs]

    domains=flatten(aux.pop("domains"))
    aux["domain/list_target_acc"]= get_domain_accuracy(flatten(aux.pop('list_target_accuracy_dom')), domains, logger.domains)
    aux["domain/sim_list_acc"]= get_domain_accuracy(flatten(aux.pop('sim_list_accuracy_dom')), domains, logger.domains)
    aux["domain/sim_target_acc"]= get_domain_accuracy(flatten(aux.pop('sim_target_accuracy_dom')), domains, logger.domains)

    sim_list_neg_accuracy_dom = aux.pop("sim_list_neg_accuracy_dom")
    d = [x[1] for x in sim_list_neg_accuracy_dom]
    correct = [x[0] for x in sim_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/sim_list_neg_acc"] = get_domain_accuracy(correct, d, logger.domains)

    sim_list_neg_accuracy_dom = aux.pop("sim_list_pos_accuracy_dom")
    d = [x[1] for x in sim_list_neg_accuracy_dom]
    correct = [x[0] for x in sim_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/sim_list_pos_acc"] = get_domain_accuracy(correct, d, logger.domains)

    # flatten nested lists
    aux["sim_preds"] = flatten(aux['sim_preds'])
    aux["list_preds"] = flatten(aux['list_preds'])

    if len(aux["target"]) > max_targets:
        aux["target"] = np.random.choice(
            aux["target"], size=max_targets, replace=False
        ).tolist()


def get_predictions(
        data: Dict,
        list_model: torch.nn.Module,
        sim_model: torch.nn.Module,
        loss_f: SimLoss,
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
    device = list_model.device
    domains = data['domain']

    masks = mask_attn(lengths, max_length_tensor, device)

    # get outputs
    list_out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

    sim_out = sim_model(speak_embds, context_separate, context_concat, prev_hist, masks)

    targets = targets.to(device)

    # Losses and preds
    list_loss = loss_f.ce(list_out, targets)
    sim_list_loss, aux = loss_f(sim_out, targets, list_out, domains)
    loss = sim_list_loss

    # logging
    # rnd_idx = np.random.randint(0, batch_size)
    # hypo = list_vocab.decode(utterance[rnd_idx])
    # caption = data["orig_utterance"][rnd_idx]
    # target = data["image_set"][rnd_idx][data["target"][rnd_idx]]
    # target = logger.img_id2path[str(target)]
    # target = wandb.Image(target, caption=f"Hypo:{hypo}\nCaption : {caption}")

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

    aux["list_loss"] = list_loss.detach().cpu().item()
    aux["sim_list_loss"] = sim_list_loss.detach().cpu().item()

    return loss, aux['sim_list_accuracy'], aux


def evaluate(
        data_loader: DataLoader,
        sim_model: torch.nn.Module,
        list_model: torch.nn.Module,
        list_vocab: Vocab,
        loss_f: torch.nn.Module,
        split: str,
) -> Dict:
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """

    auxs = []

    for ii, data in rich.progress.track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"evaluating '{split}' split...",
    ):
        loss, accuracy, aux = get_predictions(
            data, list_model, sim_model, loss_f, list_vocab=list_vocab
        )

        auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux, len(data_loader.dataset.data))

    return aux


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]


    speak_vocab = Vocab(parse_args("speak").vocab_file, is_speaker=True)

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="simulator-pretrain",
    )

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
        sim_check, _ = load_wandb_checkpoint(SIM_ALL_CE_CHK, device)
        # load args
        sim_p = sim_check["args"]
        sim_p.train_domain = domain
        sim_p.device = device
        sim_p.resume_train = common_p.resume_train

        # for debug
        sim_p.subset_size = common_p.subset_size
        sim_p.debug = common_p.debug
        sim_p.pretrain_loss = common_p.pretrain_loss

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

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(sim_model.parameters(), lr=common_p.learning_rate)
    loss_f = SimLoss(common_p.pretrain_loss, common_p.reduction,common_p.model_type,
                     alpha=common_p.focal_alpha, gamma=common_p.focal_gamma,
                     list_domain=domain, all_domains=logger.domains)

    ###################################
    ## RESUME AND EARLYSTOPPING
    ###################################

    if common_p.resume_train:
        sim_model.load_state_dict(sim_check["model_state_dict"])
        optimizer.load_state_dict(sim_check["optimizer_state_dict"])
        sim_model = sim_model.to(device)

    metric = sim_p.metric

    if metric == "loss":

        es = EarlyStopping(sim_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(sim_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model])

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    sim_p.batch_size = 1
    sim_p.shuffle = False
    data_domain=common_p.data_domain

    shuffle = common_p.shuffle
    training_loader, test_loader, val_loader = get_dataloaders(
        sim_p, speak_vocab, data_domain
    )

    if common_p.is_test:
        training_loader = []
        sim_p.epochs = 1

    load_params = {
        "batch_size": bs,
        "shuffle": shuffle,
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
        data_domain,
        load_params,
        list_vocab,
        speaker_model,
        val_loader,
        logger,
        subset_size=common_p.subset_size,
    )
    speak_test_dl = load_wandb_dataset(
        "test",
        data_domain,
        load_params,
        list_vocab,
        speaker_model,
        test_loader,
        logger,
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
                data, list_model, sim_model, loss_f, list_vocab
            )

            auxs.append(aux)

            # optimizer
            sim_model.zero_grad()
            loss.backward()
            optimizer.step()

        aux = merge_dict(auxs)
        normalize_aux(aux, len(speak_train_dl.dataset.data))
        logger.on_eval_end(
            aux, list_domain=speak_train_dl.dataset.domain, modality="train"
        )

        print(
            f"Train loss {aux['sim_list_loss']:.6f}, accuracy {aux['sim_list_accuracy']:.3f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            print(f"\nEvaluation")
            aux = evaluate(
                speak_val_dl,
                sim_model,
                list_model,
                list_vocab,
                loss_f,
                split="eval",
            )
            eval_accuracy, eval_loss = aux["sim_list_accuracy"], aux["sim_list_loss"]

            print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy:.3f} ")
            logger.on_eval_end(
                aux, list_domain=speak_val_dl.dataset.domain, modality="eval"
            )

            print(f"\nTest")
            aux = evaluate(
                speak_test_dl,
                sim_model,
                list_model,
                list_vocab,
                loss_f,
                split="test",
            )

            test_accuracy, test_loss = aux["sim_list_accuracy"], aux["sim_list_loss"]
            print(f"Test loss {test_loss:.6f}, accuracy {test_accuracy:.3f} ")

            logger.on_eval_end(
                aux, list_domain=speak_test_dl.dataset.domain, modality="test"
            )

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
