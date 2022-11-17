import datetime
from typing import Dict, List, Tuple

import numpy as np
import rich.progress
import torch
from torch.optim.lr_scheduler import ExponentialLR

from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, AccuracyEstimator,
                         EarlyStopping, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint,
                         load_wandb_dataset, mask_attn, merge_dict, parse_args,
                         save_model, speak2list_vocab, translate_utterance, mask_oov_embeds, set_seed)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.models.simulator.SimulatorModel import SimulatorModel
from src.wandb_logging import ListenerLogger
from torch import optim, nn
from torch.utils.data import DataLoader


def normalize_aux(aux, data_length, all_domains, max_targets=3):
    aux["loss"] = np.mean(aux["loss"])

    aux["sim_list_accuracy"] = np.sum(aux["sim_list_accuracy"]) / data_length
    aux["list_target_accuracy"] = np.sum(aux["list_target_accuracy"]) / data_length
    aux["sim_target_accuracy"] = np.sum(aux["sim_target_accuracy"]) / data_length
    aux["sim_list_neg_accuracy"] = np.sum(aux["sim_list_neg_accuracy"]) / np.sum(
        aux["neg_pred_len"]
    )
    aux["sim_list_pos_accuracy"] = np.sum(aux["sim_list_pos_accuracy"]) / np.sum(
        aux["pos_pred_len"]
    )

    if "kl_div" in aux.keys():
        aux["kl_div"] = np.sum(aux["kl_div"]) / len(aux["kl_div"])


    def flatten(xss):
        return [x for xs in xss for x in xs]

    # domains = flatten(aux.pop("domains"))
    # aux["domain/list_target_acc"] = get_domain_accuracy(
    #     flatten(aux.pop("list_target_accuracy_dom")), domains, all_domains
    # )
    # aux["domain/sim_list_acc"] = get_domain_accuracy(
    #     flatten(aux.pop("sim_list_accuracy_dom")), domains, all_domains
    # )
    # aux["domain/sim_target_acc"] = get_domain_accuracy(
    #     flatten(aux.pop("sim_target_accuracy_dom")), domains, all_domains
    # )

    # sim_list_neg_accuracy_dom = aux.pop("sim_list_neg_accuracy_dom")
    # d = [x[1] for x in sim_list_neg_accuracy_dom]
    # correct = [x[0] for x in sim_list_neg_accuracy_dom]
    # d = flatten(d)
    # correct = flatten(correct)
    # aux["domain/sim_list_neg_acc"] = get_domain_accuracy(correct, d, all_domains)
    #
    # sim_list_neg_accuracy_dom = aux.pop("sim_list_pos_accuracy_dom")
    # d = [x[1] for x in sim_list_neg_accuracy_dom]
    # correct = [x[0] for x in sim_list_neg_accuracy_dom]
    # d = flatten(d)
    # correct = flatten(correct)
    # aux["domain/sim_list_pos_acc"] = get_domain_accuracy(correct, d, all_domains)

    aux.pop("sim_list_pos_accuracy_dom")
    aux.pop("sim_list_neg_accuracy_dom")
    aux.pop("domains")
    aux.pop("sim_list_accuracy_dom")
    aux.pop("sim_target_accuracy_dom")
    aux.pop("list_target_accuracy_dom")


    # flatten nested lists
    aux["sim_preds"] = flatten(aux["sim_preds"])
    aux["list_preds"] = flatten(aux["list_preds"])

    aux["list_dist"] = torch.concat(aux["list_dist"], dim=0).T
    aux["sim_dist"] = torch.concat(aux["sim_dist"], dim=0).T

    if len(aux["target"]) > max_targets:
        aux["target"] = np.random.choice(
            aux["target"], size=max_targets, replace=False
        ).tolist()


def get_predictions(
    data: Dict,
    list_model: torch.nn.Module,
    sim_model: torch.nn.Module,
    loss_f: nn.CrossEntropyLoss,
    acc_estimator: AccuracyEstimator,
    list_vocab: Vocab,
) -> Tuple[torch.Tensor, int, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    utterance = data["speak_utterance"]
    lengths = data["speak_length"]
    targets = data["target"]
    prev_hist = data["prev_histories"]
    speak_embds = data["speak_h1embed"]
    max_length_tensor = utterance.shape[1]
    batch_size = utterance.shape[0]
    device = list_model.device
    domains = data["domain"]

    masks = mask_attn(lengths, max_length_tensor, device)

    translator(utterance)
    # get outputs
    list_out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

    sim_out = sim_model(speak_embds, utterance, context_separate, context_concat, prev_hist, masks)

    list_preds=torch.argmax(list_out,dim=1)

    # Losses and preds
    loss = loss_f(sim_out, list_preds)
    aux = acc_estimator(sim_out, targets, list_out, domains)

    aux["loss"] = loss.detach().cpu().item()

    return loss, aux["sim_list_accuracy"], aux


def evaluate(
    data_loader: DataLoader,
    sim_model: torch.nn.Module,
    list_model: torch.nn.Module,
    list_vocab: Vocab,
    loss_f: torch.nn.Module,
    acc_estimator: AccuracyEstimator,
    all_domains: List,
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
            data, list_model, sim_model, loss_f, acc_estimator, list_vocab
        )

        auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux, len(data_loader.dataset.data), all_domains=all_domains)

    return aux


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("int")
    domain = common_p.train_domain

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    with torch.no_grad():
        list_model.embeddings = mask_oov_embeds(list_model.embeddings, list_vocab, domain,
                                                replace_token=common_p.mask_oov_embed, data_path=common_p.data_path)

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
    # simulator
    ##########################


    sim_model = SimulatorModel(
        len(list_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.train_domain,
        common_p.device,
    ).to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(sim_model.parameters(), lr=common_p.learning_rate)
    loss_f = nn.CrossEntropyLoss(reduction=common_p.reduction)
    acc_estimator = AccuracyEstimator(
        domain, all_domains=logger.domains
    )
    #scheduler = ExponentialLR(optimizer, gamma=0.9)


    ###################################
    ## RESUME AND EARLYSTOPPING
    ###################################

    metric = common_p.metric

    if metric == "loss":

        es = EarlyStopping(common_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(common_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    #logger.watch_model([sim_model])

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    common_p.batch_size = 1
    common_p.shuffle = False
    data_domain = common_p.data_domain

    shuffle = common_p.shuffle
    training_loader, _, val_loader = get_dataloaders(
        common_p, speak_vocab, data_domain, splits=["train", "val"]
    )

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    if common_p.is_test:
        training_loader = []
        common_p.epochs = 1

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

    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(common_p.epochs):

        print("Epoch : ", epoch)

        auxs = []
        data = {}

        sim_model.train()
        # torch.enable_grad()
        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.progress.track(
            enumerate(speak_train_dl),
            total=len(speak_train_dl),
            description=f"Training epoch {epoch}",
        ):

            optimizer.zero_grad()

            # get datapoints
            loss, accuracy, aux = get_predictions(
                data, list_model, sim_model, loss_f, acc_estimator, list_vocab
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            optimizer.step()

        aux = merge_dict(auxs)
        normalize_aux(aux, len(speak_train_dl.dataset.data), all_domains=logger.domains)
        logger.on_eval_end(
            aux, list_domain=speak_train_dl.dataset.domain, modality="train"
        )

        print(f"Train loss {aux['loss']:.6f}, accuracy {aux['sim_list_accuracy']*100:.2f} ")

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
                acc_estimator,
                all_domains=logger.domains,
                split="eval",
            )
            eval_accuracy, eval_loss = aux["sim_list_accuracy"], aux["loss"]

            print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy*100:.3f} ")
            logger.on_eval_end(
                aux, list_domain=speak_val_dl.dataset.domain, modality="eval"
            )

            # print(f"\nTest")
            # aux = evaluate(
            #     speak_test_dl,
            #     sim_model,
            #     list_model,
            #     list_vocab,
            #     loss_f,
            #     acc_estimator,
            #     all_domains=logger.domains,
            #     split="test",
            # )
            #
            # test_accuracy, test_loss = aux["sim_list_accuracy"], aux["loss"]
            # print(f"Test loss {test_loss:.6f}, accuracy {test_accuracy:.3f} ")
            #
            # logger.on_eval_end(
            #     aux, list_domain=speak_test_dl.dataset.domain, modality="test"
            # )

        if not common_p.is_test and epoch%2 == 0:
            save_model(
                model=sim_model,
                model_type="simulator",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
                args=common_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if common_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
        print("\n\n")
