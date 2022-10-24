import datetime
from typing import Dict, List, Tuple

import numpy as np
import rich.progress
import torch
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, AccuracyEstimator,
                         EarlyStopping, IntLossPretrain, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint,
                         load_wandb_dataset, mask_attn, merge_dict, parse_args,
                         save_model, speak2list_vocab, translate_utterance, mask_oov_embeds)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import get_model
from src.wandb_logging import ListenerLogger
from torch import optim, nn
from torch.utils.data import DataLoader


def normalize_aux(aux, data_length, all_domains, max_targets=3):
    aux["loss"] = np.mean(aux["loss"])

    aux["int_list_accuracy"] = np.sum(aux["int_list_accuracy"]) / data_length
    aux["list_target_accuracy"] = np.sum(aux["list_target_accuracy"]) / data_length
    aux["int_target_accuracy"] = np.sum(aux["int_target_accuracy"]) / data_length
    aux["int_list_neg_accuracy"] = np.sum(aux["int_list_neg_accuracy"]) / np.sum(
        aux["neg_pred_len"]
    )
    aux["int_list_pos_accuracy"] = np.sum(aux["int_list_pos_accuracy"]) / np.sum(
        aux["pos_pred_len"]
    )

    if "kl_div" in aux.keys():
        aux["kl_div"] = np.sum(aux["kl_div"]) / len(aux["kl_div"])
        aux["kolmo_smir_stat"] = np.sum(aux["kolmo_smir_stat"]) / len(aux["kolmo_smir_stat"])
        aux["kolmo_smir_pval"] = np.sum(aux["kolmo_smir_pval"]) / len(aux["kolmo_smir_pval"])

    def flatten(xss):
        return [x for xs in xss for x in xs]

    domains = flatten(aux.pop("domains"))
    aux["domain/list_target_acc"] = get_domain_accuracy(
        flatten(aux.pop("list_target_accuracy_dom")), domains, all_domains
    )
    aux["domain/int_list_acc"] = get_domain_accuracy(
        flatten(aux.pop("int_list_accuracy_dom")), domains, all_domains
    )
    aux["domain/int_target_acc"] = get_domain_accuracy(
        flatten(aux.pop("int_target_accuracy_dom")), domains, all_domains
    )

    int_list_neg_accuracy_dom = aux.pop("int_list_neg_accuracy_dom")
    d = [x[1] for x in int_list_neg_accuracy_dom]
    correct = [x[0] for x in int_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/int_list_neg_acc"] = get_domain_accuracy(correct, d, all_domains)

    int_list_neg_accuracy_dom = aux.pop("int_list_pos_accuracy_dom")
    d = [x[1] for x in int_list_neg_accuracy_dom]
    correct = [x[0] for x in int_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/int_list_pos_acc"] = get_domain_accuracy(correct, d, all_domains)

    # flatten nested lists
    aux["int_preds"] = flatten(aux["int_preds"])
    aux["list_preds"] = flatten(aux["list_preds"])

    if len(aux["target"]) > max_targets:
        aux["target"] = np.random.choice(
            aux["target"], size=max_targets, replace=False
        ).tolist()


def get_predictions(
    data: Dict,
    list_model: torch.nn.Module,
    int_model: torch.nn.Module,
    loss_f: IntLossPretrain,
    acc_estimator: AccuracyEstimator,
    list_vocab: Vocab,
) -> Tuple[torch.Tensor, int, Dict]:
    """
    Extract data, get list/int out, estimate losses and create log dict

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
    domains = data["domain"]

    masks = mask_attn(lengths, max_length_tensor, device)

    translator(utterance)
    # get outputs
    list_out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

    int_out = int_model(speak_embds, context_separate, context_concat, prev_hist, masks)

    targets = targets.to(device)

    # Losses and preds
    loss = loss_f(int_out, targets, list_out, domains)
    aux = acc_estimator(int_out, targets, list_out, domains)

    aux["loss"] = loss.detach().cpu().item()

    return loss, aux["int_list_accuracy"], aux


def evaluate(
    data_loader: DataLoader,
    int_model: torch.nn.Module,
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
            data, list_model, int_model, loss_f, acc_estimator, list_vocab
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
        project="interpreter-pretrain",
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
    # interpreter
    ##########################

    int_p = common_p

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

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.Adam(int_model.parameters(), lr=common_p.learning_rate)
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
    ## RESUME AND EARLYSTOPPING
    ###################################

    metric = int_p.metric

    if metric == "loss":

        es = EarlyStopping(int_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(int_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([int_model])

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    int_p.batch_size = 1
    int_p.shuffle = False
    data_domain = common_p.data_domain

    shuffle = common_p.shuffle
    training_loader, test_loader, val_loader = get_dataloaders(
        int_p, speak_vocab, data_domain
    )

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    if common_p.is_test:
        training_loader = []
        int_p.epochs = 1

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

    for epoch in range(int_p.epochs):

        print("Epoch : ", epoch)

        auxs = []
        data = {}

        int_model.train()
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

            optimizer.zero_grad()

            # get datapoints
            loss, accuracy, aux = get_predictions(
                data, list_model, int_model, loss_f, acc_estimator, list_vocab
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

        print(f"Train loss {aux['loss']:.6f}, accuracy {aux['int_list_accuracy']:.3f} ")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            int_model.eval()

            print(f"\nEvaluation")
            aux = evaluate(
                speak_val_dl,
                int_model,
                list_model,
                list_vocab,
                loss_f,
                acc_estimator,
                all_domains=logger.domains,
                split="eval",
            )
            eval_accuracy, eval_loss = aux["int_list_accuracy"], aux["loss"]

            print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy:.3f} ")
            logger.on_eval_end(
                aux, list_domain=speak_val_dl.dataset.domain, modality="eval"
            )

            print(f"\nTest")
            aux = evaluate(
                speak_test_dl,
                int_model,
                list_model,
                list_vocab,
                loss_f,
                acc_estimator,
                all_domains=logger.domains,
                split="test",
            )

            test_accuracy, test_loss = aux["int_list_accuracy"], aux["loss"]
            print(f"Test loss {test_loss:.6f}, accuracy {test_accuracy:.3f} ")

            logger.on_eval_end(
                aux, list_domain=speak_test_dl.dataset.domain, modality="test"
            )

        if not common_p.is_test:
            save_model(
                model=int_model,
                model_type="interpreter",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
                args=int_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if int_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
