import datetime
from typing import Dict, List, Tuple
import lovely_tensors as lt

import numpy as np
import rich.progress
import torch
import wandb
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.commons import (
    SPEAKER_CHK,
    AccuracyEstimator,
    EarlyStopping,
    get_dataloaders,
    get_domain_accuracy,
    get_listener_check,
    load_wandb_checkpoint,
    load_wandb_dataset,
    mask_attn,
    mask_oov_embeds,
    merge_dict,
    parse_args,
    save_model,
    set_seed,
)
from src.commons.Translator import Translator
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import ListenerModel, SimulatorModel
from src.models.speaker.SpeakerModelEC import SpeakerModelEC
from src.wandb_logging import WandbLogger

global common_p


def normalize_aux(aux, data_length, all_domains, max_targets=0):
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

    domains = flatten(aux.pop("domains"))
    aux["domain/list_target_acc"] = get_domain_accuracy(
        flatten(aux.pop("list_target_accuracy_dom")), domains, all_domains
    )
    aux["domain/sim_list_acc"] = get_domain_accuracy(
        flatten(aux.pop("sim_list_accuracy_dom")), domains, all_domains
    )
    aux["domain/sim_target_acc"] = get_domain_accuracy(
        flatten(aux.pop("sim_target_accuracy_dom")), domains, all_domains
    )

    sim_list_neg_accuracy_dom = aux.pop("sim_list_neg_accuracy_dom")
    d = [x[1] for x in sim_list_neg_accuracy_dom]
    correct = [x[0] for x in sim_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/sim_list_neg_acc"] = get_domain_accuracy(correct, d, all_domains)

    sim_list_neg_accuracy_dom = aux.pop("sim_list_pos_accuracy_dom")
    d = [x[1] for x in sim_list_neg_accuracy_dom]
    correct = [x[0] for x in sim_list_neg_accuracy_dom]
    d = flatten(d)
    correct = flatten(correct)
    aux["domain/sim_list_pos_acc"] = get_domain_accuracy(correct, d, all_domains)

    # aux.pop("sim_list_pos_accuracy_dom")
    # aux.pop("sim_list_neg_accuracy_dom")
    # aux.pop("domains")
    # aux.pop("sim_list_accuracy_dom")
    # aux.pop("sim_target_accuracy_dom")
    # aux.pop("list_target_accuracy_dom")

    # flatten nested lists
    aux["sim_preds"] = flatten(aux["sim_preds"])
    aux["list_preds"] = flatten(aux["list_preds"])

    aux["sim_list_cm"] = wandb.plot.confusion_matrix(
        probs=None, y_true=aux["list_preds"], preds=aux["sim_preds"]
    )

    p, r, f1, s = precision_recall_fscore_support(aux["list_preds"], aux["sim_preds"])
    aux["sim_list_f1"] = f1.mean()
    aux["sim_list_precision"] = p.mean()
    aux["sim_list_recall"] = r.mean()

    aux["cohen_kappa_score"] = cohen_kappa_score(aux["list_preds"], aux["sim_preds"])
    aux["matthews_corrcoef"] = matthews_corrcoef(aux["list_preds"], aux["sim_preds"])
    # aux["list_dist"] = torch.concat(aux["list_dist"], dim=0).T
    # aux["sim_dist"] = torch.concat(aux["sim_dist"], dim=0).T

    if len(aux["target"]) > max_targets:
        aux["target"] = np.random.choice(
            aux["target"], size=max_targets, replace=False
        ).tolist()
    if max_targets == 0:
        del aux["target"]


def get_predictions(
    data: Dict,
    list_model: ListenerModel,
    sim_model: SimulatorModel,
    loss_f: nn.CrossEntropyLoss,
    acc_estimator: AccuracyEstimator,
    translator: Translator,
) -> Tuple[torch.Tensor, int, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """
    global common_p

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    utterance = data["speak_utterance"]
    lengths = data["speak_length"]
    targets = data["target"]
    speak_embds = data["speak_h1embed"]
    max_length_tensor = utterance.shape[1]
    domains = data["domain"]

    # get mask and translate utterance
    masks = mask_attn(lengths, max_length_tensor, list_model.device)
    utterance = translator.s2l(utterance)

    # get outputs
    list_out = list_model(utterance, context_separate, masks)
    sim_out = sim_model(
        separate_images=context_separate,
        utterance=utterance,
        masks=masks,
    )

    # Losses and preds
    list_preds = torch.argmax(list_out, dim=1)
    loss = loss_f(sim_out, list_preds)
    aux = acc_estimator(sim_out, targets, list_out, domains)

    if common_p.reduction == "mean":
        loss = loss.mean()
    elif common_p.reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("reduction not supported")

    aux["loss"] = loss.detach().cpu().item()

    return loss, aux["sim_list_accuracy"], aux


def evaluate(
    data_loader: DataLoader,
    sim_model: SimulatorModel,
    list_model: ListenerModel,
    translator: Translator,
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
            data, list_model, sim_model, loss_f, acc_estimator, translator
        )

        auxs.append(aux)

    aux = merge_dict(auxs)
    normalize_aux(aux, len(data_loader.dataset.data), all_domains=all_domains)

    return aux


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048
    global common_p

    common_p = parse_args("sim")
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

    logger = WandbLogger(
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

    list_check = get_listener_check(domain)

    list_checkpoint, _ = load_wandb_checkpoint(
        list_check,
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.device = device
    list_args.reset_paths()

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    list_model = ListenerModel(
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
        list_model.embeddings = mask_oov_embeds(
            list_model.embeddings,
            list_vocab,
            domain,
            replace_token=common_p.mask_oov_embed,
            data_path=common_p.data_path,
        )

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

    common_speak_p = parse_args("speak")

    # init speak model and load state

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
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

    optimizer = optim.AdamW(
        sim_model.parameters(), lr=common_p.learning_rate, weight_decay=0.0001
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=2, factor=0.5, verbose=True, threshold=0.5
    )
    loss_f = nn.CrossEntropyLoss(reduction="none")
    acc_estimator = AccuracyEstimator(domain, all_domains=logger.domains)

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

    # logger.watch_model([sim_model],)

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    common_p.batch_size = 1
    common_p.shuffle = False
    data_domain = common_p.data_domain

    training_loader, _, val_loader = get_dataloaders(
        common_p, speak_vocab, data_domain, splits=["train", "val"]
    )

    translator = Translator(speak_vocab, list_vocab, device=device)

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
        "all",
        load_params,
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        subset_size=common_p.subset_size,
    )

    load_params = {
        "batch_size": bs,
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
        "all",
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

        sim_model.train()

        # randomize order of data
        speak_train_dl.dataset.randomize_target_location()
        speak_val_dl.dataset.randomize_target_location()

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
                data,
                list_model,
                sim_model,
                loss_f,
                acc_estimator,
                translator,
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            nn.utils.clip_grad_value_(sim_model.parameters(), clip_value=1.0)
            optimizer.step()

        aux = merge_dict(auxs)
        normalize_aux(aux, len(speak_train_dl.dataset.data), all_domains=logger.domains)
        logger.on_eval_end(
            aux, list_domain=speak_train_dl.dataset.domain, modality="train"
        )

        print(
            f"Train loss {aux['loss']:.6f}, accuracy {aux['sim_list_accuracy'] * 100:.2f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            print(f"\nEvaluation")
            auxs = []

            for ii, data in rich.progress.track(
                enumerate(speak_val_dl),
                total=len(speak_val_dl),
                description=f"evaluating...",
            ):
                loss, accuracy, aux = get_predictions(
                    data, list_model, sim_model, loss_f, acc_estimator, translator
                )

                auxs.append(aux)

            aux = merge_dict(auxs)
            normalize_aux(
                aux, len(speak_val_dl.dataset.data), all_domains=logger.domains
            )

            eval_accuracy, eval_loss = aux["sim_list_accuracy"], aux["loss"]

            scheduler.step(eval_accuracy)

            print(
                f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f} "
            )
            logger.on_eval_end(
                aux, list_domain=speak_val_dl.dataset.domain, modality="eval"
            )

        ###################################
        ##  Saving and early stopping
        ###################################

        if epoch > 0 and epoch % 2 == 0:
            save_model(
                model=sim_model,
                model_type=f"Simulator{domain.capitalize()}",
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


if __name__ == "__main__":
    try:
        lt.monkey_patch()

        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, finishing run")
        wandb.finish()
