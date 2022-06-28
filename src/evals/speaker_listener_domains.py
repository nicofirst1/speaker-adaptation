import copy
import os
from typing import Optional, List, Dict

import numpy as np
import rich.progress
import torch
import wandb
from torch.utils.data import DataLoader

from src.commons import (
    hypo2utterance,
    mask_attn,
    get_domain_accuracy,
    get_dataloaders,
    parse_args,
    load_wandb_checkpoint,
    LISTENER_CHK_DICT,
    SPEAKER_CHK,
)
from src.data.dataloaders import Vocab
from src.models import ListenerModel_hist, SpeakerModel_hist, get_model
from src.wandb_logging import ListenerLogger, WandbLogger


def log_table(golden_metrics, gen_metrics, in_domain=True):
    golden_aux = golden_metrics.pop("aux")
    gen_aux = gen_metrics.pop("aux")
    # define difference between golden and generated out_domain metrics
    diff_dict = dict_diff(golden_metrics, gen_metrics)

    ### datapoint table
    table_columns = ["list_domain", "modality", "mrr", "accuracy"]
    if not in_domain:
        table_columns += [f"{dom}" for dom in logger.domains]
    table_columns += ["target", "utt"]

    # reappend aux
    gen_metrics["aux"] = gen_aux
    golden_metrics["aux"] = golden_aux
    diff_dict["aux"] = golden_aux

    # define table data rows
    data = []
    data.append(
        generate_table_data(
            list_args.train_domain, "golden", table_columns, golden_metrics
        )
    )
    data.append(
        generate_table_data(
            list_args.train_domain, "gen", table_columns, gen_metrics
        )
    )
    data.append(
        generate_table_data(
            list_args.train_domain, "diff", table_columns, diff_dict
        )
    )

    # create table and log
    table = wandb.Table(columns=table_columns, data=data)
    return table

def dict_diff(golden_dict, gen_dict):
    res = {}
    for k, v1 in golden_dict.items():

        v2 = gen_dict[k]
        if isinstance(v1, dict):
            res[k] = dict_diff(v1, v2)
        else:
            res[k] = v1 - v2

    return res


def evaluate_trained_model(
    dataloader: DataLoader,
    list_model: torch.nn.Module,
    vocab: Vocab,
    domain: str,
    logger: WandbLogger,
    split: str,
    speak_model: Optional[torch.nn.Module] = None,
):
    accuracies = []
    ranks = []
    domains = []
    fake_loss = torch.as_tensor([0])
    in_domain = domain == dataloader.dataset.domain

    modality=split
    if in_domain:
        modality += "/in_domain"
    else:
        modality += "/out_domain"

    if speak_model is None:
        modality += "_golden"
    else:
        modality += "_generated"

    for ii, data in rich.progress.track(
        enumerate(dataloader),
        total=len(dataloader),
        description=f"Eval on domain '{domain}' with '{modality}' modality",
    ):

        if not in_domain:
            if data['domain'][0]==domain:
                continue

        if speak_model is not None:
            # generate hypo with speaker
            target_img_feats = data["target_img_feats"]
            prev_utterance = data["prev_utterance"]
            prev_utt_lengths = data["prev_length"]
            visual_context = data["concat_context"]

            # generate hypo with speaker
            hypo, _, _ = speak_model.generate_hypothesis(
                prev_utterance, prev_utt_lengths, visual_context, target_img_feats
            )
            utterance = hypo2utterance(hypo, vocab)
        else:
            # else take them from golden caption
            utterance = data["utterance"]
            hypo = data["orig_caption"]

        # get datapoints
        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        lengths = [utterance.shape[1]]
        targets = data["target"]
        prev_hist = data["prev_histories"]

        max_length_tensor = utterance.shape[1]
        masks = mask_attn(lengths, max_length_tensor, list_model.device)

        # get listener output
        out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

        preds = torch.argmax(out, dim=1)

        correct = torch.eq(preds, targets).float().item()
        accuracies.append(correct)

        scores_ranked, images_ranked = torch.sort(out.squeeze(), descending=True)

        if out.shape[0] > 1:
            for s in range(out.shape[0]):
                # WARNING - assumes batch size > 1
                rank_target = images_ranked[s].tolist().index(targets[s].item())
                ranks.append(rank_target + 1)  # no 0

        else:
            rank_target = images_ranked.tolist().index(targets.item())
            ranks.append(rank_target + 1)  # no 0

        domains += data["domain"]

    accuracy = np.mean(accuracies)

    # normalize based on batches
    metrics = {}

    MRR = np.sum([1 / r for r in ranks]) / len(ranks)
    metrics["mrr"] = MRR
    metrics["accuracy"] = accuracy

    # log image\hypo and utterance
    img = data["image_set"][0][data["target"][0]]
    img = logger.img_id2path[str(img)]
    origin_utt = data["orig_utterance"][0]
    utt = hypo if speak_model is not None else origin_utt

    img = wandb.Image(img, caption=utt)

    metrics["aux"] = dict(target=img, utt=utt)

    logger.on_eval_end(
        copy.deepcopy(metrics),
        list_domain=dataloader.dataset.domain,
        modality=modality,
    )

    if "out" in modality:
        domain_accuracy = get_domain_accuracy(accuracies, domains, logger.domains)
        metrics["domain_accuracy"] = domain_accuracy

    return metrics


def generate_table_data(
    domain: str, modality: str, table_columns: List, metrics: Dict
) -> List:
    data = [domain, modality]
    for key in table_columns:
        if key in ["modality", "list_domain"]:
            continue
        elif key in metrics.keys():
            data.append(metrics[key])
        elif "domain_accuracy" in metrics.keys() and key in metrics["domain_accuracy"].keys():
            data.append(metrics["domain_accuracy"][key])
        elif key in metrics["aux"].keys():
            data.append(metrics["aux"][key])
        else:
            raise KeyError(f"No key '{key}' found in dict")
    return data


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    common_args = parse_args("list")

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)

    # load args
    speak_p = speak_check["args"]
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    print(speak_p)

    # for reproducibility
    seed = speak_p.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    ####################################
    # SPEAKER
    ####################################
    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)

    img_dim = 2048
    model = get_model("speak", speak_p.model_type)
    speaker_model = model(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        speak_p.beam_size,
        speak_p.max_len,
        speak_p.top_k,
        speak_p.top_p,
        device,
        use_beam=speak_p.use_beam,
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ####################################
    # LISTENER
    ####################################
    dom = common_args.train_domain
    url = LISTENER_CHK_DICT[dom]

    # for every listener
    list_checkpoint, _ = load_wandb_checkpoint(url, device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.vocab_file = "vocab.csv"
    list_args.vectors_file = os.path.basename(list_args.vectors_file)
    list_args.device = device

    # for debug
    list_args.subset_size = common_args.subset_size
    list_args.debug = common_args.debug
    list_args.test_split = common_args.test_split

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
        device,
    )

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)

    # add debug label
    tags = []
    if list_args.debug or list_args.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=list_vocab,
        opts=vars(list_args),
        group=list_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="speaker-list-dom",
    )

    # todo: log captions
    # todo make table



    with torch.no_grad():
        list_model.eval()

        train_loader, test_loader, val_loader = get_dataloaders(list_args, list_vocab, list_args.train_domain)

        ########################
        #  EVAL DOMAIN-SPEC
        ########################

        # GOLDEN
        golden_metrics = evaluate_trained_model(
            dataloader=val_loader,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="eval",

        )

        print(golden_metrics)

        # GENERATED
        gen_metrics=evaluate_trained_model(
            dataloader=val_loader,
            speak_model=speaker_model,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="eval",

        )

        print(gen_metrics)


        table = log_table(golden_metrics, gen_metrics)
        logger.log_to_wandb(dict(eval_in_domain=table), commit=True)

        ########################
        #  TEST DOMAIN-SPEC
        ########################
        print(f"Test on '{list_args.train_domain}' domain")

        print(f"Test on '{list_args.train_domain}' domain with golden caption ")
        golden_metrics = evaluate_trained_model(
            dataloader=test_loader,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="test",

        )
        print(golden_metrics)
        gen_metrics = evaluate_trained_model(
            dataloader=test_loader,
            speak_model=speaker_model,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="test",
        )
        print(gen_metrics)

        table = log_table(golden_metrics, gen_metrics)
        logger.log_to_wandb(dict(test_in_domain=table), commit=True)

        ########################
        #  EVAL OOD
        ########################
        _, test_loader, val_loader = get_dataloaders(list_args, list_vocab, "all")

        print(f"Eval on 'all' domain with golden caption")

        golden_metrics = evaluate_trained_model(
            dataloader=val_loader,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="eval",

        )
        print(golden_metrics)

        print(f"Eval on 'all' domain")

        # GENERATED
        gen_metrics = evaluate_trained_model(
            dataloader=val_loader,
            speak_model=speaker_model,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="eval",

        )
        print(gen_metrics)



        table = log_table(golden_metrics, gen_metrics, in_domain=False)
        logger.log_to_wandb(dict(eval_out_domain=table), commit=True)

        ########################
        #  TEST ODD
        ########################

        print(f"Test on 'all' domain with golden caption")
        golden_metrics = evaluate_trained_model(
            dataloader=test_loader,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="test",

        )
        print(golden_metrics)

        print(f"Test on 'all' domain")

        gen_metrics = evaluate_trained_model(
            dataloader=test_loader,
            speak_model=speaker_model,
            list_model=list_model,
            vocab=list_vocab,
            domain=dom,
            logger=logger,
            split="test",

        )
        print(gen_metrics)


        table = log_table(golden_metrics, gen_metrics, in_domain=False)
        logger.log_to_wandb(dict(test_out_domain=table), commit=True)

        logger.wandb_close()
