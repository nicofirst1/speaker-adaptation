import copy
import os
from typing import Dict, List, Optional

import numpy as np
import rich.progress
import torch
import wandb
from torch.utils.data import DataLoader

from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, get_dataloaders,
                         get_domain_accuracy, load_wandb_checkpoint, mask_attn,
                         parse_args, set_seed, speak2list_vocab,
                         translate_utterance, get_domain_mrr, get_simulator_check, AccuracyEstimator)
from src.data.dataloaders import Vocab
from src.models.listener.ListenerModel import ListenerModel
from src.models.simulator.SimulatorModel import SimulatorModel
from src.models.speaker.SpeakerModel import SpeakerModel
from src.wandb_logging import WandbLogger


def gen_list_table(
        golden_metrics: Dict, gen_metrics: Dict, cur_domain: str
) -> wandb.Table:
    """
    Create and fill a wandb table for the generated,golden and difference metrics.
    Parameters
    ----------
    golden_metrics : metrics for speaker generated utterances
    gen_metrics : metrics for the golden captions
    in_domain: true, if the metrics come from an in domain setting

    Returns
    -------

    """

    ### datapoint table

    table_columns = ["list_domain", "golden_acc",
                     "golden_mrr", "gen_acc", "gen_mrr"]

    # define table data rows
    data = [cur_domain]
    data += [golden_metrics['list_accuracy'], golden_metrics['list_mrr'], gen_metrics['list_accuracy'],
             gen_metrics['list_mrr']]

    # create table and log
    table = wandb.Table(columns=table_columns, data=[data])
    return table


def gen_sim_table(
        golden_metrics: Dict, gen_metrics: Dict, cur_domain: str
) -> wandb.Table:
    """
    Create and fill a wandb table for the generated,golden and difference metrics.
    Parameters
    ----------
    golden_metrics : metrics for speaker generated utterances
    gen_metrics : metrics for the golden captions
    in_domain: true, if the metrics come from an in domain setting

    Returns
    -------

    """

    table_columns = ["list_domain", "sim_list_acc", "sim_list_pos_acc", "sim_list_neg_acc", "sim_target_acc",
                     "list_target_acc"]

    # define table data rows
    data = [cur_domain]
    data += [gen_metrics['sim_accuracy'], gen_metrics['sim_pos_accuracy'], gen_metrics['sim_neg_accuracy'],
             gen_metrics['sim_trg_acc'], gen_metrics['list_accuracy']]

    # create table and log
    table = wandb.Table(columns=table_columns, data=[data])
    return table


def gen_list_domain_table(
        ind_gen: Dict,ood_gen: Dict, cur_domain: str
) -> wandb.Table:
    """
    Create and fill a wandb table for the generated,golden and difference metrics.
    Parameters
    ----------
    golden_metrics : metrics for speaker generated utterances
    gen_metrics : metrics for the golden captions
    in_domain: true, if the metrics come from an in domain setting

    Returns
    -------

    """

    ood_gen=ood_gen['domain_list_accuracy']
    ood_gen[dom]=ind_gen['list_accuracy']
    ood_gen.pop("all")
    ood_gen=sorted(list(ood_gen.items()))

    table_columns = ["list_domain"]+list(map(lambda x: x[0], ood_gen))

    # define table data rows
    data = [cur_domain] + list(map(lambda x: x[1], ood_gen))

    # create table and log
    table = wandb.Table(columns=table_columns, data=[data])
    return table



def evaluate_and_log(listener: ListenerModel, speaker: SpeakerModel, simulator: SimulatorModel, logger: WandbLogger,
                     cur_domain: str, data_loader: DataLoader, translator, split: str):
    print(f"{split} on '{cur_domain}' domain with golden caption ")
    golden_metrics = evaluate_trained_model(
        dataloader=data_loader,
        list_model=listener,
        sim_model=simulator,
        translator=translator,
        domain=dom,
        logger=logger,
        split=split,
    )

    print(golden_metrics)

    print(f"{split} on '{cur_domain}' domain")
    gen_metrics = evaluate_trained_model(
        dataloader=data_loader,
        speak_model=speaker,
        list_model=listener,
        sim_model=simulator,
        translator=translator,
        domain=dom,
        logger=logger,
        split=split,
    )
    print(gen_metrics)

    in_domain = cur_domain == data_loader.dataset.domain

    l_table = gen_list_table(golden_metrics, gen_metrics, cur_domain)
    s_table = gen_sim_table(golden_metrics, gen_metrics, cur_domain)

    l_label = f"{split}_IND_list" if in_domain else f"{split}_OOD_list"
    s_label = f"{split}_IND_sim" if in_domain else f"{split}_OOD_sim"

    logs = {l_label: l_table, s_label: s_table}

    logger.log_to_wandb(logs, commit=True)

    return gen_metrics, golden_metrics


def evaluate_trained_model(
        dataloader: DataLoader,
        list_model: torch.nn.Module,
        sim_model: torch.nn.Module,
        translator,
        domain: str,
        logger: WandbLogger,
        split: str,
        speak_model: Optional[torch.nn.Module] = None,
):
    list_accuracies = []
    sim_accuracies = []
    sim_pos_accuracies = []
    sim_neg_accuracies = []
    sim_trg_acc = []
    sim_kl_div = []

    ranks = []
    domains = []
    in_domain = domain == dataloader.dataset.domain
    acc_estimator = AccuracyEstimator(domain, all_domains=logger.domains)

    # define modality for wandb panels
    modality = split
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
            description=f"Domain '{domain}' with '{modality}' modality",
    ):

        # skip indomain samples
        if not in_domain:
            if data["domain"][0] == domain:
                continue

        if speak_model is not None:
            # generate hypo with speaker
            target_img_feats = data["target_img_feats"]
            prev_utterance = data["prev_utterance"]
            prev_utt_lengths = data["prev_length"]
            visual_context = data["concat_context"]

            # generate hypo with speaker
            utterance, _, decoder_hid = speak_model.generate_hypothesis(
                prev_utterance, prev_utt_lengths, visual_context, target_img_feats
            )

            hypo = [speak_vocab.decode(sent) for sent in utterance][0]
            translator(utterance)

            # utterance = hypo2utterance(hypo, vocab)
        else:
            # else take them from golden caption
            utterance = data["utterance"]
            hypo = data["orig_utterance"]

        # get datapoints
        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        lengths = torch.as_tensor([utterance.shape[1]])
        targets = data["target"]

        max_length_tensor = utterance.shape[1]
        masks = mask_attn(lengths, max_length_tensor, list_model.device)

        # get listener output
        list_out = list_model(utterance, context_separate, context_concat, masks)
        list_out = list_out.squeeze(-1)
        list_preds = torch.argmax(list_out, dim=1)
        list_correct = torch.eq(list_preds, targets).float().item()
        list_accuracies.append(list_correct)

        if speak_model is not None:
            # get simulator output
            sim_out = sim_model(decoder_hid, utterance, context_separate, context_concat, masks)
            sim_preds = torch.argmax(sim_out, dim=1)
            sim_correct = torch.eq(sim_preds, list_preds).float().item()
            sim_accuracies.append(sim_correct)
            aux = acc_estimator(sim_out, targets, list_out, data["domain"])

            sim_pos_accuracies.append(aux["sim_list_pos_accuracy"])
            sim_neg_accuracies.append(aux["sim_list_neg_accuracy"])
            sim_trg_acc.append(aux["sim_target_accuracy"])
            sim_kl_div.append(aux["kl_div"])

        scores_ranked, images_ranked = torch.sort(list_out.squeeze(), descending=True)
        rank_target = images_ranked.tolist().index(targets.item())
        ranks.append(rank_target + 1)  # no 0

        domains += data["domain"]

    list_accuracy = np.mean(list_accuracies)
    sim_accuracy = np.mean(sim_accuracies)
    MRR = np.sum([1 / r for r in ranks]) / len(ranks)
    sim_pos_accuracy = np.mean(sim_pos_accuracies)
    sim_neg_accuracy = np.mean(sim_neg_accuracies)
    sim_trg_acc = np.mean(sim_trg_acc)
    sim_kl_div = np.mean(sim_kl_div)

    metrics = {}
    metrics["list_mrr"] = MRR
    metrics["list_accuracy"] = list_accuracy
    metrics["sim_accuracy"] = sim_accuracy
    metrics["sim_pos_accuracy"] = sim_pos_accuracy
    metrics["sim_neg_accuracy"] = sim_neg_accuracy
    metrics["sim_trg_acc"] = sim_trg_acc
    metrics["sim_kl_div"] = sim_kl_div

    # log image\hypo and utterance
    # img = data["image_set"][0][data["target"][0]]
    # img = logger.img_id2path[str(img)]
    # origin_utt = data["orig_utterance"][0]
    # utt = hypo if speak_model is not None else origin_utt
    #
    # img = wandb.Image(img, caption=utt)

    # metrics["aux"] = dict(target=img, utt=utt)
    # filter out nan values
    metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
    logger.on_eval_end(
        copy.deepcopy(metrics),
        list_domain=dataloader.dataset.domain,
        modality=modality,
    )

    if "out" in modality:
        domain_accuracy = get_domain_accuracy(list_accuracies, domains, logger.domains)
        metrics["domain_list_accuracy"] = domain_accuracy
        if len(sim_accuracies) > 0:
            domain_accuracy = get_domain_accuracy(sim_accuracies, domains, logger.domains)
            metrics["domain_sim_accuracy"] = domain_accuracy
        domain_mrr = get_domain_mrr(ranks, domains, logger.domains)
        metrics["domain_mrr"] = domain_mrr

    return metrics


def generate_table_row(
        domain: str, modality: str, table_columns: List, metrics: Dict
) -> List:
    """
    Generate wandb table rows for the log_table function above
    Parameters
    ----------
    domain
    modality
    table_columns
    metrics

    Returns
    -------

    """

    data = [domain, modality]
    for key in table_columns:
        if key in ["modality", "list_domain"]:
            continue
        elif key in metrics.keys():
            data.append(metrics[key])
        elif (
                "domain_accuracy" in metrics.keys()
                and key in metrics["domain_accuracy"].keys()
        ):
            data.append(metrics["domain_accuracy"][key])

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

    # for reproducibility
    seed = common_args.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ####################################
    # SPEAKER
    ####################################
    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    speak_p = speak_check["args"]

    img_dim = 2048

    speaker_model = SpeakerModel(
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
        device=device,
        use_beam=speak_p.use_beam,
        use_prev_utterances=False,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ####################################
    # LISTENER
    ####################################
    dom = common_args.train_domain
    url = LISTENER_CHK_DICT[dom]

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
        device,
    )

    # load from checkpoint
    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)

    ##########################
    # SIMULATOR
    ##########################
    common_p = parse_args("sim")

    if common_p.force_resume_url == "":
        check = get_simulator_check(common_p.sim_domain, common_p.golden_data_perc)
    else:
        check = common_p.force_resume_url
    sim_check, _ = load_wandb_checkpoint(check, device)

    # load args
    sim_p = sim_check["args"]
    common_p.train_domain = dom
    common_p.device = device

    # override common_p with sim_p
    common_p.hidden_dim = sim_p.hidden_dim
    common_p.attention_dim = sim_p.attention_dim
    common_p.dropout_prob = sim_p.dropout_prob

    sim_model = SimulatorModel(
        len(list_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.sim_domain,
        common_p.device,
    ).to(device)

    if common_p.type_of_int != "untrained":
        sim_model.load_state_dict(sim_check["model_state_dict"])

    sim_model = sim_model.to(device)
    sim_model = sim_model.eval()

    ##########################
    # Logger
    ##########################

    logger = WandbLogger(
        vocab=list_vocab,
        opts=vars(common_p),
        group=list_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        project="test_pipeline",
        tags=common_args.tags,
    )
    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    ########################################
    # EVALUATE
    ########################################

    with torch.no_grad():
        list_model.eval()

        ########################
        #  IN DOMAIN
        ########################

        # get dataloaders
        train_loader, test_loader, val_loader = get_dataloaders(
            list_args, speak_vocab, list_args.train_domain
        )

        ind_gen, _=evaluate_and_log(listener=list_model, speaker=speaker_model, data_loader=test_loader, logger=logger,
                         split="test", translator=translator, simulator=sim_model, cur_domain=dom)

        ########################
        #  OOD
        ########################
        _, test_loader, val_loader = get_dataloaders(list_args, speak_vocab, "all")

        ood_gen,_=evaluate_and_log(listener=list_model, speaker=speaker_model, data_loader=test_loader, logger=logger,
                         split="test", translator=translator, simulator=sim_model, cur_domain=dom)


        table=gen_list_domain_table(ind_gen, ood_gen, dom)
        logger.log_to_wandb(dict(list_domain=table), commit=True)

        logger.wandb_close()
