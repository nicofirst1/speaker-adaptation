import concurrent
import datetime
import random
import string
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

import lovely_tensors as lt
import numpy as np
import pandas as pd
import rich.progress
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from src.commons import (
    SPEAKER_CHK,
    AccuracyEstimator,
    get_dataloaders,
    get_listener_check,
    get_simulator_check,
    load_wandb_checkpoint,
    mask_attn,
    mask_oov_embeds,
    parse_args,
    set_seed, SPEAKER_CHK_EC,
)
from src.commons.Translator import Translator
from src.commons.model_utils import get_mask_from_utts
from src.data.dataloaders import Vocab
from src.models import ListenerModel, SpeakerModel
from src.models.simulator.SimulatorModel import SimulatorModel
from src.models.simulator.SimulatorModel_old import SimulatorModel_old
from src.models.speaker.SpeakerModelEC import SpeakerModelEC
from src.wandb_logging import WandbLogger


def generate_ood_table(
    df: pd.DataFrame, s_iter: int, domain: List[str], list_domain: str, sim_domain: str
) -> wandb.Table:
    rows = []
    columns = [
        "listener domain",
        "simulator domain",
        "target domain",
        "golden acc",
        "original acc",
        "adapted acc",
    ]
    ood_accs = {d: 0 for d in domain}
    for d in sorted(ood_accs.keys()):
        # filter out target with this domain
        ood_df = df[df["target domain"] == d]

        # get acc for this domain
        golden_acc = ood_df["golden_acc"].mean()
        original_acc = ood_df["original_acc"].mean()

        # adapted accs are a matrix so remove -1, sum and get mean
        adapt_acc_idx_0 = df.columns.get_loc("adapted_acc_s0")
        adapt_acc_idx_1 = adapt_acc_idx_0 + s_iter

        adapt_acc = ood_df.iloc[:, adapt_acc_idx_0:adapt_acc_idx_1]
        adapt_acc = adapt_acc[adapt_acc != -1]
        adapt_acc = adapt_acc.dropna(axis=1, how="all")
        adapt_mean = 0
        idx = 0
        for index, row in adapt_acc.iterrows():
            row = row.dropna()
            adapt_mean += row[-1]
            idx += 1

        if idx != 0:
            adapt_mean /= idx

        # append to rows
        rows.append(
            [
                list_domain,
                sim_domain,
                d,
                golden_acc,
                original_acc,
                adapt_mean,
            ]
        )

    table = wandb.Table(columns=columns, data=rows)
    return table


def predict(
    data: Dict,
    speak_model: SpeakerModelEC,
    list_model: ListenerModel,
    sim_model,
    criterion,
    adapt_lr: float,
    s: int,
):
    ## extract data
    context_separate = data["separate_images"]
    target_img_feats = data["target_img_feats"]
    targets = data["target"]
    golden_utt_ids = data["utterance"]

    ##################################
    # Get results for golden captions
    ##################################
    lengths = [golden_utt_ids.shape[1]]
    max_length_tensor = golden_utt_ids.shape[1]
    masks = mask_attn(lengths, max_length_tensor, device)

    golden_list_out = list_model(golden_utt_ids, context_separate, masks)
    golden_list_out.squeeze(dim=0)
    golden_acc = torch.argmax(golden_list_out.squeeze(dim=-1), dim=1)
    golden_acc = torch.eq(golden_acc, targets.squeeze()).double().item()

    ################################################
    #   Get results with original hypo
    ################################################
    # generate hypothesis
    utterance, logs, decoder_hid = speak_model.generate_hypothesis(
        context_separate,
        target_img_feats,
    )

    utterance = translator.s2l(utterance)

    history_att = logs["history_att"]

    # translate utt to ids and feed to listener
    lengths = [utterance.shape[1]]
    max_length_tensor = utterance.shape[1]

    masks = mask_attn(lengths, max_length_tensor, device)

    list_out = list_model(utterance, context_separate, masks)

    # get  accuracy
    list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
    list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
    original_acc = list_target_accuracy

    ################################################
    #   Get results with adapted hypo
    ################################################
    # decoder_hid = normalize(decoder_hid)
    h0 = decoder_hid.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([h0], lr=adapt_lr)
    optimizer.zero_grad()

    # repeat for s interations
    s_accs = []
    s_loss = []
    sim_accuracy = []
    sim_list_acc = []

    # perform loop
    i = 0
    while i < s:
        set_seed(seed)

        if isinstance(sim_model, SimulatorModel_old):
            sim_out = sim_model(
                h0,
                utterance,
                context_separate,
                masks,
            )

        else:
            sim_out = sim_model(
                context_separate,
                utterance,
                masks,
                speaker_embeds=h0,
            )

        # compute loss and perform backprop
        loss = criterion(sim_out, targets)
        aux = acc_estimator(sim_out, targets, list_out, data["domain"])
        loss.backward()
        optimizer.step()

        s_loss.append(loss.detach().item())

        # get modified hypo
        utterance, dec_logit = speak_model.nucleus_sampling(h0, history_att)
        utterance = translator.s2l(utterance)

        # generate utt for list
        # translate utt to ids and feed to listener
        masks = get_mask_from_utts(utterance, translator.list_vocab, device)

        list_out = list_model(utterance, context_separate, masks)

        # get  accuracy
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
        s_accs.append(list_target_accuracy)

        sim_accuracy.append(aux["sim_target_accuracy"])
        sim_list_acc.append(aux["sim_list_accuracy"])
        s_accs.append(aux["list_target_accuracy"])

        s_loss.append(loss.detach().item())

        # break if listener gets it right
        if aux["sim_target_accuracy"]:
            break
        i += 1

    res = dict(
        golden_acc=golden_acc,
        original_acc=original_acc,
        adapted_list_target_acc=s_accs,
        sim_accuracy=sim_accuracy,
        sim_list_acc=sim_list_acc,
        s_loss=s_loss,
    )
    return res


from rich.progress import Progress

def evaluate_subset(
    progress: Progress,
    task_id: int,
    data_subset,
    speak_model: SpeakerModel,
    sim_model: SimulatorModel,
    list_model: ListenerModel,
    criterion: nn.CrossEntropyLoss,
    adapt_lr: float = 0.1,
    s: int = 1,
) -> List[Dict]:
    results = []

    for data in data_subset:
        rs = predict(data, speak_model, list_model, sim_model, criterion, adapt_lr, s)
        results.append(rs)
        progress.update(task_id, advance=1)

    return results


def evaluate(
    data_loader: DataLoader,
    speak_model: SpeakerModel,
    sim_model: SimulatorModel,
    list_model: ListenerModel,
    criterion: nn.CrossEntropyLoss,
    split: str,
    adapt_lr: float = 0.1,
    s: int = 1,
) -> pd.DataFrame:
    """
    Perform evaluation of given split
    Parameters
    ----------
    data_loader
    speak_model
    sim_model
    list_model
    criterion
    split
    adapt_lr
    s

    Returns
    -------
    dataloader for analysis

    """

    n_threads = 8

    # Split the data into subsets
    data_list = list(data_loader)
    data_subsets = np.array_split(data_list, n_threads)

    # Create a shared Progress object
    progress = Progress()
    progress.start()

    # Create tasks for each subset
    tasks = [progress.add_task(f"[cyan]Evaluating on split {split} (Thread {i + 1})", total=len(subset)) for i, subset
             in enumerate(data_subsets)]

    # Use ThreadPoolExecutor to process subsets concurrently
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_results = [
            executor.submit(evaluate_subset, progress, tasks[i], data_subsets[i], speak_model, sim_model, list_model,
                            criterion, adapt_lr, s) for i in range(n_threads)]
        results = []
        for future in concurrent.futures.as_completed(future_results):
            results.extend(future.result())

    progress.stop()

    ##############################
    # METRICS
    ##############################

    # normalize results
    golden_acc = [x["golden_acc"] for x in results]
    original_acc = [x["original_acc"] for x in results]
    sim_acc = [x["sim_accuracy"][-1] for x in results]
    sim_list_acc = [x["sim_list_acc"][-1] for x in results]
    adapted_acc = [x["adapted_list_target_acc"][-1] for x in results]

    golden_accs = np.array(golden_acc).mean()
    original_accs = np.array(original_acc).mean()
    sim_accs = np.array(sim_acc).mean()
    sim_list_accs = np.array(sim_list_acc).mean()
    adapted_accs = np.array(adapted_acc).mean()

    adapt_golden_imporv = adapted_accs - golden_accs
    adapt_original_imporv = adapted_accs - original_accs

    metrics = dict(
        original_accs=original_accs,
        adapted_accs=adapted_accs,
        sim_accs=sim_accs,
        golden_accs=golden_accs,
        # hypo_table=hypo_table,
        sim_list_accs=sim_list_accs,
        adapt_golden_imporv=adapt_golden_imporv,
        adapt_original_imporv=adapt_original_imporv,
    )

    # console.print(metrics)
    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=split)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
    domain = common_p.train_domain
    lt.monkey_patch()

    ##########################
    # LISTENER
    ##########################

    list_checkpoint = get_listener_check(common_p.train_domain)
    list_checkpoint, _ = load_wandb_checkpoint(list_checkpoint, device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    # mask OOV words in the vocab

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

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK_EC['food'], device)
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################

    if common_p.force_resume_url == "":
        check = get_simulator_check(common_p.sim_domain)
        model = SimulatorModel_old
    else:
        check = common_p.force_resume_url
        model = SimulatorModel_old

    sim_check, _ = load_wandb_checkpoint(check, device)

    # load args
    sim_p = sim_check["args"]

    sim_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
        common_p.sim_domain,
        device,
    ).to(device)

    if common_p.type_of_int != "untrained":
        sim_model.load_state_dict(sim_check["model_state_dict"])

    sim_model = sim_model.to(device)
    sim_model = sim_model.eval()

    ###################################
    ##  LOGGER
    ###################################

    flag = common_p.type_of_int

    logger = WandbLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        project=f"fast-adaptive-speaker-{common_p.type_of_int}",
        tags=common_p.tags,
    )

    metric = common_p.metric
    sweep_config = wandb.config

    is_sweep = sweep_config._check_locked("adapt_lr")

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating hypothesis
    sim_p.batch_size = 1
    common_p.batch_size = 1
    train_dl_dom, test_dl_dom, val_dl_dom = get_dataloaders(
        common_p, speak_vocab, domain
    )
    train_dl_all, test_dl_all, val_dl_all = get_dataloaders(
        common_p, speak_vocab, domain="all"
    )

    translator = Translator(speak_vocab, list_vocab, device)

    ###################################
    ##  LOSS
    ###################################

    loss_f = nn.CrossEntropyLoss(reduction=common_p.reduction)
    acc_estimator = AccuracyEstimator(domain, all_domains=logger.domains)

    ###################################
    ##  EVAL LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    sim_model.eval()
    uid = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    ##################
    # OOD TEST
    ##################
    print(f"\nTest split for domain all")
    evaluate(
        test_dl_all,
        speaker_model,
        sim_model,
        list_model,
        criterion=loss_f,
        split="out_domain_test",
        adapt_lr=common_p.adapt_lr,
        s=common_p.s_iter,
    )

    logger.on_train_end({}, epoch_id=0)
