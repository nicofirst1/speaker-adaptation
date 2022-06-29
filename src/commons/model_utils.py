import os
from collections import Counter, defaultdict
from os.path import isfile, join
from typing import Dict, List, Tuple

import torch

import wandb
from src.commons.Params import Params
from src.wandb_logging import WandbLogger


def mask_attn(
    actual_num_tokens: torch.Tensor, max_num_tokens: int, device: torch.device
) -> torch.Tensor:
    """
    Maske attention function
    Parameters
    ----------
    actual_num_tokens : length of the utterance vector
    max_num_tokens : max lenght
    device

    Returns
    -------

    """
    masks = []

    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (
            max_num_tokens - actual_num_tokens[n]
        )

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def hypo2utterance(hypo:str, vocab):

    """
    Transform a hypothesis string into a tensor of utterances ids given the vocabulary
    Parameters
    ----------
    hypo
    vocab : A vocab class

    Returns
    -------

    """

    utterance = vocab.encode(hypo.strip().split(" "), add_special_tokens=False)
    utterance = utterance.unsqueeze(dim=0)
    utterance = utterance.long()

    return utterance


def get_domain_accuracy(
    accuracy: torch.Tensor, domains: torch.Tensor, all_domains: List[str]
) -> Dict[str, float]:
    """
    Return a dict of domain:accuracy for all the domains in 'all_domains:
    Parameters
    ----------
    accuracy : tensor of boolean values to map the correct prediction of index i
    domains : tensor of string, domains for prediction of index i
    all_domains : list of all possible domains

    Returns
    -------
        dictionary mapping domain->accuracy
    """
    assert len(accuracy) == len(domains)

    domain_accs = {d: 0 for d in all_domains}
    domain_accs["all"] = 0

    # add all the correct guesses
    for idx in range(len(domains)):
        if accuracy[idx]:
            dom = domains[idx]
            domain_accs[dom] += 1
            domain_accs["all"] += 1

    # count number of domains
    c = Counter(domains)

    # divide by number of domain's sample
    for k, v in c.items():
        domain_accs[k] /= v

    domain_accs["all"] /= len(accuracy)

    return domain_accs


def save_model(
    model: torch.nn.Module,
    model_type: str,
    epoch: int,
    accuracy: float,
    optimizer: torch.optim.Optimizer,
    args: Params,
    timestamp: str,
    logger: WandbLogger,
    **kwargs,
):
    """
    Save model in torch and wandb
    """
    seed = args.seed
    file_name = model_type + "_" + str(seed) + "_" + timestamp + ".pth"

    dir_path = join(args.working_dir, "saved_models")

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_name = join(dir_path, file_name)

    save_dict = {
        "accuracy": accuracy,
        "args": args,  # more detailed info, metric, model_type etc
        "epoch": str(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    save_dict.update(kwargs)
    torch.save(save_dict, file_name, pickle_protocol=5)
    logger.save_model(file_name, type(model).__name__, epoch, args)

    print("Model saved and logged to wandb")


def load_wandb_file(url: str, datadir="") -> str:
    """
    Load a wandb file and return the path to the downloaded file
    Parameters
    ----------
    url
    datadir : if given then check if the file is present in the dir. Used when offline

    Returns
    -------

    """
    if datadir == "":
        api = wandb.Api()
        artifact = api.artifact(url)

        datadir = artifact.download()

    files = [f for f in os.listdir(datadir) if isfile(join(datadir, f))]

    if len(files) > 1:
        raise FileExistsError(f"More than one checkpoint found in {datadir}!")
    files = join(datadir, files[0])
    return files


def load_wandb_checkpoint(url: str, device: str, datadir="") -> Tuple[Dict, str]:
    """
    Download a wandb model artifact and extract checkpoint with torch
    Parameters
    ----------
    url
    device
    datadir : if given then check if the file is present in the dir. Used when offline

    Returns
    -------

    """

    file = load_wandb_file(url, datadir)
    checkpoint = torch.load(file, map_location=device)

    return checkpoint, file


def merge_dict(dicts:List[Dict])->Dict[List]:
    """
    Merge a list of dict with same keys into a dict of lists
    Parameters
    ----------
    dicts

    Returns
    -------

    """
    dd = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            dd[key].append(value)

    return dd
