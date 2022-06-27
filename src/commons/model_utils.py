import os
from collections import Counter
from os.path import isfile, join
from typing import Dict, List, Tuple

import torch
import wandb

from src.commons.Params import Params
from src.wandb_logging import WandbLogger


def mask_attn(
    actual_num_tokens: torch.Tensor, max_num_tokens: int, device: torch.device
) -> torch.Tensor:
    masks = []

    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (
            max_num_tokens - actual_num_tokens[n]
        )

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def hypo2utterance(hypo, vocab):
    # # encode with list vocab
    # utterance = tokenizer.tokenize(hypo)
    #
    # if any(["#" in t for t in utterance]):
    #     # idk why byt surfboard is tokenized as 'surf' '##board' that raise an error, so skip
    #     raise ValueError()

    utterance = vocab.encode(hypo.strip().split(" "), add_special_tokens=False)
    utterance = utterance.unsqueeze(dim=0)
    utterance = utterance.long()

    return utterance


def get_domain_accuracy(
    accuracy: torch.Tensor, domains: torch.Tensor, all_domains: List[str]
) -> Dict[str, float]:
    """
    return a dict of domain:accuracy for all the domains in 'all_domains:
    Parameters
    ----------
    accuracy
    domains
    all_domains

    Returns
    -------

    """
    assert len(accuracy) == len(domains)

    domain_accs = {d: 0 for d in all_domains}
    domain_accs["all"] = 0

    for idx in range(len(domains)):
        if accuracy[idx]:
            dom = domains[idx]
            domain_accs[dom] += 1
            domain_accs["all"] += 1

    c = Counter(domains)

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

def load_wandb_file(url: str,  datadir="") -> str:
    """
    Load a wandb file and return the path to the downloaded file
    Parameters
    ----------
    url
    datadir

    Returns
    -------

    """
    if datadir=="":
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

    Returns
    -------

    """

    file=load_wandb_file(url,  datadir)
    checkpoint = torch.load(file, map_location=device)

    return checkpoint, file
