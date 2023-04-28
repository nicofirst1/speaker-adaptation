import os
from collections import Counter, defaultdict
from os.path import isfile, join
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb

from src.commons.Params import Params
from src.data.dataloaders import Vocab
from src.wandb_logging import AbstractWandbLogger


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def mask_attn(
    actual_num_tokens: torch.Tensor, max_num_tokens: int, device: torch.device
) -> torch.Tensor:
    """
    Maske attention function
    Parameters
    ----------
    actual_num_tokens : length of the utterance vector
    max_num_tokens : max length
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

    masks = torch.tensor(masks).unsqueeze(-1).to(device)

    return masks


def get_mask_from_utts(
    utts: torch.Tensor, vocab: Vocab, device: torch.device
) -> torch.Tensor:
    """
    Return a mask for the utterances, where the padding is masked
    Padding is 0 and end of sentence is 3

    Parameters
    ----------
    utts : tensor of utterances [batch_size, max_len]
    vocab : A vocab class
    device

    """
    mask = torch.ones_like(utts).to(device)
    mask[utts == vocab["<pad>"]] = 0
    mask[utts == vocab["<eos>"]] = 0

    # invert mask
    mask = mask == 0

    return mask.unsqueeze(-1)


def hypo2utterance(hypo: str, vocab):
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


def get_domain_mrr(
    ranks: torch.Tensor, domains: torch.Tensor, all_domains: List[str]
) -> Dict[str, float]:
    """
    Return a dict of domain:mrr for all the domains in 'all_domains:
    Parameters
    ----------
    ranks : tensor of boolean values to map the correct prediction of index i
    domains : tensor of string, domains for prediction of index i
    all_domains : list of all possible domains

    Returns
    -------
        dictionary mapping domain->accuracy
    """
    assert len(ranks) == len(domains)

    domain_accs = {d: 0 for d in all_domains}
    domain_accs["all"] = 0

    # add all the correct guesses
    for idx in range(len(domains)):
        dom = domains[idx]
        domain_accs[dom] += 1 / ranks[idx]
        domain_accs["all"] += 1 / ranks[idx]

    # count number of domains
    c = Counter(domains)

    # divide by number of domain's sample
    for k, v in c.items():
        domain_accs[k] /= v

    domain_accs["all"] /= len(ranks)

    return domain_accs


def save_model(
    model: torch.nn.Module,
    model_type: str,
    epoch: int,
    accuracy: float,
    optimizer: torch.optim.Optimizer,
    args: Params,
    timestamp: str,
    logger: AbstractWandbLogger,
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
        # "optimizer_state_dict": optimizer.state_dict(),
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

    new_url = url
    if datadir == "":
        api = wandb.Api()
        try:
            artifact = api.artifact(url)
        except wandb.errors.CommError:
            # try legacy names
            new_url = url.replace("interpreter", "simulator").replace(
                "Interpreter", "Simulator"
            )
            artifact = api.artifact(new_url)

        datadir = artifact.download()

    files = [f for f in os.listdir(datadir) if isfile(join(datadir, f))]

    if len(files) > 1:
        raise FileExistsError(f"More than one checkpoint found in {datadir}!")
    files = join(datadir, files[0])

    if new_url != url:
        project = artifact.project.replace("simulator", "interpreter").replace(
            "Simulator", "Interpreter"
        )
        name = (
            artifact.name.replace("simulator", "interpreter")
            .replace("Simulator", "Interpreter")
            .split(":")[0]
        )
        new_art = wandb.Artifact(
            name,
            type=artifact.type,
            description=artifact.description,
            metadata=artifact.metadata,
        )
        new_art.add_file(files)

        if wandb.run is None:
            wandb.init(
                project=project,
                entity="adaptive-speaker",
                settings=wandb.Settings(start_method="fork"),
            )

        wandb.log_artifact(new_art)

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


def merge_dict(dicts: List[Dict]) -> Dict:
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
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            dd[key].append(value)

    return dd


def draw_grad_graph(params, input, output, file_name="grad_graph.png"):
    import torchviz

    (grad_x,) = torch.autograd.grad(output, input, create_graph=True)
    params.update({"grad_x": grad_x, "in": input, "out": output})
    file = torchviz.make_dot((grad_x, input, output), params=params)
    file.render(file_name)
    return file


def logprobs_from_logits(logits, labels):
    """
    taken from https://github.com/lvwerra/trl/blob/d1c75293287483883f42f79b253d96315662bb1b/trl/core.py#L91
    """
    logp = torch.log_softmax(logits, dim=-1)
    if logp.ndim > 2:
        logp = logp.permute(1, 0, 2)
        labels = labels.unsqueeze(-1)

    logpy = torch.gather(logp, -1, labels).squeeze(-1)
    return logpy


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Standardizes a tensor
    """
    return (tensor - tensor.mean()) / tensor.std()


def change2random(tensor: torch.Tensor) -> torch.Tensor:
    """
    debug function
    """
    if tensor.dtype == torch.int64:
        t = torch.randint(0, 1000, tensor.shape)

    else:
        t = torch.rand(tensor.shape)

    return t.to(tensor.device)


def to_concat_context(separate_images: torch.Tensor) -> torch.Tensor:
    """
    Concatenate context images into one image
    Parameters
    ----------
    separate_images : [batch_size, context_size, features]

    Returns
    -------
    concatenated_images : [batch_size,  context_size * features]
    """
    batch_size, context_size, features = separate_images.shape
    concatenated_images = separate_images.view(batch_size, -1)
    return concatenated_images.to(separate_images.device)
