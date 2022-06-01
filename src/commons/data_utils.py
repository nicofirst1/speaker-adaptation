import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from data.dataloaders import (AbstractDataset, ListenerDataset, SpeakerDataset,
                              Vocab)


def get_dataloaders(args: argparse.Namespace, vocab: Vocab, domain: str = None) -> Tuple[
    DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load dataloaders based on args
    Parameters
    ----------
    args
    vocab
    domain : only used whit listener dataset.

    Returns
    -------

    """
    if domain == "all":
        domain = "speaker"

    datasets = []
    # generate kwargs for different splits
    for split in ["train", "test", "val"]:
        kwargs = {
            "utterances_file": f"{split}_{args.utterances_file}",
            "vectors_file": args.vectors_file,
            "chain_file": f"{split}_{args.chains_file}",
            "orig_ref_file": f"{split}_{args.orig_ref_file}",
            "split": split,
            "subset_size": args.subset_size,
            "image_size": args.image_size,
            "img2dom_file": args.img2dom_file,
        }

        if domain is not None:
            kwargs["domain"] = domain
            kwargs["data_dir"] = args.data_path

            _set = ListenerDataset(**kwargs)
        else:
            kwargs["data_dir"] = args.speaker_data
            _set = SpeakerDataset(**kwargs)

        datasets.append(_set)


    load_params = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "collate_fn": AbstractDataset.get_collate_fn(
            args.device, vocab["<sos>"], vocab["<eos>"], vocab["<nohs>"]
        ),
    }

    load_params_test = {
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": AbstractDataset.get_collate_fn(
            args.device, vocab["<sos>"], vocab["<eos>"], vocab["<nohs>"]
        ),
    }

    training_loader = torch.utils.data.DataLoader(datasets[0], **load_params)
    training_beam_loader = torch.utils.data.DataLoader(datasets[0], **load_params_test)

    test_loader = torch.utils.data.DataLoader(datasets[1], **load_params_test)

    val_loader = torch.utils.data.DataLoader(datasets[2], **load_params_test)

    return training_loader, test_loader, val_loader, training_beam_loader
