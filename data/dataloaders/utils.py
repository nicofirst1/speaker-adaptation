import argparse
import json
import os
from typing import Dict, Tuple, List

import torch

from data.dataloaders.AbstractDataset import AbstractDataset
from data.dataloaders.ListenerDataset import ListenerDataset
from data.dataloaders.SpeakerDataset import SpeakerDataset
from data.dataloaders.Vocab import Vocab


def get_dataloaders(args: argparse.Namespace, vocab: Vocab, domain: str = None):
    if domain == "all":
        domain = "speaker"

    datasets = []
    for split in ['train', 'test', 'val']:
        kwargs = {
            "utterances_file": f"{split}_{args.utterances_file}",
            "vectors_file": args.vectors_file,
            "chain_file": f"{split}_{args.chains_file}",
            "orig_ref_file": f"{split}_{args.orig_ref_file}",
            "split": split,
            "subset_size": args.subset_size,
            "image_size":args.image_size
        }

        if domain is not None:
            kwargs['domain'] = domain
            kwargs['data_dir'] = args.data_path

            _set = ListenerDataset(**kwargs)
        else:
            kwargs['data_dir'] = args.speaker_data
            _set = SpeakerDataset(**kwargs)

        datasets.append(_set)

    print("vocab len", len(vocab))
    print("train len", len(datasets[0]), "longest sentence", datasets[0].max_len)
    print("test len", len(datasets[1]), "longest sentence", datasets[1].max_len)
    print("val len", len(datasets[2]), "longest sentence", datasets[2].max_len)

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

    val_loader = torch.utils.data.DataLoader(datasets[2], **load_params)

    return training_loader, test_loader, val_loader, training_beam_loader

