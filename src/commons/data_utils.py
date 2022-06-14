import argparse
from typing import Optional, Tuple

import rich
import torch
from torch.utils.data import DataLoader

from src.commons.model_utils import hypo2utterance
from src.data.dataloaders import AbstractDataset, ListenerDataset, SpeakerDataset, Vocab


def get_dataloaders(
    args: argparse.Namespace,
    vocab: Vocab,
    domain: str = None,
    unary_val_bs: Optional[bool] = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataloaders based on args
    Parameters
    ----------
    args
    vocab
    domain : only used whit listener dataset.
    unary_val_bs: validation batch size ==1

    Returns
    -------

    """

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
            "data_dir": args.data_path,
        }

        if domain is not None:
            kwargs["domain"] = domain

            _set = ListenerDataset(**kwargs)
        else:
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

    load_params_val = load_params_test if unary_val_bs else load_params

    training_loader = torch.utils.data.DataLoader(datasets[0], **load_params)

    test_loader = torch.utils.data.DataLoader(datasets[1], **load_params_test)

    val_loader = torch.utils.data.DataLoader(datasets[2], **load_params_val)

    return training_loader, test_loader, val_loader


def speaker_augmented_dataloader(
    dataloader: DataLoader,
    vocab: Vocab,
    speak_model: torch.nn.Module,
    batch_size: int,
    split_name: str,
    shuffle: Optional[bool] = False,
) -> DataLoader:
    """
    Augment the canon dataloader with speaker generated utterances and embeddings

    """
    for ii, data in rich.progress.track(
        enumerate(dataloader),
        total=len(dataloader),
        description=f"Generating hypotesis for split '{split_name}'",
    ):
        # get datapoints
        target_img_feats = data["target_img_feats"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        visual_context = data["concat_context"]

        # generate hypo with speaker
        hypo, _, h1 = speak_model.generate_hypothesis(
            prev_utterance, prev_utt_lengths, visual_context, target_img_feats
        )
        utterance = hypo2utterance(hypo, vocab)

        dataloader.dataset.data[ii]["speak_utterance"] = utterance.squeeze().tolist()
        dataloader.dataset.data[ii]["speak_h1embed"] = h1.squeeze().tolist()

    dp = next(iter(dataloader)).keys()
    assert (
        "speak_utterance" in dp and "speak_h1embed" in dp
    ), "dataloader update did not work"

    load_params = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": AbstractDataset.get_collate_fn(
            speak_model.device, vocab["<sos>"], vocab["<eos>"], vocab["<nohs>"]
        ),
    }

    dataloader = torch.utils.data.DataLoader(dataloader.dataset, **load_params)

    return dataloader
