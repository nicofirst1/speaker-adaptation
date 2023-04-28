import argparse
import copy
import os.path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rich.progress
import rich.table
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from src.commons.model_utils import load_wandb_file
from src.commons.wandb_checkpoints import DATASET_CHK
from src.data.dataloaders import (AbstractDataset, ListenerDataset,
                                  SpeakerDataset, SpeakerUttDataset, Vocab)
from src.wandb_logging import AbstractWandbLogger
from torch.utils.data import DataLoader


def show_img(data, id2path, split_name, hypo="", idx=-1):
    log_dir = f"img_hypo_captions/{split_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    batch_size = data["target"].shape[0]
    rnd_idx = np.random.randint(0, batch_size)

    caption = data["orig_utterance"][rnd_idx]
    target = data["image_set"][rnd_idx][data["target"][rnd_idx]]
    target_id = target
    img_set = data["image_set"][rnd_idx]
    speak_utt = data["speak_utterance"][rnd_idx]

    if isinstance(speak_utt, torch.Tensor):
        speak_utt = speak_utt.tolist()

    target = id2path[str(target)]

    img = Image.open(target)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 16)
    draw.rectangle([0, 0, img.size[0], 50], fill=(0, 0, 0))

    to_write = f"Hypo :{hypo}\nCaption : {caption}\n Speak ids :{speak_utt}\n Img_set: {img_set}"

    draw.text((0, 0), to_write, (255, 255, 255), font=font)
    img.show()

    if idx < 0:
        idx = len(
            [
                name
                for name in os.listdir(log_dir)
                if os.path.isfile(os.path.join(log_dir, name))
            ]
        )
    # img.save(os.path.join(log_dir,f"{idx}_{target_id}.png"))


def get_dataloaders(
    args: argparse.Namespace,
    vocab: Vocab,
    domain: str = None,
    unary_val_bs: Optional[bool] = True,
    splits: Optional[List[str]] = ["train", "val", "test"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataloaders based on args. Can be either the speaker or the listener
    Parameters
    ----------
    splits
    args
    vocab
    domain : only used whit listener dataset.
    unary_val_bs: validation batch size ==1

    Returns
    -------

    """

    datasets = {}
    # copy args to avoid modifying original
    args_copy = copy.deepcopy(args)
    # generate kwargs for different splits
    for split in splits:

        # differenciate between seen/unseen/merged tests
        if domain is not None and split == "test":
            if args_copy.test_split != "all":
                args_copy.orig_ref_file = (
                    f"{args_copy.test_split}_{args_copy.orig_ref_file}"
                )
                args_copy.chains_file = (
                    f"{args_copy.test_split}_{args_copy.chains_file}"
                )

        kwargs = {
            "utterances_file": f"{split}_{args_copy.utterances_file}",
            "vectors_file": args_copy.vectors_file,
            "chain_file": f"{split}_{args_copy.chains_file}",
            "orig_ref_file": f"{split}_{args_copy.orig_ref_file}",
            "split": split,
            "subset_size": args_copy.subset_size,
            "image_size": args_copy.image_size,
            "img2dom_file": args_copy.img2dom_file,
            "data_dir": args_copy.data_path,
        }

        if domain is not None:
            kwargs["domain"] = domain

            _set = ListenerDataset(**kwargs)
        else:
            _set = SpeakerDataset(**kwargs)

        datasets[split] = _set

    load_params = {
        "batch_size": args_copy.batch_size,
        "shuffle": args_copy.shuffle,
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

    train_loader = None
    val_loader = None
    test_loader = None

    if "train" in splits:
        train_loader = DataLoader(datasets["train"], **load_params)
    if "val" in splits:
        val_loader = DataLoader(datasets["val"], **load_params_val)
    if "test" in splits:
        test_loader = DataLoader(datasets["test"], **load_params_test)

    return train_loader, test_loader, val_loader


def load_wandb_dataset(
    split: str,
    domain: str,
    load_params: Dict,
    listener_vocab: Vocab,
    speaker_model: torch.nn.Module,
    dataloader: DataLoader,
    logger: AbstractWandbLogger,
    subset_size: Optional[int] = -1,
    test_split: Optional[str] = "all",
) -> DataLoader:
    """
    Load speaker augmented dataset from wandb, if not preset generate it and upload it
    Parameters
    ----------
    split : [train,val,test]
    domain : one of the domains
    load_params : dictionary to use for the dataloader collate function
    listener_vocab :
    speaker_model
    dataloader : the original dataloader to use for generation if dataset not found on wandb
    logger : WandbLogger to restore dataset or upload it
    subset_size : reduce original size for logging

    Returns
    -------

    """

    if split == "test":
        split = f"{split}_{test_split}"

    try:
        # try to download the dataset from wandb
        file = load_wandb_file(DATASET_CHK[split][domain])
        dataset = torch.load(file)

        # crop the data according to subset size
        if subset_size != -1:
            dataset.data = {k: v for k, v in dataset.data.items() if k < subset_size}

        dl = torch.utils.data.DataLoader(dataset, **load_params)
    except wandb.errors.CommError:
        # if error then generate
        file_path = f"{split}_dataloader_{domain}.pth"

        print(f"Dataset '{file_path}' not found on wandb, generating....")
        dl = speaker_augmented_dataloader(
            dataloader,
            listener_vocab,
            speaker_model,
            split_name=split,
            load_params=load_params,
        )

        # save dataset and log to wandb
        torch.save(dl.dataset, file_path)
        logger.log_artifact(
            file_path,
            file_path.replace(".pth", ""),
            "dataset",
            epoch=0,
            metadata=dict(
                split_name=split,
                domain=domain,
                shuffle=load_params["shuffle"],
                batch_size=load_params["batch_size"],
                debug=len(logger.run.tags),
            ),
        )

    return dl


def speaker_augmented_dataloader(
    dataloader: DataLoader,
    listener_vocab: Vocab,
    speak_model: torch.nn.Module,
    split_name: str,
    load_params: Dict,
) -> DataLoader:
    """
    Augment the canon dataloader with speaker generated utterances and embeddings

    """

    if len(dataloader) == 0:
        print("Empty dataloader")
        return dataloader

    new_data = copy.deepcopy(dataloader.dataset.data)

    for ii, data in rich.progress.track(
        enumerate(dataloader),
        total=len(dataloader),
        description=f"Generating hypothesis for split '{split_name}'",
    ):
        # get datapoints
        target_img_feats = data["target_img_feats"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        visual_context = data["concat_context"]

        # generate hypo with speaker
        utterance, _, h1 = speak_model.generate_hypothesis(
            prev_utterance, prev_utt_lengths, visual_context, target_img_feats
        )

        utterance = utterance.squeeze().tolist()
        h1 = h1.squeeze().tolist()

        if not isinstance(utterance, list):
            utterance = [utterance]
        # append to new data
        new_data[ii]["speak_utterance"] = utterance
        new_data[ii]["speak_h1embed"] = h1
        new_data[ii]["speak_length"] = len(utterance)

        # show_img(data, dataloader.dataset.img_id2path,f"original_{split_name}", hypo=hypo,idx=ii)
        assert data["orig_utterance"][0] == new_data[ii]["orig_utterance"]

    new_dataset = SpeakerUttDataset(new_data, dataloader.dataset.domain)

    # check if present
    dp = next(iter(new_dataset)).keys()
    assert (
        "speak_utterance" in dp and "speak_h1embed" in dp
    ), "dataloader update did not work"

    dataloader = torch.utils.data.DataLoader(new_dataset, **load_params)

    return dataloader


def wandb2rich_table(table: wandb.Table) -> rich.table.Table:
    """
    Convert wandb table to rich table
    """

    rich_table = rich.table.Table(show_header=True, header_style="bold magenta")
    for col in table.columns:
        rich_table.add_column(col)

    rows = table.data

    # sort rows based on the third column
    rows.sort(key=lambda x: x[2], reverse=False)

    for row in rows:
        r = [str(x) for x in row]

        rich_table.add_row(*r)

    return rich_table
