import argparse
import concurrent
import copy
import os.path
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import rich.progress
import rich.table
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from rich.progress import Progress
from torch.utils.data import DataLoader

from src.commons.model_utils import load_wandb_file
from src.commons.wandb_checkpoints import DATASET_CHK
from src.data.dataloaders import (
    AbstractDataset,
    ListenerDataset,
    SpeakerDataset,
    SpeakerUttDataset,
    Vocab,
)
from src.wandb_logging import AbstractWandbLogger


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
    speaker_model: torch.nn.Module,
    dataloader: DataLoader,
    logger: AbstractWandbLogger,
    test_split: Optional[str] = "seen",
    subset_size: Optional[int] = -1,
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
            # get subset_size random keys from dataset.data
            keys = random.sample(list(dataset.data.keys()), subset_size)
            # create a new dataset with the subset
            data = {k: dataset.data[k] for k in keys}

            # replace the keys with a new range
            data = {i: data[k] for i, k in enumerate(data.keys())}
            dataset.data = data

        dl = torch.utils.data.DataLoader(dataset, **load_params)
    except wandb.errors.CommError:
        # if error then generate
        file_path = f"{split}_dataloader_{domain}.pth"

        print(f"Dataset '{file_path}' not found on wandb, generating....")
        dl = speaker_augmented_dataloader(
            dataloader,
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


def augment_data_subset(
    progress: Progress,
    task_id: int,
    data_subset,
    speak_model: torch.nn.Module,
) -> List[Dict]:
    augmented_data = []

    for data in data_subset:
        target_img_feats = data["target_img_feats"].unsqueeze(0)
        context_separate = data["separate_images"].unsqueeze(0)

        utterance, _, h0 = speak_model.generate_hypothesis(
            context_separate, target_img_feats
        )

        utterance = utterance.squeeze().tolist()
        h0 = h0.squeeze().tolist()

        if not isinstance(utterance, list):
            utterance = [utterance]

        data["speak_utterance"] = utterance
        data["speak_embed"] = h0
        data["speak_length"] = len(utterance)
        augmented_data.append(data)

        progress.update(task_id, advance=1)

    return augmented_data


def speaker_augmented_dataloader(
    dataloader: DataLoader,
    speak_model: torch.nn.Module,
    split_name: str,
    load_params: Dict,
) -> DataLoader:
    if len(dataloader) == 0:
        print("Empty dataloader")
        return dataloader

    new_data = copy.deepcopy(dataloader.dataset.data)

    n_threads = 8

    # Convert data dictionary to list of dictionaries
    new_data = [new_data[idx] for idx in range(len(new_data))]

    # Check if new_data is not empty
    if len(new_data) == 0:
        print("No data to process")
        return dataloader

    # Split the data into subsets
    data_subsets = np.array_split(new_data, n_threads)

    # Create a shared Progress object
    progress = Progress()
    progress.start()

    # Create tasks for each subset
    tasks = [
        progress.add_task(
            f"[cyan]Generating hypothesis for split '{split_name}' (Thread {i + 1})",
            total=len(subset),
        )
        for i, subset in enumerate(data_subsets)
    ]

    # Use ThreadPoolExecutor to process subsets concurrently
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_results = [
            executor.submit(
                augment_data_subset, progress, tasks[i], data_subsets[i], speak_model
            )
            for i in range(n_threads)
        ]
        augmented_data = []
        for future in concurrent.futures.as_completed(future_results):
            augmented_data.extend(future.result())

    progress.stop()

    new_dataset = SpeakerUttDataset(augmented_data, dataloader.dataset.domain)

    # check if present
    dp = next(iter(new_dataset)).keys()
    assert (
        "speak_utterance" in dp and "speak_embed" in dp
    ), "dataloader update did not work"

    dataloader = torch.utils.data.DataLoader(new_dataset, **load_params)

    return dataloader


def speaker_augmented_dataloader_old(
    dataloader: DataLoader,
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
        context_separate = data["separate_images"]

        # generate hypo with speaker
        utterance, _, h0 = speak_model.generate_hypothesis(
            context_separate, target_img_feats
        )

        utterance = utterance.squeeze().tolist()
        h0 = h0.squeeze().tolist()

        if not isinstance(utterance, list):
            utterance = [utterance]
        # append to new data
        new_data[ii]["speak_utterance"] = utterance
        new_data[ii]["speak_embed"] = h0
        new_data[ii]["speak_length"] = len(utterance)

        # show_img(data, dataloader.dataset.img_id2path,f"original_{split_name}", hypo=hypo,idx=ii)
        assert data["orig_utterance"][0] == new_data[ii]["orig_utterance"]

    new_dataset = SpeakerUttDataset(new_data, dataloader.dataset.domain)

    # check if present
    dp = next(iter(new_dataset)).keys()
    assert (
        "speak_utterance" in dp and "speak_embed" in dp
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
