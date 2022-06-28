import argparse
import copy
import os.path
from typing import Dict, Optional, Tuple

import numpy as np
import rich
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

import wandb
from src.commons.model_utils import hypo2utterance, load_wandb_file
from src.data.dataloaders import (AbstractDataset, ListenerDataset,
                                  SpeakerDataset, SpeakerUttDataset, Vocab)


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


def load_wandb_dataset(
    split: str,
    domain: str,
    load_params: Dict,
    list_vocab: Vocab,
    speaker_model: torch.nn.Module,
    dataloader: DataLoader,
    logger,
    DATASET_CHK: Dict,
    subset_size: Optional[int] = -1,
) -> DataLoader:
    """
    Load speaker augmented dataset from wandb, if not preset generate it and upload ity
    Parameters
    ----------
    split
    domain
    load_params
    list_vocab
    speaker_model
    dataloader
    logger
    DATASET_CHK
    subset_size

    Returns
    -------

    """
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
            list_vocab,
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
    vocab: Vocab,
    speak_model: torch.nn.Module,
    split_name: str,
    load_params,
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
        utterance = utterance.squeeze().tolist()
        h1 = h1.squeeze().tolist()

        if not isinstance(utterance, list):
            utterance = [utterance]

        new_data[ii]["speak_utterance"] = utterance
        new_data[ii]["speak_h1embed"] = h1

        # show_img(data, dataloader.dataset.img_id2path,f"original_{split_name}", hypo=hypo,idx=ii)
        assert data["orig_utterance"][0] == new_data[ii]["orig_utterance"]

    new_dataset = SpeakerUttDataset(new_data, dataloader.dataset.domain)
    dp = next(iter(new_dataset)).keys()
    assert (
        "speak_utterance" in dp and "speak_h1embed" in dp
    ), "dataloader update did not work"

    dataloader = torch.utils.data.DataLoader(new_dataset, **load_params)

    return dataloader
