import argparse
import copy
import os.path
import uuid
from typing import Optional, Tuple

import numpy as np
import psutil
import rich
import torch
from PIL import ImageFont, ImageDraw,Image
from torch.utils.data import DataLoader

from src.commons.model_utils import hypo2utterance
from src.data.dataloaders import AbstractDataset, ListenerDataset, SpeakerDataset, Vocab, ModifiedDataset


def show_img(data,id2path,split_name,hypo="",idx=-1):

    log_dir=f"img_hypo_captions/{split_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    batch_size=data['target'].shape[0]
    rnd_idx = np.random.randint(0, batch_size)

    caption = data['orig_utterance'][rnd_idx]
    target = data['image_set'][rnd_idx][data['target'][rnd_idx]]
    target_id=target
    img_set=data['image_set'][rnd_idx]
    speak_utt=data['speak_utterance'][rnd_idx]

    if isinstance(speak_utt,torch.Tensor):
        speak_utt=speak_utt.tolist()

    target = id2path[str(target)]

    img = Image.open(target)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 16)
    draw.rectangle([0, 0, img.size[0], 50], fill=(0,0,0))

    to_write=f"Hypo :{hypo}\nCaption : {caption}\n Speak ids :{speak_utt}\n Img_set: {img_set}"

    draw.text((0, 0),to_write , (255, 255, 255), font=font)
    img.show()

    if idx<0:
        idx=len([name for name in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, name))])
    #img.save(os.path.join(log_dir,f"{idx}_{target_id}.png"))



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

    new_data=copy.deepcopy(dataloader.dataset.data)

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
        utterance=utterance.squeeze().tolist()
        h1=h1.squeeze().tolist()

        if not isinstance(utterance,list):
            utterance=[utterance]

        new_data[ii]["speak_utterance"] = utterance
        new_data[ii]["speak_h1embed"] = h1


        #show_img(data, dataloader.dataset.img_id2path,f"original_{split_name}", hypo=hypo,idx=ii)
        assert data['orig_utterance'][0]==new_data[ii]['orig_utterance']


    new_dataset=ModifiedDataset(new_data)
    dp = next(iter(new_dataset)).keys()
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

    dataloader = torch.utils.data.DataLoader(new_dataset, **load_params)

    return dataloader
