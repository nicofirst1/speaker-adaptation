import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str,
        chain_file: str,
        utterances_file: str,
        vectors_file: str,
        orig_ref_file: str,
        img2dom_file: str,
        subset_size: int = -1,
        image_size: int = 2048,
    ):
        """
        Abstract dataclass that implements
        Parameters
        ----------
        split : either [train, val, test], the split to load from the dataset
        data_dir: data dir where to load the data from
        chain_file : name of the chain file
        utterances_file : name of the utterance file
        vectors_file : name of the vector file, can be either vectors or clip and is related to the image size
        orig_ref_file : name of the original utterance file
        img2dom_file: dict mapping image id to domains
        subset_size : how much of the dataset to load, default -1 for all
        image_size : the image size
        """

        self.data_dir = data_dir
        self.split = split


        # Load a PhotoBook utterance chain dataset
        with open(os.path.join(self.data_dir, chain_file), "r") as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue utterance dataset
        with open(os.path.join(self.data_dir, utterances_file), "rb") as file:
            self.utterances = pickle.load(file)

        # Load pre-defined image features
        with open(vectors_file, "r") as file:
            self.image_features = json.load(file)

        # Original reference sentences without unks
        with open(os.path.join(self.data_dir, orig_ref_file), "rb") as file:
            self.text_refs = pickle.load(file)

        # Original reference sentences without unks
        with open(img2dom_file, "r") as file:
            self.img2dom = json.load(file)

        self.img_dim = image_size
        self.img_count = 6  # images in the context
        self.max_len = 0

        self.data = dict()
        self.missing_images = []

        self.img2chain = defaultdict(dict)

        for chain in self.chains:
            self.img2chain[chain["target"]][chain["game_id"]] = chain["utterances"]

        if subset_size == -1:
            self.subset_size = len(self.chains)

        else:
            # if we take a subset of the data then shuffle with np (use seed)
            self.subset_size = subset_size
            np.random.shuffle(self.chains)

        self.load_data()

    def load_data(self):
        """
        Load every datapoint in the self.data attribute
        Returns
        -------

        """
        print("processing", self.split)

        # every utterance in every chain, along with the relevant history
        for chain in self.chains[: self.subset_size]:

            chain_utterances = chain["utterances"]
            game_id = chain["game_id"]

            for s in range(len(chain_utterances)):

                # this is the expected target generation
                utterance_id = tuple(
                    chain_utterances[s]
                )  # utterance_id = (game_id, round_nr, messsage_nr, img_id)
                round_nr = utterance_id[1]
                message_nr = utterance_id[2]

                # prev utterance in the chain
                for cu in range(len(chain["utterances"])):

                    if chain["utterances"][cu] == list(utterance_id):
                        if cu == 0:
                            previous_utterance = []
                        else:
                            prev_id = chain["utterances"][cu - 1]
                            previous_utterance = self.utterances[tuple(prev_id)][
                                "utterance"
                            ]

                        break

                # linguistic histories for images in the context
                # HISTORY before the expected generation (could be after the encoded history)
                prev_chains = defaultdict(list)
                prev_lengths = defaultdict(int)

                cur_utterance_obj = self.utterances[utterance_id]
                cur_utterance_text_ids = cur_utterance_obj["utterance"]

                orig_target = self.text_refs[utterance_id]["utterance"]
                orig_target = " ".join(orig_target)

                length = cur_utterance_obj["length"]

                if length > self.max_len:
                    self.max_len = length

                # assert len(cur_utterance_text_ids) != 2
                # already had added sos eos into length and IDS version

                images = cur_utterance_obj["image_set"]
                target = cur_utterance_obj["target"]  # index of correct img

                target_image = images[target[0]]

                images = list(np.random.permutation(images))
                target = [images.index(target_image)]

                context_separate = torch.zeros(self.img_count, self.img_dim)

                im_counter = 0

                reference_chain = []

                for im in images:

                    context_separate[im_counter] = torch.tensor(self.image_features[im])

                    if im == images[target[0]]:
                        target_img_feats = context_separate[im_counter]
                        ref_chain = self.img2chain[im][game_id]

                        for rc in ref_chain:
                            rc_tuple = (rc[0], rc[1], rc[2], im)
                            reference_chain.append(
                                " ".join(self.text_refs[rc_tuple]["utterance"])
                            )

                    im_counter += 1

                    if (
                        game_id in self.img2chain[im]
                    ):  # was there a linguistic chain for this image in this game
                        temp_chain = self.img2chain[im][game_id]

                        hist_utterances = []

                        for t in range(len(temp_chain)):

                            _, t_round, t_message, _ = temp_chain[
                                t
                            ]  # (game_id, round_nr, messsage_nr, img_id)

                            if t_round < round_nr:
                                hist_utterances.append((game_id, t_round, t_message))

                            elif t_round == round_nr:

                                if t_message < message_nr:
                                    hist_utterances.append(
                                        (game_id, t_round, t_message)
                                    )

                        if len(hist_utterances) > 0:

                            # ONLY THE MOST RECENT history
                            for hu in [hist_utterances[-1]]:
                                hu_tuple = (hu[0], hu[1], hu[2], im)
                                prev_chains[im].extend(
                                    self.utterances[hu_tuple]["utterance"]
                                )

                        else:
                            # no prev reference to that image
                            prev_chains[im] = []

                    else:
                        # image is in the game but never referred to
                        prev_chains[im] = []

                    prev_lengths[im] = len(prev_chains[im])

                # ALWAYS 6 IMAGES IN THE CONTEXT

                context_concat = context_separate.reshape(self.img_count * self.img_dim)

                if target_image not in self.img2dom.keys():
                    if target_image not in self.missing_images:
                        self.missing_images.append(target_image)
                        print(f"No domain for image '{target_image}'")

                    domain = "unk"

                else:
                    domain = self.img2dom[target_image]

                self.data[len(self.data)] = {
                    "utterance": cur_utterance_text_ids,
                    "orig_utterance": orig_target,  # without unk, eos, sos, pad
                    "image_set": images,
                    "concat_context": context_concat,
                    "separate_images": context_separate,
                    "prev_utterance": previous_utterance,
                    "prev_length": len(previous_utterance),
                    "target": target,
                    "target_img_feats": target_img_feats,
                    "length": length,
                    "prev_histories": prev_chains,
                    "prev_history_lengths": prev_lengths,
                    "reference_chain": reference_chain,
                    "domain": domain,
                }

    def change_data(self, new_data:Dict):
        self.data=new_data

    @staticmethod
    def get_collate_fn(device, SOS, EOS, NOHS):
        def collate_fn(data):

            max_utt_length = max(d["length"] for d in data)
            max_prevutt_length = max([d["prev_length"] for d in data])
            if "speak_utterance" in data[0].keys():
                max_speak_length = max(len(d["speak_utterance"]) for d in data)

            batch = defaultdict(list)

            for sample in data:

                for key in data[0].keys():

                    if key == "utterance":

                        padded = sample[key] + [0] * (max_utt_length - sample["length"])

                        # print('utt', padded)
                    elif key == "speak_utterance":

                        padded = sample[key] + [0] * (  max_speak_length - len(sample[key])  )

                    elif key == "prev_utterance":

                        if len(sample[key]) == 0:
                            # OTHERWISE pack_padded wouldn't work
                            padded = [NOHS] + [0] * (
                                max_prevutt_length - 1
                            )  # SPECIAL TOKEN FOR NO HIST

                        else:
                            padded = sample[key] + [0] * (
                                max_prevutt_length - len(sample[key])
                            )

                        # print('prevutt', padded)

                    elif key == "prev_length":

                        if sample[key] == 0:
                            # wouldn't work in pack_padded
                            padded = 1

                        else:
                            padded = sample[key]

                    elif key == "image_set":

                        padded = [int(img) for img in sample["image_set"]]

                        # print('img', padded)

                    elif key == "prev_histories":

                        padded = sample["prev_histories"]

                    else:
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                # print(key)

                if key in ["separate_images", "concat_context", "target_img_feats"]:
                    batch[key] = torch.stack(batch[key])

                elif key in [
                    "utterance",
                    "prev_utterance",
                    "target",
                    "length",
                    "prev_length",
                    "speak_utterance",
                ]:
                    batch[key] = torch.Tensor(batch[key]).long()
                elif key == "speak_h1embed":
                    batch[key] = torch.Tensor(batch[key]).float()

                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

                    # for instance targets can be long and sent to device immediately

            return batch

        return collate_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]





def imgid2path(data_path: str) -> Dict[str, str]:
    """
    Return a dict mapping image id with the path in data
    :param data_path:
    :return:
    """
    image_path = os.path.join(data_path, "photobook_coco_images")

    images = [f for f in os.listdir(image_path) if "jpg" in f]
    imgs_ids = [int(x.rsplit("_", 1)[1].split(".")[0]) for x in images]
    imgs_ids = [str(x) for x in imgs_ids]

    images = [os.path.join(image_path, x) for x in images]

    return dict(zip(imgs_ids, images))


def load_imgid2domain(file_path: str) -> Tuple[Dict[str, str], List[str]]:
    with open(file_path, "r+") as f:
        img2domain = json.load(f)

    domains = set(img2domain.values())
    return img2domain, domains


def generate_imgid2domain(data_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Return a dict correlating image id to image domain and a list of all domains
    :param data_path: location of data
    :return:
    """
    chains_path = os.path.join(data_path, "speaker")
    # chains_path=data_path
    chain_dict = {}
    for split in ["train", "test", "val"]:
        with open(os.path.join(chains_path, f"{split}_text_chains.json"), "r") as file:
            utt = json.load(file)
            chain_dict.update(utt)

    chain_dict = {k.split("/")[1]: k.split("/")[0] for k in chain_dict.keys()}
    chain_dict = {int(k.split("_")[-1].split(".")[0]): v for k, v in chain_dict.items()}

    domain_dict = {
        "person_motorcycle": "vehicles",
        "car_motorcycle": "vehicles",
        "bus_truck": "vehicles",
        "car_truck": "vehicles",
        "person_suitcase": "outdoor",
        "person_umbrella": "outdoor",
        "person_surfboard": "outdoor",
        "person_elephant": "outdoor",
        "person_bicycle": "outdoor",
        "person_car": "outdoor",
        "person_train": "outdoor",
        "person_bench": "outdoor",
        "person_truck": "outdoor",
        "bowl_dining_table": "food",
        "cup_dining_table": "food",
        "cake_dining_table": "food",
        "person_oven": "appliances",
        "dining_table_refrigerator": "appliances",
        "person_refrigerator": "appliances",
        "dining_table_laptop": "indoor",
        "couch_laptop": "indoor",
        "person_bed": "indoor",
        "person_couch": "indoor",
        "person_tv": "indoor",
        "couch_dining_table": "indoor",
        "person_teddy_bear": "indoor",
        "chair_couch": "indoor",
    }

    chain_dict = {k: domain_dict[v] for k, v in chain_dict.items()}

    domains = list(set(domain_dict.values()))
    return chain_dict, domains
