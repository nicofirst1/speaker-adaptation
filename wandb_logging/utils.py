import json
import os
from typing import Dict, List, Tuple

import torch


def imgid2path(data_path: str) -> Dict[str, str]:
    """
    Return a dict mapping image id with the path in data
    :param data_path:
    :return:
    """
    image_path = os.path.join(data_path, "photobook_coco_images")

    images = [f for f in os.listdir(image_path) if "jpg" in f]
    imgs_ids = [int(x.rsplit("_", 1)[1].split(".")[0]) for x in images]

    images = [os.path.join(image_path, x) for x in images]

    return dict(zip(imgs_ids, images))


def imgid2domain(data_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Return a dict correlating image id to image domain and a list of all domains
    :param data_path: location of data
    :return:
    """
    chains_path = os.path.join(data_path, "chains-domain-specific", "speaker")
    chain_dict = {}
    for split in ["train", "test", "val"]:
        with open(os.path.join(chains_path, f"{split}.json"), "r") as file:
            chain_dict.update(json.load(file))

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


def save_model(
        model, model_type, epoch, accuracy, optimizer, args, timestamp, logger, **kwargs
):
    seed = args.seed
    file_name = (
            "saved_models/"
            + model_type
            + "_"
            + str(seed)
            + "_"
            + timestamp
            + ".pkl"
    )

    save_dict = {
        "accuracy": accuracy,
        "args": args,  # more detailed info, metric, model_type etc
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    save_dict.update(kwargs)
    torch.save(save_dict,file_name)
    logger.save_model(file_name, type(model).__name__, epoch, args)