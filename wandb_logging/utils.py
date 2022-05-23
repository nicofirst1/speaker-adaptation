import json
import os
from os.path import join, isfile
from typing import Dict, List, Tuple

import torch
import wandb


def save_model(
        model, model_type, epoch, accuracy, optimizer, args, timestamp, logger, **kwargs
):
    seed = args.seed
    file_name = (
            model_type
            + "_"
            + str(seed)
            + "_"
            + timestamp
            + ".pth"
    )

    dir_path=join(args.working_dir,"saved_models")

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_name=join(dir_path,file_name)

    save_dict = {
        "accuracy": accuracy,
        "args": args,  # more detailed info, metric, model_type etc
        "epoch": str(epoch),
        "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
    }
    save_dict.update(kwargs)
    torch.save(save_dict,file_name,pickle_protocol=5)
    logger.save_model(file_name, type(model).__name__, epoch, args)

    print("Model saved and logged to wandb")


def load_wandb_checkpoint(url,device):
    api = wandb.Api()
    artifact = api.artifact(url)

    datadir = artifact.download()

    files = [f for f in os.listdir(datadir) if isfile(join(datadir, f))]

    if len(files) > 1:
        raise FileExistsError(f"More than one checkpoint found in {datadir}!")

    checkpoint = torch.load(join(datadir, files[0]), map_location=device)

    return checkpoint