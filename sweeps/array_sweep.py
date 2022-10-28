import os
from time import sleep

import torch
import wandb
import yaml
from src.commons import parse_args

if __name__ == "__main__":

    TRIALS= 20
    common_p = parse_args("int")

    if "speaker_sweep.json" in common_p.sweep_file:
        common_p = parse_args("speak")

    with open(common_p.sweep_file, "r") as stream:
        sweep_config = yaml.safe_load(stream)

    # add train domain
    sweep_config["parameters"].update(
        dict(train_domain=dict(value=common_p.train_domain), sweep_file=dict(value=common_p.sweep_file))
    )
    sweepid = wandb.sweep(
        sweep_config, project=sweep_config["project"], entity="adaptive-speaker"
    )

    cuda_devices=torch.cuda.device_count()

    for i in range(cuda_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        wandb.agent(sweepid, count=TRIALS//cuda_devices)
        sleep(10)
