import os
from time import sleep

import torch
from joblib._multiprocessing_helpers import mp

import wandb
import yaml
from src.commons import parse_args

if __name__ == "__main__":

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
    procs = []
    for i in range(cuda_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        proc = mp.Process(target=wandb.agent, args=( sweepid,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

