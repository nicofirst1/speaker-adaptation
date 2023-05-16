import os
import sys

import torch
import yaml
from joblib._multiprocessing_helpers import mp

import wandb
from src.commons import parse_args

if __name__ == "__main__":
    common_p = parse_args("sim")

    if "speaker_sweep.json" in common_p.sweep_file:
        common_p = parse_args("speak")

    with open(common_p.sweep_file, "r") as stream:
        sweep_config = yaml.safe_load(stream)

    args = sys.argv[1:]
    args = args[0:-1:2]
    args = [x.replace("--", "") for x in args]

    for_update = {}
    for arg in args:
        for_update[arg] = dict(value=common_p.__getattribute__(arg))

    # add train domain
    sweep_config["parameters"].update(for_update)
    kwargs = dict(project=sweep_config["project"], entity="adaptive-speaker")

    if common_p.sweep_id == "":
        sweepid = wandb.sweep(
            sweep_config, **kwargs
        )
    else:
        sweepid = common_p.sweep_id

    cuda_devices = torch.cuda.device_count()
    if cuda_devices == 0:
        cuda_devices = 1
    procs = []
    for i in range(cuda_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        proc = mp.Process(target=wandb.agent, args=(sweepid,), kwargs=kwargs)
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
