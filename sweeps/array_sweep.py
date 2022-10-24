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
        sweep_config, project=sweep_config["project"], entity="sweep"
    )
    wandb.agent(sweepid)
