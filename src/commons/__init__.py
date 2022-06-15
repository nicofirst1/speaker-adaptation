from .data_utils import get_dataloaders
from .EarlyStopping import EarlyStopping
from .model_utils import (
    get_domain_accuracy,
    hypo2utterance,
    load_wandb_checkpoint,
    mask_attn,
    save_model,
)
from .Params import parse_args

__all__ = [
    "get_dataloaders",
    "mask_attn",
    "hypo2utterance",
    "get_domain_accuracy",
    "save_model",
    "load_wandb_checkpoint",
    "parse_args",
    "EarlyStopping",
]


## take all listener at epoch 20
LISTENER_CHK_DICT = dict(
    all="adaptive-speaker/listener/ListenerModel_all:v153",
    appliances="adaptive-speaker/listener/ListenerModel_appliances:v297",
    food="adaptive-speaker/listener/ListenerModel_food:v225",
    indoor="adaptive-speaker/listener/ListenerModel_indoor:v293",
    outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v274",
    vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v248",
)

#SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel:v203"

SPEAKER_CHK="adaptive-speaker/speaker/SpeakerModel_no_hist:v20"
SIM_ALL_CHK = "adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_all:v30"

SIM_DOMAIN_CHK = dict(
    # epoch 83
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_food:v200",
    # epoch 50
    all="adaptive-speaker/simulator-pretrain/SimulatorModel_all:v89",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel_appliances:v170",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel_indoor:v12",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel_outdoor:v104",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel_vehicles:v110",
)
