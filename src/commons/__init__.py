from .data_utils import (get_dataloaders, load_wandb_dataset,
                         speaker_augmented_dataloader)
from .EarlyStopping import EarlyStopping
from .model_utils import (get_domain_accuracy, hypo2utterance,
                          load_wandb_checkpoint, mask_attn, merge_dict,
                          save_model)
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
    "speaker_augmented_dataloader",
    "load_wandb_dataset",
    "merge_dict",
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

# SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel:v203"

SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v20"
SIM_ALL_CHK = "adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_all:v699"


SIM_DOMAIN_CHK = dict(
    # epoch 83
    all=SIM_ALL_CHK,
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_food:v962",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_appliances:v971",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_indoor:v778",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_outdoor:v679",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_vehicles:v798",
)


TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-pretrain/train_dataloader_all:latest",
    food="adaptive-speaker/simulator-pretrain/train_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-pretrain/train_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-pretrain/train_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-pretrain/train_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-pretrain/train_dataloader_vehicles:latest",
)


VAL_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-pretrain/val_dataloader_all:latest",
    food="adaptive-speaker/simulator-pretrain/val_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-pretrain/val_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-pretrain/val_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-pretrain/val_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-pretrain/val_dataloader_vehicles:latest",
)

TEST_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-pretrain/test_dataloader_all:latest",
    food="adaptive-speaker/simulator-pretrain/test_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-pretrain/test_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-pretrain/test_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-pretrain/test_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-pretrain/test_dataloader_vehicles:latest",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test=TEST_DATASET_CHK,
)
