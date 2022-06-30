
##########################
# WANDB CHECKPOINTS
##########################

########
# LIST
########
LISTENER_CHK_DICT = dict(
    all="adaptive-speaker/listener/ListenerModel_all:v184",
    appliances="adaptive-speaker/listener/ListenerModel_appliances:v297",
    food="adaptive-speaker/listener/ListenerModel_food:v225",
    indoor="adaptive-speaker/listener/ListenerModel_indoor:v293",
    outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v274",
    vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v248",
)

########
# SPEAK
########
SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v20"

########
# SIM
########
SIM_ALL_CHK = "adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_all:latest"


SIM_DOMAIN_CHK = dict(
    # epoch 83
    all=SIM_ALL_CHK,
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_food:v962",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_appliances:v971",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_indoor:v778",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_outdoor:v679",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_vehicles:v798",
)

########
# DATA
########
TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-eval/train_dataloader_all:latest",
    food="adaptive-speaker/simulator-eval/train_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-eval/train_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-eval/train_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-eval/train_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-eval/train_dataloader_vehicles:latest",
)


VAL_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-eval/val_dataloader_all:latest",
    food="adaptive-speaker/simulator-eval/val_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-eval/val_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-eval/val_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-eval/val_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-eval/val_dataloader_vehicles:latest",
)

TEST_ALL_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-eval/test_all_dataloader_all:latest",
    food="adaptive-speaker/simulator-eval/test_all_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-eval/test_all_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-eval/test_all_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-eval/test_all_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-eval/test_all_dataloader_vehicles:latest",
)


TEST_SEEN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-eval/test_seen_dataloader_all:latest",
    food="adaptive-speaker/simulator-eval/test_seen_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-eval/test_seen_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-eval/test_seen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-eval/test_seen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-eval/test_seen_dataloader_vehicles:latest",
)

TEST_UNSEEN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-eval/test_unseen_dataloader_all:latest",
    food="adaptive-speaker/simulator-eval/test_unseen_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-eval/test_unseen_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-eval/test_unseen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-eval/test_unseen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-eval/test_unseen_dataloader_vehicles:latest",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test_all=TEST_ALL_DATASET_CHK,
    test_seen=TEST_SEEN_DATASET_CHK,
    test_unseen=TEST_UNSEEN_DATASET_CHK,
)
