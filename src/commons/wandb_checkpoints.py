##########################
# WANDB CHECKPOINTS
##########################

########
# LIST
########

# listeners checkpoint with various values of golden_data_percent
LISTENER_CHK_DICT = dict(
    all="adaptive-speaker/listener/ListenerModel_all:v199",
    appliances="adaptive-speaker/listener/ListenerModel_appliances:v339",
    food="adaptive-speaker/listener/ListenerModel_food:v274",
    indoor="adaptive-speaker/listener/ListenerModel_indoor:v331",
    outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v520",
    vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v287",
)


def get_listener_check(domain, golden_data_percent):
    return LISTENER_CHK_DICT[domain]


########
# SPEAK
########
SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v105"

########
# SIM
########


SIM_CHECKPOINTS = dict(
    all="adaptive-speaker/simulator-pretrain/SimulatorModel:v951",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel:v918",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel:v1010",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v1016",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v867",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel:v1012",
)


def get_simulator_check(domain, golden_data_percent):
    return SIM_CHECKPOINTS[domain]


########
# DATA
########
TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/train_dataloader_all:v0",
    food="adaptive-speaker/speaker-gen-data/train_dataloader_food:v0",
    appliances="adaptive-speaker/speaker-gen-data/train_dataloader_appliances:v0",
    indoor="adaptive-speaker/speaker-gen-data/train_dataloader_indoor:v0",
    outdoor="adaptive-speaker/speaker-gen-data/train_dataloader_outdoor:v0",
    vehicles="adaptive-speaker/speaker-gen-data/train_dataloader_vehicles:v0",
)

VAL_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/val_dataloader_all:v0",
    food="adaptive-speaker/speaker-gen-data/val_dataloader_food:v0",
    appliances="adaptive-speaker/speaker-gen-data/val_dataloader_appliances:v0",
    indoor="adaptive-speaker/speaker-gen-data/val_dataloader_indoor:v0",
    outdoor="adaptive-speaker/speaker-gen-data/val_dataloader_outdoor:v0",
    vehicles="adaptive-speaker/speaker-gen-data/val_dataloader_vehicles:v0",
)

TEST_ALL_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_all_dataloader_all:v0",
    food="adaptive-speaker/speaker-gen-data/test_all_dataloader_food:v0",
    appliances="adaptive-speaker/speaker-gen-data/test_all_dataloader_appliances:v0",
    indoor="adaptive-speaker/speaker-gen-data/test_all_dataloader_indoor:v0",
    outdoor="adaptive-speaker/speaker-gen-data/test_all_dataloader_outdoor:v0",
    vehicles="adaptive-speaker/speaker-gen-data/test_all_dataloader_vehicles:v0",
)

TEST_SEEN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_seen_dataloader_all:v0",
    food="adaptive-speaker/speaker-gen-data/test_seen_dataloader_food:v0",
    appliances="adaptive-speaker/speaker-gen-data/test_seen_dataloader_appliances:v0",
    indoor="adaptive-speaker/speaker-gen-data/test_seen_dataloader_indoor:v0",
    outdoor="adaptive-speaker/speaker-gen-data/test_seen_dataloader_outdoor:v0",
    vehicles="adaptive-speaker/speaker-gen-data/test_seen_dataloader_vehicles:v0",
)

TEST_UNSEEN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_all:v0",
    food="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_food:v0",
    appliances="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_appliances:v0",
    indoor="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_indoor:v0",
    outdoor="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_outdoor:v0",
    vehicles="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_vehicles:v0",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test_all=TEST_ALL_DATASET_CHK,
    test_seen=TEST_SEEN_DATASET_CHK,
    test_unseen=TEST_UNSEEN_DATASET_CHK,
)
