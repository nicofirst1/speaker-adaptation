##########################
# WANDB CHECKPOINTS
##########################

########
# LIST
########
LISTENER_CHK_DICT = dict(
    all="adaptive-speaker/listener/ListenerModel_all:v184",
    appliances="adaptive-speaker/listener/ListenerModel_appliances:v297",
    food="adaptive-speaker/listener/ListenerModel_no_hist_food:v32",
    indoor="adaptive-speaker/listener/ListenerModel_indoor:v293",
    outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v274",
    vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v248",
)

# listeners checkpoint with various values of golden_data_percent
LISTENER_CHK_DICT_1 = dict(
    all="adaptive-speaker/listener/ListenerModel_all:v199",
    appliances="adaptive-speaker/listener/ListenerModel_appliances:v339",
    food="adaptive-speaker/listener/ListenerModel_food:v274",
    indoor="adaptive-speaker/listener/ListenerModel_indoor:v331",
    outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v520",
    vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v287",
)

#
# LISTENER_CHK_DICT_0 = dict(
#     all="adaptive-speaker/listener/ListenerModel_all:v209",
#     appliances="",
#     food="adaptive-speaker/listener/ListenerModel_food:v276",
#     indoor="",
#     outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v417",
#     vehicles="",
# )
# LISTENER_CHK_DICT_05 = dict(
#     all="adaptive-speaker/listener/ListenerModel_all:v207",
#     appliances="",
#     food="adaptive-speaker/listener/ListenerModel_food:v275",
#     indoor="",
#     outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v405",
#     vehicles="",
# )


def get_listener_check(domain, golden_data_percent):
   return LISTENER_CHK_DICT_1[domain]


########
# SPEAK
########
SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v105"

########
# INT
########

SIM_CHECKPOINTS = dict(
    
    all="",
    food="",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

SIM_CHECKPOINTS_1 = dict(
    
    all="adaptive-speaker/simulator-pretrain/SimulatorModel:v951",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel:v918",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel:v1010",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v1016",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v867",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel:v1012",
)

# SIM_CHECKPOINTS_05 = dict(
#
#     all="adaptive-speaker/simulator-pretrain/SimulatorModel:v971",
#     food="adaptive-speaker/simulator-pretrain/SimulatorModel:v909",
#     appliances="",
#     indoor="",
#     outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v870",
#     vehicles="",
# )
#
#
# SIM_CHECKPOINTS_0 = dict(
#
#     all="adaptive-speaker/simulator-pretrain/SimulatorModel:v970",
#     food="adaptive-speaker/simulator-pretrain/SimulatorModel:v866",
#     appliances="",
#     indoor="",
#     outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel:v943",
#     vehicles="",
# )


def get_simulator_check(domain, golden_data_percent):
   return SIM_CHECKPOINTS_1[domain]


#
# SIM_CHECKPOINTS = dict(
#     
#     all="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_all:v251",
#     food="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_food:v97",
#     appliances="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_appliances:v109",
#     indoor="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_indoor:v129",
#     outdoor="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_outdoor:v242",
#     vehicles="adaptive-speaker/simulator-pretrain/InterpreterModel_tom_vehicles:v205",
# )


########
# DATA
########
TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/train_dataloader_all:latest",
    food="adaptive-speaker/speaker-gen-data/train_dataloader_food:latest",
    appliances="adaptive-speaker/speaker-gen-data/train_dataloader_appliances:latest",
    indoor="adaptive-speaker/speaker-gen-data/train_dataloader_indoor:latest",
    outdoor="adaptive-speaker/speaker-gen-data/train_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/speaker-gen-data/train_dataloader_vehicles:latest",
)

VAL_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/val_dataloader_all:latest",
    food="adaptive-speaker/speaker-gen-data/val_dataloader_food:latest",
    appliances="adaptive-speaker/speaker-gen-data/val_dataloader_appliances:latest",
    indoor="adaptive-speaker/speaker-gen-data/val_dataloader_indoor:latest",
    outdoor="adaptive-speaker/speaker-gen-data/val_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/speaker-gen-data/val_dataloader_vehicles:latest",
)

TEST_ALL_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_all_dataloader_all:latest",
    food="adaptive-speaker/speaker-gen-data/test_all_dataloader_food:latest",
    appliances="adaptive-speaker/speaker-gen-data/test_all_dataloader_appliances:latest",
    indoor="adaptive-speaker/speaker-gen-data/test_all_dataloader_indoor:latest",
    outdoor="adaptive-speaker/speaker-gen-data/test_all_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/speaker-gen-data/test_all_dataloader_vehicles:latest",
)

TEST_SEEN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_seen_dataloader_all:latest",
    food="adaptive-speaker/speaker-gen-data/test_seen_dataloader_food:latest",
    appliances="adaptive-speaker/speaker-gen-data/test_seen_dataloader_appliances:latest",
    indoor="adaptive-speaker/speaker-gen-data/test_seen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/speaker-gen-data/test_seen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/speaker-gen-data/test_seen_dataloader_vehicles:latest",
)

TEST_UNSEEN_DATASET_CHK = dict(
    all="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_all:latest",
    food="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_food:latest",
    appliances="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_appliances:latest",
    indoor="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/speaker-gen-data/test_unseen_dataloader_vehicles:latest",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test_all=TEST_ALL_DATASET_CHK,
    test_seen=TEST_SEEN_DATASET_CHK,
    test_unseen=TEST_UNSEEN_DATASET_CHK,
)
