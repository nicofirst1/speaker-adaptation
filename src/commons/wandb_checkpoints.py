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
SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v105"

########
# SIM
########

# regex for simulator checks is: SIM_{model_type}_{pretrain_loss}:
# model_type:
#      1. no_hist: predicts list out without using context hist
#      2. hist: predicts list out using context hist
#      3. binary: predicts if the list will be correct or not
#      4. domain: predicts datapoint domain
# pretrain_loss:
#   pretrain loss for simulator == listener out,
#       1. cross entropy [ce]
#       2. Kullback-Leibler Divergence [kl]
#   the following work only with model_type = binary
#       3. Binary cross entropy [bce]
#       4. Focal bce [fbce]


SIM_NOHIST_CE_CHK = dict(
    # epoch 83
    all="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_all:v1346",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_food:v962",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_appliances:v971",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_indoor:v1036",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_outdoor:v679",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_vehicles:v1058",
)

SIM_NOHIST_KL_CHK = dict(
    # epoch 20
    all="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_all:v1140",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_food:",
    appliances="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_appliances:v1218",
    indoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_indoor:v934",
    outdoor="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_outdoor:v843",
    vehicles="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_vehicles:v932",
)

SIM_BINARY_BCE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_binary_food:v223",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

SIM_BINARY_FBCE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_binary_food:v171",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

SIM_DOMAIN_CE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_domain_food:v139",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

SIM_DOMAIN_KL_OOD_CHK = dict(
    # epoch 20
    all="",
    food="adaptive-speaker/simulator-pretrain/SimulatorModel_no_hist_food:v1399",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

SIM_CHECKPOINTS = dict(
    SIM_NOHIST_CE_CHK=SIM_NOHIST_CE_CHK,
    SIM_NOHIST_KL_CHK=SIM_NOHIST_KL_CHK,
    SIM_BINARY_BCE_CHK=SIM_BINARY_BCE_CHK,
    SIM_BINARY_FBCE_CHK=SIM_BINARY_FBCE_CHK,
    SIM_DOMAIN_CE_CHK=SIM_DOMAIN_CE_CHK,
    SIM_DOMAIN_KL_OOD_CHK=SIM_DOMAIN_KL_OOD_CHK,

)


def get_sim_chk(type_of_sim, model_type, pretrain_loss, domain):
    """
    Return the correct simulator checkpoint
    """

    sim_chk="SIM_"+model_type.replace("_","").upper()+"_"+pretrain_loss.upper()+"_CHK"
    sim_chk=SIM_CHECKPOINTS[sim_chk][domain]

    return sim_chk


########
# DATA
########
TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-data/train_dataloader_all:latest",
    food="adaptive-speaker/simulator-data/train_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-data/train_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-data/train_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-data/train_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-data/train_dataloader_vehicles:latest",
)

VAL_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-data/val_dataloader_all:latest",
    food="adaptive-speaker/simulator-data/val_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-data/val_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-data/val_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-data/val_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-data/val_dataloader_vehicles:latest",
)

TEST_ALL_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-data/test_all_dataloader_all:latest",
    food="adaptive-speaker/simulator-data/test_all_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-data/test_all_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-data/test_all_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-data/test_all_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-data/test_all_dataloader_vehicles:latest",
)

TEST_SEEN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-data/test_seen_dataloader_all:latest",
    food="adaptive-speaker/simulator-data/test_seen_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-data/test_seen_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-data/test_seen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-data/test_seen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-data/test_seen_dataloader_vehicles:latest",
)

TEST_UNSEEN_DATASET_CHK = dict(
    all="adaptive-speaker/simulator-data/test_unseen_dataloader_all:latest",
    food="adaptive-speaker/simulator-data/test_unseen_dataloader_food:latest",
    appliances="adaptive-speaker/simulator-data/test_unseen_dataloader_appliances:latest",
    indoor="adaptive-speaker/simulator-data/test_unseen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/simulator-data/test_unseen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/simulator-data/test_unseen_dataloader_vehicles:latest",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test_all=TEST_ALL_DATASET_CHK,
    test_seen=TEST_SEEN_DATASET_CHK,
    test_unseen=TEST_UNSEEN_DATASET_CHK,
)
