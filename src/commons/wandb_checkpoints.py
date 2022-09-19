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

########
# SPEAK
########
SPEAKER_CHK = "adaptive-speaker/speaker/SpeakerModel_no_hist:v105"

########
# INT
########

# regex for interpreter checks is: SIM_{model_type}_{pretrain_loss}:
# model_type:
#      1. no_hist: predicts list out without using context hist
#      2. hist: predicts list out using context hist
#      3. binary: predicts if the list will be correct or not
#      4. domain: predicts datapoint domain
# pretrain_loss:
#   pretrain loss for interpreter == listener out,
#       1. cross entropy [ce]
#       2. Kullback-Leibler Divergence [kl]
#   the following work only with model_type = binary
#       3. Binary cross entropy [bce]
#       4. Focal bce [fbce]


INT_NOHIST_CE_CHK = dict(
    # epoch 10
    all="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_all:latest",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_food:v10",
    appliances="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_appliances:v10",
    indoor="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_indoor:v10",
    outdoor="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_outdoor:v10",
    vehicles="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_vehicles:v10",
)

INT_NOHIST_KL_CHK = dict(
    # epoch 20
    all="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_all:v1140",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_food:v116",
    appliances="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_appliances:v41",
    indoor="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_indoor:v39",
    outdoor="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_outdoor:v31",
    vehicles="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_vehicles:v42",
)

INT_BINARY_BCE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_binary_food:v223",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

INT_BINARY_FBCE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_binary_food:v171",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

INT_DOMAIN_CE_CHK = dict(
    # epoch 83
    all="",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_domain_food:v139",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

INT_DOMAIN_KL_OOD_CHK = dict(
    # epoch 20
    all="",
    food="adaptive-speaker/interpreter-pretrain/InterpreterModel_no_hist_food:v1399",
    appliances="",
    indoor="",
    outdoor="",
    vehicles="",
)

INT_CHECKPOINTS = dict(
    INT_NOHIST_CE_CHK=INT_NOHIST_CE_CHK,
    INT_NOHIST_KL_CHK=INT_NOHIST_KL_CHK,
    INT_BINARY_BCE_CHK=INT_BINARY_BCE_CHK,
    INT_BINARY_FBCE_CHK=INT_BINARY_FBCE_CHK,
    INT_DOMAIN_CE_CHK=INT_DOMAIN_CE_CHK,
    INT_DOMAIN_KL_OOD_CHK=INT_DOMAIN_KL_OOD_CHK,
)


def get_int_chk(model_type, pretrain_loss, domain):
    """
    Return the correct interpreter checkpoint
    """

    int_chk = (
        "INT_"
        + model_type.replace("_", "").upper()
        + "_"
        + pretrain_loss.upper()
        + "_CHK"
    )
    int_chk = INT_CHECKPOINTS[int_chk][domain]

    return int_chk


########
# DATA
########
TRAIN_DATASET_CHK = dict(
    all="adaptive-speaker/interpreter-data/train_dataloader_all:latest",
    food="adaptive-speaker/interpreter-data/train_dataloader_food:latest",
    appliances="adaptive-speaker/interpreter-data/train_dataloader_appliances:latest",
    indoor="adaptive-speaker/interpreter-data/train_dataloader_indoor:latest",
    outdoor="adaptive-speaker/interpreter-data/train_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/interpreter-data/train_dataloader_vehicles:latest",
)

VAL_DATASET_CHK = dict(
    all="adaptive-speaker/interpreter-data/val_dataloader_all:latest",
    food="adaptive-speaker/interpreter-data/val_dataloader_food:latest",
    appliances="adaptive-speaker/interpreter-data/val_dataloader_appliances:latest",
    indoor="adaptive-speaker/interpreter-data/val_dataloader_indoor:latest",
    outdoor="adaptive-speaker/interpreter-data/val_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/interpreter-data/val_dataloader_vehicles:latest",
)

TEST_ALL_DATASET_CHK = dict(
    all="adaptive-speaker/interpreter-data/test_all_dataloader_all:latest",
    food="adaptive-speaker/interpreter-data/test_all_dataloader_food:latest",
    appliances="adaptive-speaker/interpreter-data/test_all_dataloader_appliances:latest",
    indoor="adaptive-speaker/interpreter-data/test_all_dataloader_indoor:latest",
    outdoor="adaptive-speaker/interpreter-data/test_all_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/interpreter-data/test_all_dataloader_vehicles:latest",
)

TEST_SEEN_DATASET_CHK = dict(
    all="adaptive-speaker/interpreter-data/test_seen_dataloader_all:latest",
    food="adaptive-speaker/interpreter-data/test_seen_dataloader_food:latest",
    appliances="adaptive-speaker/interpreter-data/test_seen_dataloader_appliances:latest",
    indoor="adaptive-speaker/interpreter-data/test_seen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/interpreter-data/test_seen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/interpreter-data/test_seen_dataloader_vehicles:latest",
)

TEST_UNSEEN_DATASET_CHK = dict(
    all="adaptive-speaker/interpreter-data/test_unseen_dataloader_all:latest",
    food="adaptive-speaker/interpreter-data/test_unseen_dataloader_food:latest",
    appliances="adaptive-speaker/interpreter-data/test_unseen_dataloader_appliances:latest",
    indoor="adaptive-speaker/interpreter-data/test_unseen_dataloader_indoor:latest",
    outdoor="adaptive-speaker/interpreter-data/test_unseen_dataloader_outdoor:latest",
    vehicles="adaptive-speaker/interpreter-data/test_unseen_dataloader_vehicles:latest",
)

DATASET_CHK = dict(
    train=TRAIN_DATASET_CHK,
    val=VAL_DATASET_CHK,
    test_all=TEST_ALL_DATASET_CHK,
    test_seen=TEST_SEEN_DATASET_CHK,
    test_unseen=TEST_UNSEEN_DATASET_CHK,
)
