import numpy as np
import torch

from src.commons import (
    LISTENER_CHK_DICT,
    SPEAKER_CHK,
    get_dataloaders,
    load_wandb_checkpoint,
    load_wandb_dataset,
    parse_args,
)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import ListenerModel
from src.models.speaker.SpeakerModelEC import SpeakerModelEC
from src.wandb_logging import WandbLogger


def compute_domain(common_p, domain):
    """
    Augment dataloader with speaker utterances and embeddings.
    Ran this script once to upload everything on wanbd
    Parameters
    ----------
    domain

    Returns
    -------

    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(
        LISTENER_CHK_DICT[domain],
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.reset_paths()

    # for reproducibility
    seed = list_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    list_model = ListenerModel(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK,
        device,
    )  # datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1]))
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    # init speak model and load state

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ###################################
    ##  LOGGER
    ###################################

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    shuffle = False
    training_loader, test_loader, val_loader = get_dataloaders(
        common_p, speak_vocab, domain
    )

    # train
    load_params = {
        "batch_size": bs,
        "shuffle": shuffle,
        "collate_fn": AbstractDataset.get_collate_fn(
            speaker_model.device,
            list_vocab["<sos>"],
            list_vocab["<eos>"],
            list_vocab["<nohs>"],
        ),
    }

    load_wandb_dataset(
        "train",
        domain,
        load_params,
        speaker_model,
        training_loader,
        logger,
    )

    # eval

    load_wandb_dataset(
        "val",
        domain,
        load_params,
        speaker_model,
        val_loader,
        logger,
    )

    load_wandb_dataset(
        "test",
        domain,
        load_params,
        speaker_model,
        test_loader,
        logger,
        test_split=common_p.test_split,
    )


if __name__ == "__main__":
    common_p = parse_args("list")
    common_p.test_split = "seen"

    logger = WandbLogger(
        vocab=None,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        project="speaker-gen-data",
    )
    domains = ["food", "appliances", "vehicles", "outdoor", "indoor"]

    for d in domains:
        compute_domain(common_p, d)
