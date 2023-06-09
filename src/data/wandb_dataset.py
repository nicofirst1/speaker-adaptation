import numpy as np
import torch
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, get_dataloaders,
                         load_wandb_checkpoint, load_wandb_dataset, parse_args)
from src.data.dataloaders import AbstractDataset, Vocab
from src.models import ListenerModel, SpeakerModel
from src.wandb_logging import ListenerLogger


def compute_domain(common_p):
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

    domain = common_p.train_domain

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
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

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

    speaker_model = SpeakerModel(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.beam_size,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
        use_beam=common_speak_p.use_beam,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ###################################
    ##  LOGGER
    ###################################

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating the new dataloaders
    common_p.batch_size = 1
    common_p.shuffle = False

    shuffle = common_p.shuffle
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
        list_vocab,
        speaker_model,
        training_loader,
        logger,
        subset_size=common_p.subset_size,
    )

    # eval

    load_wandb_dataset(
        "val",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        val_loader,
        logger,
        subset_size=common_p.subset_size,
    )

    load_wandb_dataset(
        "test",
        domain,
        load_params,
        list_vocab,
        speaker_model,
        test_loader,
        logger,
        subset_size=common_p.subset_size,
        test_split=common_p.test_split,
    )


if __name__ == "__main__":

    splits = ["all", "seen", "unseen"]
    common_p = parse_args("list")

    logger = ListenerLogger(
        vocab=None,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        project="speaker-gen-data",
    )

    for s in splits:
        common_p.test_split = s
        compute_domain(common_p)
