import argparse
import dataclasses
import inspect
import os.path
from dataclasses import dataclass
from os.path import join
from typing import Optional

import torch
from rich.console import Console

console = Console()


def parse_args(mode):
    if mode == "speak":
        parser = SpeakerArguments()
    elif mode == "list":
        parser = ListenerArguments()
    elif mode == "int":
        parser = InterpreterArguments()

    return parser


def get_working_dir():
    pwd = os.getcwd()
    pwd = pwd.split("pb_speaker_adaptation")[0]
    pwd = join(pwd, "pb_speaker_adaptation")
    return pwd


def get_dataset_path():
    return join(get_working_dir(), "dataset")


@dataclass
class Params:
    ############################################
    # Training params
    ############################################
    # Number of epochs for training
    epochs: Optional[int] = 60
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.0001

    # Set to true for disabling wandb logging
    debug: Optional[bool] = False
    tags: Optional[str] = ""
    device: Optional[str] = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    # This seed will be set for torch, numpy and random
    seed: Optional[int] = 42

    # Early stopping patience
    patience: Optional[int] = 30

    # If true use wandb checkpoints
    resume_train: Optional[bool] = False

    # Which simulator to use with domain specific listener.
    # Can be either [domain, general, untrained].
    # If domain then symmetric setting, else use general simulator for all domains
    type_of_int: Optional[str] = "domain"

    # reduction for crossentropy loss
    reduction: Optional[str] = "sum"

    ############################################
    # Data
    ############################################

    # -1 is the full dataset, if you put 10, it will only use 10 chains
    subset_size: Optional[int] = -1
    # Shuffle dataloader
    shuffle: Optional[bool] = False
    # If True, log listener dataset for embedding viz (computationally intensive)
    log_data: Optional[bool] = False
    # if true empty train split and load test
    is_test: Optional[bool] = False
    # the train domain data, if empty then it is set to train_domain (same as listener)
    data_domain: Optional[str] = ""

    ############################################
    # PATH
    ############################################

    wandb_dir: Optional[str] = "wandb_out"

    working_dir: str = get_working_dir()

    # Vision feature to use. Either vectors from resnet or clip
    vectors_file: Optional[str] = "vectors.json"

    sweep_file: Optional[str] = ""
    test_split: Optional[str] = "all"

    ############################################
    # Listener
    ############################################

    # Domain to train the listener on
    train_domain: Optional[str] = "outdoor"

    ############################################
    # Speaker
    ############################################

    # beam search size
    beam_size: Optional[int] = 5
    # max length for prediction
    max_len: Optional[int] = 30

    def merge(self, other):
        """
        Merge self with another Parmas class, overlapping attributes will raise an error
        Parameters
        ----------
        other : Params, other params class

        Returns
        -------

        """
        for k, v in dataclasses.asdict(other).items():

            if hasattr(self, k):
                raise KeyError(f"Attribute '{k}' is already present in {self}")

            self.__setattr__(k, v)

    def __str__(self):
        """
        Used for printing class attributes
        :return:
        """
        to_print = f"{self.__class__}:\n"

        attrb = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attrb = [
            a for a in attrb if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        sorted(attrb)
        for k, v in attrb:
            to_print += f"\t{k}: '{v}'\n"

        # todo: add rich colors

        return to_print

    def __init__(self):
        self.__parse_args()
        self.data_path = get_dataset_path()

    def __post_init__(self):
        """
        Get current datapath, fix file paths
        Returns
        -------

        """
        self.working_dir = get_working_dir()
        self.data_path = get_dataset_path()

        if "vectors.json" in self.vectors_file:  # from resnet
            img_dim = 2048
        elif "clip.json" in self.vectors_file:
            img_dim = 512
        else:
            raise KeyError(f"No valid image vector for file '{self.vectors_file}'")

        self.image_size = img_dim
        self.vectors_file = join(self.data_path, self.vectors_file)

        if self.data_domain == "":
            self.data_domain = self.train_domain

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        att = self.__get_attributes()

        # Create the parser to capture CLI arguments.
        parser = argparse.ArgumentParser()

        # for every attribute add an arg instance
        for k, v in att.items():
            if isinstance(v, bool):
                parser.add_argument(
                    "-" + k.lower(),
                    action="store_true",
                    default=v,
                )
            else:
                parser.add_argument(
                    "--" + k.lower(),
                    type=type(v),
                    default=v,
                )

        args, unk = parser.parse_known_args()
        for k, v in vars(args).items():
            self.__setattr__(k, v)

        if len(unk) > 0:
            console.print(
                f"Some arguments are unk.\n{unk} for class {type(self).__name__}",
                style="rgb(255,165,0)",
            )

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def check_parameters(self):
        """
        Check for correctness of parameters
        Returns
        -------

        """
        valid_vf = ["vectors.json", "clip.json"]
        assert self.vectors_file in valid_vf, (
            f"Invalid vector file '{self.vectors_file}'. " f"Should be in {valid_vf}"
        )

        valid_dom = [
            "appliances",
            "food",
            "indoor",
            "outdoor",
            "speaker",
            "vehicles",
            "all",
        ]
        assert (
                self.train_domain in valid_dom
        ), f"Invalid train domain '{self.train_domain}'./n Should be in {valid_dom}"

        valid_type_of_int = ["domain", "general", "untrained"]

        assert (
                self.type_of_int in valid_type_of_int
        ), f"Invalid simulator type '{self.type_of_int}'./n Should be in {valid_type_of_int}"

        valid_test_split = ["all", "seen", "unseen"]
        assert (
                self.test_split in valid_test_split
        ), f"Invalid model test split '{self.test_split}' not in '{valid_test_split}'"

    def reset_paths(self):
        self.vocab_file = "vocab.csv"
        self.vectors_file = os.path.basename(self.vectors_file)
        self.__post_init__()


class ListenerArguments(Params):
    """
    Arguments for listener
    """

    #########################
    #   PATHS
    #########################

    vocab_file: Optional[str] = "vocab.csv"
    utterances_file: Optional[str] = "ids_utterances.pickle"
    chains_file: Optional[str] = "text_chains.json"
    orig_ref_file: Optional[str] = "text_utterances.pickle"

    #########################
    #   Model
    #########################

    # check the check_parameters method for a list of possible values
    embed_type: Optional[str] = "scratch"
    embed_dim: Optional[int] = 768
    hidden_dim: Optional[int] = 512
    attention_dim: Optional[int] = 512
    dropout_prob: Optional[float] = 0.0

    metric: Optional[str] = "accs"
    golden_data_perc: Optional[float] = 1.0

    def __init__(self):
        super(ListenerArguments, self).__init__()
        self.check_parameters()

        self.__post_init__()

    def __post_init__(self):
        super(ListenerArguments, self).__post_init__()

        self.vocab_file = join(self.data_path, self.vocab_file)
        self.vectors_file = join(self.data_path, self.vectors_file)
        self.img2dom_file = join(self.data_path, "img2dom.json")

    def check_parameters(self):
        super(ListenerArguments, self).check_parameters()

        valis_metr = ["accs", "loss"]
        assert (
                self.metric in valis_metr
        ), f"Invalid metric '{self.metric}' not in '{valis_metr}'"

        if self.embed_type == "sratch":
            assert (
                    self.embed_dim == 768
            ), f"With scratch embeddings size must be equal to 768, got '{self.embed_dim}'"


class InterpreterArguments(Params):
    """
    Arguments for simulator
    """

    #########################
    #   PATHS
    #########################
    vocab_file: Optional[str] = "vocab.csv"
    utterances_file: Optional[str] = "ids_utterances.pickle"
    chains_file: Optional[str] = "text_chains.json"
    orig_ref_file: Optional[str] = "text_utterances.pickle"

    #########################
    #   Model
    #########################

    embed_type: Optional[str] = "scratch"
    embed_dim: Optional[int] = 768
    mask_oov_embed: Optional[str] = "unk"
    golden_data_perc: Optional[float] = 1.0

    #########################

    hidden_dim: Optional[int] = 128
    embedding_dim: Optional[int] = 128
    attention_dim: Optional[int] = 128
    dropout_prob: Optional[float] = 0.0
    int_domain: Optional[str] = ""

    # when != "", ignore the canonical wandb checkpoint and load this
    force_resume_url = ""

    #########################
    #   Other
    #########################

    metric: Optional[str] = "accs"
    s_iter: Optional[int] = 20
    adapt_lr: Optional[float] = 0.5
    log_train: Optional[bool] = False

    def __init__(self):
        super(InterpreterArguments, self).__init__()

        if self.int_domain == "":
            self.int_domain = self.train_domain

        self.check_parameters()

        self.__post_init__()

    def __post_init__(self):
        super(InterpreterArguments, self).__post_init__()

        self.vocab_file = join(self.data_path, self.vocab_file)
        self.vectors_file = join(self.data_path, self.vectors_file)
        self.img2dom_file = join(self.data_path, "img2dom.json")

    def check_parameters(self):
        super(InterpreterArguments, self).check_parameters()
        valis_metr = ["accs", "loss"]
        assert (
                self.metric in valis_metr
        ), f"Invalid metric '{self.metric}'not in '{valis_metr}'"

        valid_mask_oov_embed = ["none", "unk", "zero"]
        assert (
                self.mask_oov_embed in valid_mask_oov_embed
        ), f"Invalid mask_oov_embed '{self.mask_oov_embed}' not in '{valid_mask_oov_embed}'"

        if self.embed_type == "sratch":
            assert (
                    self.embed_dim == 768
            ), f"With scratch embeddings size must be equal to 768, got '{self.embed_dim}'"


class SpeakerArguments(Params):
    """
    Arguments for speaker
    """

    #########################
    #   Paths
    #########################
    vocab_file: Optional[str] = "vocab.csv"
    utterances_file: Optional[str] = "ids_utterances.pickle"
    chains_file: Optional[str] = "text_chains.json"
    orig_ref_file: Optional[str] = "text_utterances.pickle"

    #########################
    #   Model
    #########################
    normalize: Optional[bool] = False
    embedding_dim: Optional[int] = 1024
    hidden_dim: Optional[int] = 512
    attention_dim: Optional[int] = 512
    dropout_prob: Optional[float] = 0.0
    metric: Optional[str] = "cider"

    # if true use beam search, else nucleus sampling
    use_beam: Optional[bool] = False
    # nucleus sampling top probability
    top_p: Optional[float] = 0.9
    # nucleus sampling top k
    top_k: Optional[float] = 0.0

    def __init__(self):
        super(SpeakerArguments, self).__init__()
        self.check_parameters()

        self.__post_init__()

    def __post_init__(self):
        super(SpeakerArguments, self).__post_init__()

        self.speaker_data = join(self.data_path, "speaker")

        self.img2dom_file = join(self.data_path, "img2dom.json")
        self.vocab_file = join(self.data_path, "speaker", self.vocab_file)
