import dataclasses
import os.path
from dataclasses import dataclass, field
from os.path import join
from typing import Optional

import torch
from transformers import HfArgumentParser


def parse_args(mode):
    if mode == "speak":
        parser = HfArgumentParser(
            (SpeakerArguments, DataTrainingArguments)
        )
    elif mode == "list":
        parser = HfArgumentParser(
            (ListenerArguments, DataTrainingArguments)
        )

    mode_p, data_p = parser.parse_args_into_dataclasses()

    mode_p.merge(data_p)

    return mode_p


class AbstractDataclass:
    def merge(self, other):
        for k, v in dataclasses.asdict(other).items():

            if hasattr(self, k):
                raise KeyError(f"Attribute '{k}' is already present in {self}")

            self.__setattr__(k, v)


def get_dataset_path():
    pwd = os.getcwd()
    pwd = pwd.split("pb_speaker_adaptation")[0]
    pwd = join(pwd, "pb_speaker_adaptation")
    return join(pwd, "dataset")


@dataclass
class ListenerArguments(AbstractDataclass):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    vocab_file: Optional[str] = field(
        default="vocab.csv",
        metadata={
            "help": "Vocabulary path"
        },
    )
    train_domain: Optional[str] = field(
        default="food",
        metadata={
            "help": "domain to train the listener on"
        }
    )

    embed_type: Optional[str] = field(
        default="scratch", )

    embed_dim: Optional[int] = field(
        default=768,
    )
    vectors_file: Optional[str] = field(
        default="vectors.json",
    )
    model_type: Optional[str] = field(
        default="scratch_rrr",
    )

    breaking: Optional[bool] = field(
        default=False,
    )

    hidden_dim: Optional[int] = field(
        default=512,
    )
    attention_dim: Optional[int] = field(
        default=512,
    )
    dropout_prob: Optional[float] = field(
        default=0.0,
    )
    metric: Optional[str] = field(
        default="accs",
    )
    reduction: Optional[str] = field(
        default="sum",
        metadata={
            "help": "reduction for crossentropy loss"
        }
    )
    log_data: Optional[bool] = field(
        default=False,
    )

    def __post_init__(self):
        self.data_path = get_dataset_path()

        assert self.vectors_file in ["vectors.json", "clip.json"], "Invalid vector file"
        assert self.metric in ["accs", "loss"], "Invalid metric"
        assert self.train_domain in ["appliances", "food", "indoor", "outdoor",
                                     "speaker", "vehicles", "all", ], "Invalid train domain"

        self.vocab_file = join(self.data_path, self.vocab_file)

        if self.embed_type == "sratch":
            assert self.embed_dim == 768, f"With scratch embeddings size must be equal to 768, got '{self.embed_dim}'"


@dataclass
class SpeakerArguments(AbstractDataclass):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    vocab_file: Optional[str] = field(
        default="vocab.csv",
        metadata={
            "help": "Vocabulary path"
        },
    )
    utterances_file: Optional[str] = field(
        default="ids_utterances.pickle",
    )

    chains_file: Optional[str] = field(
        default="text_chains.json",
    )

    orig_ref_file: Optional[str] = field(
        default="text_utterances.pickle",
    )
    vectors_file: Optional[str] = field(
        default="vectors.json",
    )
    model_type: Optional[str] = field(
        default="hist_att",
    )
    normalize: Optional[bool] = field(
        default=False,
    )
    breaking: Optional[bool] = field(
        default=False,
    )

    embedding_dim: Optional[int] = field(
        default=512,
    )
    hidden_dim: Optional[int] = field(
        default=512,
    )
    attention_dim: Optional[int] = field(
        default=512,
    )
    dropout_prob: Optional[float] = field(
        default=0.0,
    )
    metric: Optional[str] = field(
        default="cider",
    )
    reduction: Optional[str] = field(
        default="sum",
        metadata={
            "help": "reduction for crossentropy loss"
        }
    )
    beam_size: Optional[int] = field(
        default=5,
    )

    def __post_init__(self):
        self.data_path = get_dataset_path()

        self.speaker_data = join(self.data_path, "speaker")

        # self.utterances_file = join(self.speaker_data, self.utterances_file)
        # self.chains_file = join(self.speaker_data, self.chains_file)
        # self.orig_ref_file = join(self.speaker_data, self.orig_ref_file)
        self.vectors_file = join(self.data_path, self.vectors_file)
        self.vocab_file = join(self.speaker_data, self.vocab_file)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    shuffle: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Shuffle dataloader"
        },
    )
    epochs: Optional[int] = field(
        default=60,
        metadata={
            "help": "Number of epochs for training"
        },
    )
    debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Set to true for debugging"
        },
    )
    batch_size: Optional[int] = field(
        default=64,
    )
    learning_rate: Optional[float] = field(
        default=0.001,
    )
    seed: Optional[int] = field(
        default=42,
    )

    device: Optional[str] = field(
        default="cpu",
    )
    subset_size: Optional[int] = field(
        default=-1,
        metadata={
            "help": "-1 is the full dataset, if you put 10, it will only use 10 chains"
        }
    )

    wandb_dir: Optional[str] = field(
        default="wandb_out",
    )

    def __post_init__(self):
        self.device = torch.device(self.device)
