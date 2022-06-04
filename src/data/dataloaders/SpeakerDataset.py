import os

from src.data.dataloaders.AbstractDataset import AbstractDataset


class SpeakerDataset(AbstractDataset):
    def __init__(self, **kwargs):
        kwargs["data_dir"] = os.path.join(kwargs["data_dir"], "speaker")

        super(SpeakerDataset, self).__init__(**kwargs)
