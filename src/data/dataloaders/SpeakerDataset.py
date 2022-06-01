from src.data.dataloaders.AbstractDataset import AbstractDataset


class SpeakerDataset(AbstractDataset):
    def __init__(self, **kwargs):
        kwargs["data_dir"] = kwargs["data_dir"] + "/chains-domain-specific/speaker"

        super(SpeakerDataset, self).__init__(**kwargs)
