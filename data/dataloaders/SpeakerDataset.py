from data.dataloaders.AbstractDataset import AbstractDataset


class SpeakerDataset(AbstractDataset):
    def __init__(
            self,
            **kwargs

    ):
        super(SpeakerDataset, self).__init__(**kwargs)
