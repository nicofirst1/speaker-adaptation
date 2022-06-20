from src.data.dataloaders.AbstractDataset import AbstractDataset, imgid2path


class ListenerDataset(AbstractDataset):
    def __init__(self, domain, **kwargs):


        self.img_id2path = imgid2path(kwargs["data_dir"] )

        kwargs["data_dir"] = kwargs["data_dir"] + "/chains-domain-specific/" + domain

        super(ListenerDataset, self).__init__(**kwargs)

        self.domain = domain



class SpeakerUttDataset(ListenerDataset):

    def __init__(self, data, domain):
        self.data=data
        self.domain=domain