from src.data.dataloaders.AbstractDataset import AbstractDataset, imgid2path


class EcDataset(AbstractDataset):
    def __init__(self, domain, **kwargs):
        self.img_id2path = imgid2path(kwargs["data_dir"])

        kwargs["data_dir"] = kwargs["data_dir"] + "/chains-domain-specific/" + domain

        super(EcDataset, self).__init__(**kwargs)

        self.domain = domain


