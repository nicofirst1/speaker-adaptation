from data.dataloaders.AbstractDataset import AbstractDataset


class ListenerDataset(AbstractDataset):
    def __init__(self, domain, **kwargs):
        kwargs['data_dir'] = kwargs['data_dir'] + '/chains-domain-specific/' + domain

        super(ListenerDataset, self).__init__(**kwargs)

        self.domain = domain

