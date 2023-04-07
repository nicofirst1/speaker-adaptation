import numpy
import torch

from src.data.dataloaders.AbstractDataset import AbstractDataset, imgid2path


class EcDataset(AbstractDataset):
    def __init__(self, domain, episodes, device, **kwargs):
        self.img_id2path = imgid2path(kwargs["data_dir"])

        kwargs["data_dir"] = kwargs["data_dir"] + "/chains-domain-specific/" + domain

        super(EcDataset, self).__init__(**kwargs)

        self.domain = domain
        self.episodes = episodes
        self.device = device

        domain_images = [x['image_set'] for x in self.data.values()]
        domain_images = [x for sub_list in domain_images for x in sub_list]
        self.domain_images = list(set(domain_images))

    def __getitem__(self, item):
        # choose 6 random images
        random_set = numpy.random.choice(self.domain_images, 6)

        # randomly choose a target
        target_index = numpy.random.choice(range(6))

        # use image features for target set
        image_set = [self.image_features[x] for x in random_set]

        image_set = torch.as_tensor(image_set, dtype=torch.float32)
        target_index = torch.as_tensor(target_index, dtype=torch.int64)

        # to device
        image_set = image_set.to(self.device)
        target_index = target_index.to(self.device)

        data=dict(
            image_set=image_set,
            target_index=target_index,
            image_ids=random_set
        )

        return data

    def __len__(self):
        return self.episodes
