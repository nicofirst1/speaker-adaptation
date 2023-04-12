from collections import defaultdict

import numpy
import torch

from src.data.dataloaders.AbstractDataset import AbstractDataset, imgid2path


class EcDataset(AbstractDataset):
    def __init__(self, domain, episodes, batch_size, device, **kwargs):
        self.img_id2path = imgid2path(kwargs["data_dir"])

        kwargs["data_dir"] = kwargs["data_dir"] + "/chains-domain-specific/" + domain

        super(EcDataset, self).__init__(**kwargs)

        self.domain = domain
        self.episodes = episodes
        self.device = device
        self.batch_size = batch_size

        domain_images = [x['image_set'] for x in self.data.values()]
        domain_images = [x for sub_list in domain_images for x in sub_list]
        self.domain_images = sorted(list(set(domain_images)))

        self.randomize_data()

    def randomize_data(self):

        self.data = []


        for _ in range(self.episodes*self.batch_size):
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
            target_id = random_set[target_index]
            target_img_feat = image_set[target_index]

            a=1

            data = dict(
                image_set=image_set,
                image_ids=random_set,
                target_index=target_index,
                target_id=target_id,
                target_img_feat=target_img_feat,

            )
            self.data.append(data)

    def __getitem__(self, item):

        data = self.data[item]

        return data

    def __len__(self):
        return self.episodes

    def get_collate_fn(self):
        """
        Collate function for batching
        Parameters
        ----------
        device
        SOS
        EOS
        NOHS

        Returns
        -------

        """

        def collate_fn(data):

            batch = defaultdict(list)

            for sample in data:

                for key in sample.keys():
                    batch[key].append(sample[key])

            for key in batch.keys():

                if key in ["target_index", "image_set", "target_img_feat"]:
                    batch[key] = torch.stack(batch[key]).to(self.device)

            return batch

        return collate_fn
