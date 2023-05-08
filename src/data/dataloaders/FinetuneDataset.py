import json
from collections import defaultdict

import numpy
import torch
from torch.utils.data import Dataset


class FinetuneDataset(Dataset):
    """
    Dataset for the fine-tuning

    @param domain: domain to be used for the fine-tuning
    @param num_images: number of images to be chosen
    @param device: device to be used
    @param vectors_file: file containing image features
    @param img2dom_file: file containing image to domain mapping
    """

    def __init__(
        self,
        domain: str,
        num_images: int,
        device,
        vectors_file: str,
        img2dom_file: str,
    ):
        with open(vectors_file, "r") as file:
            self.image_features = json.load(file)

            # Original reference sentences without unks
        with open(img2dom_file, "r") as file:
            self.img2dom = json.load(file)



        if domain != "all":
            # filter out images not in the domain
            domain_images = []
            for img_id, dom in self.img2dom.items():
                if dom == domain:
                    domain_images.append(img_id)
            self.image_features = {
                k: v for k, v in self.image_features.items() if k in domain_images
            }

        not_present_ids = set(self.img2dom.keys()) - set(self.image_features.keys())
        print("not present ids", len(not_present_ids))
        for id in not_present_ids:
            self.img2dom.pop(id)

        self.domain = domain
        self.num_images = num_images
        self.device = device

        domains = set(self.img2dom.values())
        domain_images = {dom: [] for dom in domains}
        for img_id, dom in self.img2dom.items():
            domain_images[dom].append(img_id)

        self.domain_images = domain_images

        self.randomize_data()

    def randomize_data(self):
        self.data = []

        samples_for_domain = self.num_images // len(self.domain_images)

        for dom in self.domain_images.keys():
            domain_images = self.domain_images[dom]

            for _ in range(samples_for_domain):

                # choose 6 random images
                random_set = numpy.random.choice(domain_images, 6)

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

                data = dict(
                    image_set=image_set,
                    image_ids=random_set,
                    target_index=target_index,
                    target_id=target_id,
                    target_img_feat=target_img_feat,
                    domain=dom,
                )
                self.data.append(data)

        # shuffle data
        numpy.random.shuffle(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        return data

    def __len__(self):
        return len(self.data)

    def get_collate_fn(self):
        """
        Collate function for batching
        Parameters
        ----------

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
