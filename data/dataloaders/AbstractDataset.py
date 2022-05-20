import json
import os
import pickle
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(
            self,
            split,
            data_dir,
            chain_file,
            utterances_file,
            vectors_file,
            subset_size,
            image_size,

    ):

        self.data_dir = data_dir
        self.split = split

        # Load a PhotoBook utterance chain dataset
        with open(os.path.join(self.data_dir, chain_file), "r") as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue utterance dataset
        with open(os.path.join(self.data_dir, utterances_file), "rb") as file:
            self.utterances = pickle.load(file)

        # Load pre-defined image features
        with open(vectors_file, "r") as file:
            self.image_features = json.load(file)

        self.img_dim = image_size
        self.img_count = 6  # images in the context

        self.data = dict()

        self.img2chain = defaultdict(dict)

        for chain in self.chains:
            self.img2chain[chain["target"]][chain["game_id"]] = chain["utterances"]

        if subset_size == -1:
            self.subset_size = len(self.chains)

        else:
            self.subset_size = subset_size
            np.random.shuffle(self.chains)

        print("processing", self.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
