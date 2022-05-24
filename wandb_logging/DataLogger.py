from collections import Counter
from typing import Dict

import wandb

from data.dataloaders import imgid2path, imgid2domain
from data.dataloaders.ListenerDataset import ListenerDataset
from wandb_logging.WandbLogger import WandbLogger


class DataLogger(WandbLogger):
    def __init__(self, vocab, **kwargs):
        """
        Args:
            models: list of torch.Modules to watch with wandb
            **kwargs:
        """
        super().__init__(project="data", **kwargs)

        self.vocab = vocab

        # create a dict from img_id to path
        data_path = self.opts["data_path"]

        self.img_id2path = imgid2path(data_path)

        # create a dict from img_id to domain
        self.img_id2domain, self.domains = imgid2domain(data_path)

        ### datapoint table
        table_columns = ["model domain"]
        table_columns += [f"img_{i}" for i in range(6)]
        table_columns += ["utt", "hist"]
        self.dt_table = wandb.Table(columns=table_columns)

    def log_domain_balance(self, modality: str) -> Dict:
        """
        Create  a table to log number of domain images
        :param modality:
        :return:
        """

        count = Counter(self.img_id2domain.values())
        columns = ["domain", "img_num"]
        new_table = wandb.Table(columns=columns, data=list(count.items()))
        logs = {f"domain_stats/{modality}": new_table}

        return logs

    def log_viz_embeddings(self, dataset: ListenerDataset, modality: str) -> Dict:
        """
        Log image embeddings
        :param dataset: the listerner dataset
        :param modality:
        :return:
        """

        data = []
        skipped = 0
        for img_id, img_emb in dataset.image_features.items():
            img_id = int(img_id)
            if img_id not in self.img_id2domain.keys():
                skipped += 1
                continue
            img_domain = self.img_id2domain[img_id]
            img = wandb.Image(self.img_id2path[img_id], caption=f"Domain: {img_domain}")
            data.append((img, img_domain, img_emb))

        print(f"Skipped {skipped} images")
        # create table
        columns = ["image", "domain", "viz_embed"]
        new_table = wandb.Table(columns=columns, data=data)

        logs = {f"viz_embed/{modality}": new_table}

        return logs

    def log_dataset(self, dataset: ListenerDataset, modality: str):

        logs = {}

        logs.update(self.log_domain_balance(modality))
        logs.update(self.log_viz_embeddings(dataset, modality))

        self.log_to_wandb(logs, commit=True)
