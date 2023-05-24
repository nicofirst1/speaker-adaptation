from collections import Counter
from typing import Dict

import wandb
from src.data.dataloaders import imgid2path, load_imgid2domain
from src.data.dataloaders.ListenerDataset import ListenerDataset
from src.wandb_logging.WandbLogger import WandbLogger


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
        self.img_id2domain, self.domains = load_imgid2domain(
            kwargs["opts"]["img2dom_file"]
        )

        ### datapoint table
        table_columns = ["model domain"]
        table_columns += [f"img_{i}" for i in range(6)]
        table_columns += ["utt", "hist"]
        self.dt_table = wandb.Table(columns=table_columns)

    def log_domain_balance(self, dataset: ListenerDataset, modality: str) -> Dict:
        """
        Create  a table to log number of domain images
        :param modality:
        :return:
        """

        domains = [x["domain"] for x in dataset.data.values()]

        count = Counter(domains)
        tot = len(domains)

        columns = ["domain", "img_num", "perc"]
        data = []
        for k, v in count.items():
            data.append((k, v, v / tot))

        new_table = wandb.Table(columns=columns, data=data)
        logs = {f"domain_stats/{modality}": new_table}

        return logs

    def log_target_balance(self, dataset: ListenerDataset, modality: str) -> Dict:
        """
        Create  a table to log number of domain images
        :param modality:
        :return:
        """

        domains = [x["target"][0] for x in dataset.data.values()]

        count = Counter(domains)
        tot = len(domains)

        columns = ["target", "img_num", "perc"]
        data = []
        for k, v in count.items():
            data.append((k, v, v / tot))

        new_table = wandb.Table(columns=columns, data=data)
        logs = {f"target_stats/{modality}": new_table}

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
        for d in dataset.data.values():

            # get img_id of target
            img_id = d["image_set"][d["target"][0]]
            img_id = str(img_id)

            # skip if id not in dict
            if img_id not in self.img_id2domain.keys():
                skipped += 1
                continue

            # get domain and embed
            img_domain = self.img_id2domain[img_id]
            img_emb = dataset.image_features[img_id]

            # make image and append
            img = wandb.Image(self.img_id2path[img_id], caption=f"Domain: {img_domain}")
            data.append((img, img_domain, img_emb))

        print(
            f"Skipped {skipped} image out of {len(dataset)} ({skipped / len(dataset) * 100:.3f}%)"
        )
        # create table
        columns = ["image", "domain", "viz_embed"]
        new_table = wandb.Table(columns=columns, data=data)

        logs = {f"viz_embed/{modality}": new_table}

        return logs

    def log_dataset(self, dataset: ListenerDataset, modality: str):

        logs = {}

        logs.update(self.log_domain_balance(dataset, modality))
        logs.update(self.log_target_balance(dataset, modality))
        # logs.update(self.log_viz_embeddings(dataset, modality))

        self.log_to_wandb(logs, commit=True)
