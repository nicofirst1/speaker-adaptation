import random
from typing import Any, Dict

import PIL.Image
import torch
import wandb
from PIL import ImageOps

from src.data.dataloaders import imgid2path, load_imgid2domain
from src.wandb_logging.WandbLogger import WandbLogger


class ListenerLogger(WandbLogger):
    def __init__(self, vocab, project=None, **kwargs):
        """
        Args:
            models: list of torch.Modules to watch with wandb
            **kwargs:
        """

        if project is None:
            project = "listener"

        super().__init__(project=project, **kwargs)

        self.vocab = vocab

        # create a dict from img_id to path
        data_path = self.opts["data_path"]
        self.img_id2path = imgid2path(data_path)

        # create a dict from img_id to domain
        self.img_id2domain, self.domains = load_imgid2domain(kwargs['opts']['img2dom_file'])

        ### datapoint table
        table_columns = ["model domain"]
        table_columns += [f"img_{i}" for i in range(6)]
        table_columns += ["utt", "hist"]
        self.dt_table = wandb.Table(columns=table_columns)

        ### domain table

        columns = [
            "model domain",
            "all",
            "appliances",
            "food",
            "indoor",
            "outdoor",
            "vehicles",
        ]
        self.domain_table = wandb.Table(columns)

        # viz embedding data
        self.embedding_data = {}

    def log_datapoint(self, data_point: Dict, preds, modality: str) -> Dict:
        """
        Log a datapoint into a wandb table
        :param data_point: datapoint as it comes from the dataloader
        :param preds: prediction of model, after argmax
        :return:
        """
        # get random idx for logging
        batch_size = len(data_point["image_set"])
        idx = random.randint(0, batch_size - 1)

        imgs = data_point["image_set"][idx]
        utt = data_point["utterance"][idx].cpu().numpy()
        target = data_point["target"][idx].cpu().numpy()
        hist = data_point["prev_histories"][idx]
        preds = preds[idx].detach().cpu().numpy()

        ## convert to int
        preds = int(preds)
        target = int(target)

        # remove empty hists
        hist = [x for x in hist.values() if len(x)]

        # convert to words
        translate_list = lambda utt: " ".join([self.vocab.index2word[x] for x in utt])
        hist = [translate_list(x) for x in hist]
        utt = translate_list(utt)
        utt = utt.replace(" <pad>", "")

        # get imgs domain
        imgs_domains = [self.img_id2domain[str(img)] for img in imgs]

        # read image
        imgs = [self.img_id2path[x] for x in imgs]
        imgs = [PIL.Image.open(x) for x in imgs]

        ## add red border to pred if wrong

        if preds != target:
            imgs[preds] = ImageOps.expand(imgs[preds], border=10, fill="red")

        imgs[target] = ImageOps.expand(imgs[target], border=10, fill="green")

        data = [self.opts["train_domain"]]
        data += [
            wandb.Image(img, caption=f"Domain: {dom}")
            for img, dom in zip(imgs, imgs_domains)
        ]
        data += [utt, hist]
        new_table = wandb.Table(columns=self.dt_table.columns, data=[data])

        logs = dict(data_table=new_table)

        logs = {f"{k}/{modality}": v for k, v in logs.items()}

        self.log_to_wandb(logs)

        return logs

    def log_viz_embeddings(self, data_point, modality):
        """
        Log image embeddings
        :param data_point:
        :param modality:
        :return:
        """
        # get random idx for logging
        batch_size = len(data_point["image_set"])
        idx = random.randint(0, batch_size - 1)

        imgs = data_point["image_set"][idx]
        img_emb = data_point["separate_images"][idx].cpu().numpy()
        img_emb = [list(x) for x in img_emb]

        # get imgs domain
        imgs_domains = [self.img_id2domain[img] for img in imgs]

        # read images
        imgs = [self.img_id2path[x] for x in imgs]
        imgs = [
            wandb.Image(img, caption=f"Domain: {dom}")
            for img, dom in zip(imgs, imgs_domains)
        ]

        # transform to matrix
        data = list(zip(imgs, imgs_domains, img_emb))

        if modality not in self.embedding_data.keys():
            self.embedding_data[modality] = []

        self.embedding_data[modality] += data

        # create table
        columns = ["image", "domain", "viz_embed"]
        new_table = wandb.Table(columns=columns, data=self.embedding_data[modality])

        logs = {f"viz_embed/{modality}": new_table}

        self.log_to_wandb(logs, commit=False)

    def on_train_end(self, metrics: Dict[str, Any], epoch_id: int):
        metrics["epochs"] = epoch_id
        self.epochs = epoch_id

        self.log_to_wandb(metrics, commit=True)

    def on_eval_end(self, metrics: Dict[str, Any], list_domain: int, modality: str):

        # get and log domain accuracy table
        logs = {}
        if "out_domain" in modality:
            domain_accuracy = metrics["domain_accuracy"]
            domain_accuracy = sorted(domain_accuracy.items(), key=lambda item: item[0])

            data = [self.opts["train_domain"]]
            data += [x[1] for x in domain_accuracy]

            # self.domain_table.add_data(*data)
            new_table = wandb.Table(columns=self.domain_table.columns, data=[data])
            logs["domain_acc_table"] = new_table

            # log plot for each domain
            logs["domain_acc_plots"] = dict(domain_accuracy)

        logs.update(metrics)
        logs = {f"{modality}/{k}": v for k, v in logs.items()}

        self.log_to_wandb(logs, commit=False)

    def on_batch_end(
            self,
            loss: torch.Tensor,
            data_point: Dict[str, Any],
            aux: Dict[str, Any],
            batch_id: int,
            modality: str,
    ):

        logging_step = (
            self.train_logging_step if modality == "train" else self.val_logging_step
        )

        # do not log
        if batch_id > 0 and logging_step % batch_id != 0:
            return

        logs = {}
        logs.update(aux)

        for k, v in logs.items():
            if isinstance(k, torch.Tensor):
                logs[k] = v.detach().item()

        # apply correct flag
        logs = {f"{modality}/{k}": v for k, v in logs.items()}

        if modality not in self.steps.keys():
            self.steps[modality] = -1

        # update steps for this modality
        self.steps[modality] += 1
        logs[f"{modality}/steps"] = self.steps[modality]

        self.log_to_wandb(logs, commit=False)
