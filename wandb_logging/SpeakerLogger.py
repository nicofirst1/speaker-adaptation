import random
from collections import Counter
from typing import Any, Dict, List

import PIL.Image
import torch
import wandb
from PIL import ImageOps
from torch import nn

from data.dataloaders import imgid2domain, imgid2path
from wandb_logging.WandbLogger import WandbLogger


class SpeakerLogger(WandbLogger):
    def __init__(self, vocab, **kwargs):
        """
        Args:
            models: list of torch.Modules to watch with wandb
            **kwargs:
        """
        super().__init__(project="speaker", **kwargs)

        self.vocab = vocab

        # create a dict from img_id to path
        data_path = self.opts["data_path"]
        self.img_id2path = imgid2path(data_path)

        # create a dict from img_id to domain
        self.img_id2domain, self.domains = imgid2domain(data_path)

        ### datapoint table

        self.dt_table = {}

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

    def watch_model(self, models: List[nn.Module]):
        for idx, mod in enumerate(models):
            wandb.watch(mod, log_freq=1000, log_graph=True, idx=idx, log="all")

    def log_domain_accuracy(self, data_point: Dict, preds) -> Dict:
        """
        Log a datapoint into a wandb table
        :param data_point: datapoint as it comes from the dataloader
        :param preds: prediction of model, after argmax
        :return:
        """

        imgs = data_point["image_set"]
        target = data_point["target"].cpu().numpy().squeeze()
        preds = preds.detach().cpu().numpy().squeeze()

        if target.size == 1:
            # if target is unidimensional then exapnd dim
            target = [target]

        imgs_class = []
        for idx in range(len(imgs)):
            img = imgs[idx][target[idx]]
            imgs_class.append(self.img_id2domain[img])

        # estimate number of correct
        correct = preds == target

        domain_accs = {d: 0 for d in self.domains}
        domain_accs["all"] = 0

        for idx in range(len(imgs_class)):
            if correct[idx]:
                dom = imgs_class[idx]
                domain_accs[dom] += 1
                domain_accs["all"] += 1

        c = Counter(imgs_class)

        for k, v in c.items():
            domain_accs[k] /= v

        domain_accs["all"] /= len(correct)

        return domain_accs

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
        utt = data_point["orig_utterance"][idx]
        target = data_point["target"][idx].cpu().numpy()
        target_ids = data_point["target_utt_ids"]
        if len(target_ids) > 0:
            target_ids = target_ids[idx].cpu().numpy()
        hist = data_point["prev_histories"][idx]
        preds = preds[idx]

        ## convert to int
        if isinstance(preds, torch.Tensor):
            preds = torch.argmax(preds.detach().cpu(), dim=0).numpy()
        target = int(target)

        # convert to words
        translate_list = lambda lst: " ".join([self.vocab.index2word[x] for x in lst])
        # hist = translate_list([int(x) for x in hist])

        preds = translate_list(preds)
        target_ids = translate_list(target_ids)
        target_ids = target_ids.replace("<pad>", "")

        # get imgs domain
        # imgs_domains = [self.img_id2domain[img] for img in imgs]
        imgs_domains = []
        for img in imgs:
            try:
                k = self.img_id2domain[img]
            except KeyError:
                k = "Error"
                print(f"Error for img '{img}'")

            imgs_domains.append(k)

        # read image
        imgs = [self.img_id2path[x] for x in imgs]
        imgs = [PIL.Image.open(x) for x in imgs]

        ## add red border to pred if wrong
        imgs[target] = ImageOps.expand(imgs[target], border=10, fill="green")

        data = [imgs_domains[target]]
        data += [
            wandb.Image(img, caption=f"Domain: {dom}")
            for img, dom in zip(imgs, imgs_domains)
        ]
        data += [utt, target_ids, preds]

        if modality not in self.dt_table.keys():
            self.dt_table[modality] = []

        self.dt_table[modality].append(data)

        table_columns = ["img domain"]
        table_columns += [f"img_{i}" for i in range(6)]
        table_columns += ["utt", "target_ids", "preds"]

        new_table = wandb.Table(columns=table_columns, data=self.dt_table[modality])

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

    def on_eval_end(
        self,
        metrics: Dict[str, Any],
        model_params: Dict[str, Any],
        model_out: Dict[str, Any],
        data_point: Dict[str, Any],
    ):

        # get and log domain accuracy table

        logs = {f"eval/{k}": v for k, v in metrics.items()}

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
        logs["loss"] = loss.detach().item()

        data_point["target_utt_ids"] = aux["target_utt_ids"]

        # apply correct flag
        logs = {f"{modality}/{k}": v for k, v in logs.items()}

        if modality not in self.steps.keys():
            self.steps[modality] = -1

        # update steps for this modality
        self.steps[modality] += 1
        logs[f"{modality}/steps"] = self.steps[modality]

        self.log_to_wandb(logs, commit=False)
