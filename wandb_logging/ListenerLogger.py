import json
import os
import random
from collections import Counter
from os import listdir
from typing import Dict, Any, List

import PIL.Image
import torch
import wandb
from PIL import ImageOps
from torch import nn

from wandb_logging.WandbLogger import WandbLogger


class ListenerLogger(WandbLogger):

    def __init__(self, vocab, **kwargs):
        """
        Args:
            models: list of torch.Modules to watch with wandb
            **kwargs:
        """
        super().__init__(**kwargs)

        self.vocab = vocab

        # create a dict from img_id to path
        data_path = self.opts['data_path']
        image_path = os.path.join(data_path, 'photobook_coco_images')

        images = [f for f in listdir(image_path) if "jpg" in f]
        imgs_ids = [int(x.rsplit("_", 1)[1].split(".")[0]) for x in images]

        images = [os.path.join(image_path, x) for x in images]

        self.img_id2path = dict(zip(imgs_ids, images))

        # create a dict from img_id to domain

        chains_path = os.path.join(data_path, 'chains-domain-specific', 'speaker')
        chain_dict = {}
        for split in ['train', 'test', 'val']:
            with open(os.path.join(chains_path, f"{split}.json"), 'r') as file:
                chain_dict.update(json.load(file))

        chain_dict = {k.split("/")[1]: k.split("/")[0] for k in chain_dict.keys()}
        chain_dict = {int(k.split("_")[-1].split(".")[0]): v for k, v in chain_dict.items()}

        domain_dict = {'person_motorcycle': 'vehicles', 'car_motorcycle': 'vehicles', 'bus_truck': 'vehicles',
                       'car_truck': 'vehicles', 'person_suitcase': 'outdoor', 'person_umbrella': 'outdoor',
                       'person_surfboard': 'outdoor', 'person_elephant': 'outdoor', 'person_bicycle': 'outdoor',
                       'person_car': 'outdoor', 'person_train': 'outdoor', 'person_bench': 'outdoor',
                       'person_truck': 'outdoor',
                       'bowl_dining_table': 'food', 'cup_dining_table': 'food', 'cake_dining_table': 'food',
                       'person_oven': 'appliances', 'dining_table_refrigerator': 'appliances',
                       'person_refrigerator': 'appliances',
                       'dining_table_laptop': 'indoor', 'couch_laptop': 'indoor', 'person_bed': 'indoor',
                       'person_couch': 'indoor',
                       'person_tv': 'indoor', 'couch_dining_table': 'indoor', 'person_teddy_bear': 'indoor',
                       'chair_couch': 'indoor'}

        chain_dict = {k: domain_dict[v] for k, v in chain_dict.items()}
        self.img_id2domain = chain_dict
        self.domains = list(set(domain_dict.values()))

        ### datapoint table

        table_columns = [f'img_{i}' for i in range(6)]
        table_columns += ['utt', 'hist']
        self.dt_table = wandb.Table(columns=table_columns)

        ### domain table

        columns = ['model domain', 'all', 'appliances', 'food', 'indoor', 'outdoor', 'vehicles']
        self.domain_table = wandb.Table(columns)

    def watch_model(self, models: List[nn.Module]):
        for idx, mod in enumerate(models):
            wandb.watch(mod, log_freq=1000, log_graph=True, idx=idx, log="all")

    def on_batch_end(self, loss: torch.Tensor, data_point: Dict[str, Any],
                     aux: Dict[str, Any], batch_id: int,
                     modality: str):

        logging_step = self.train_logging_step if modality == "train" else self.val_logging_step

        # do not log
        if batch_id > 0 and logging_step % batch_id != 0:
            return

        logs = {}
        logs.update(aux)
        logs['loss'] = loss.detach().item()
        logs.update(self.log_datapoint(data_point, aux['preds']))

        if "out_domain" in modality:
            self.log_domain_accuracy(data_point, aux['preds'])

        # apply correct flag
        logs = {f"{modality}/{k}": v for k, v in logs.items()}

        if modality not in self.steps.keys():
            self.steps[modality] = -1

        # update steps for this modality
        self.steps[modality] += 1
        logs[f'{modality}/steps'] = self.steps[modality]

        self.log_to_wandb(logs, commit=False)

    def log_domain_accuracy(self, data_point: Dict, preds) -> Dict:
        """
        Log a datapoint into a wandb table
        :param data_point: datapoint as it comes from the dataloader
        :param preds: prediction of model, after argmax
        :return:
        """

        imgs = data_point['image_set']
        target = data_point['target'].cpu().numpy().squeeze()
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
        domain_accs['all'] = 0

        for idx in range(len(imgs_class)):
            if correct[idx]:
                dom = imgs_class[idx]
                domain_accs[dom] += 1
                domain_accs["all"] += 1

        c = Counter(imgs_class)

        for k, v in c.items():
            domain_accs[k] /= v

        domain_accs['all'] /= len(correct)

        return domain_accs

    def log_datapoint(self, data_point: Dict, preds) -> Dict:
        """
        Log a datapoint into a wandb table
        :param data_point: datapoint as it comes from the dataloader
        :param preds: prediction of model, after argmax
        :return:
        """
        # get random idx for logging
        batch_size = len(data_point['image_set'])
        idx = random.randint(0, batch_size - 1)

        imgs = data_point['image_set'][idx]
        utt = data_point['utterance'][idx].cpu().numpy()
        target = data_point['target'][idx].cpu().numpy()
        hist = data_point['prev_histories'][idx]
        preds = preds[idx].detach().cpu().numpy()

        ## convert to int
        preds = int(preds)
        target = int(target)

        # remove empty hists
        hist = [x for x in hist if len(x)]

        # convert to words
        translate_list = lambda utt: " ".join([self.vocab.index2word[x] for x in utt])
        hist = [translate_list(x) for x in hist]
        utt = translate_list(utt)
        utt = utt.replace(" <pad>", "")

        # read image
        imgs = [self.img_id2path[x] for x in imgs]
        imgs = [PIL.Image.open(x) for x in imgs]

        ## add red border to pred if wrong

        if preds != target:
            imgs[preds] = ImageOps.expand(imgs[preds], border=10, fill='red')

        imgs[target] = ImageOps.expand(imgs[target], border=10, fill='green')

        imgs = [wandb.Image(x) for x in imgs]
        imgs += [utt, hist]
        self.dt_table.add_data(*imgs)

        logs = dict(
            data_table=self.dt_table
        )

        return logs

    def on_train_end(self, metrics: Dict[str, Any], epoch_id: int):
        metrics['epochs'] = epoch_id
        self.log_to_wandb(metrics, commit=True)

    def on_eval_end(self, metrics: Dict[str, Any], list_domain: int, modality: str):

        # get and log domain accuracy table
        logs = {}
        if "out_domain" in modality:
            domain_accuracy = metrics['domain_accuracy']
            domain_accuracy = sorted(domain_accuracy.items(), key=lambda item: item[0])

            data = [self.opts['train_domain']]
            data += [x[1] for x in domain_accuracy]

            #self.domain_table.add_data(*data)
            new_table = wandb.Table(
                columns=self.domain_table.columns, data=[data]
            )
            logs["domain_acc"] = new_table

        logs['mrr'] = metrics['mrr']
        logs['loss'] = metrics['loss']

        logs = {f"{modality}/{k}": v for k, v in logs.items()}

        self.log_to_wandb(logs, commit=False)
