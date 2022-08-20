from typing import Tuple

import torch
from torch import nn

from src.commons import get_domain_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossWeighted(nn.Module):
    def __init__(self):
        super(LossWeighted, self).__init__()
        self.wp=1
        self.wa=1
        self.min_sim_list_acc=0.75
        self.min_list_target_acc=0.50

    def forward(self, pretrain, adaptive)->Tuple[float,float]:
        batch_size=len(adaptive['list_preds'])

        sim_list_acc=pretrain['sim_list_accuracy']/batch_size
        list_target_acc=adaptive['list_target_accuracy']/batch_size

        p_loss=pretrain['loss']
        adapt_loss=adaptive['loss']
        loss_mag_rateo=p_loss/(adapt_loss+ 1e-6)

        if sim_list_acc<self.min_sim_list_acc and loss_mag_rateo<1:
            self.wa=loss_mag_rateo *0.5
            self.wp=1

        elif sim_list_acc>self.min_sim_list_acc and loss_mag_rateo>1:
            self.wp=1/loss_mag_rateo *0.5
            self.wa=1






    def get_losses(self):
        return self.wp,self.wa

class SimLossPretrain(torch.nn.Module):

    def __init__(self, loss_type, reduction, sim_model_type, alpha, gamma, list_domain, all_domains):

        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.fbce_alpha = alpha
        self.fbce_gamma = gamma
        self.list_domain = list_domain
        self.all_domains = all_domains
        self.sim_model_type = sim_model_type
        # create an index dict for domains
        self.domain2idx = {d: idx for idx, d in enumerate(sorted(all_domains))}
        self.idx2domain = {idx: d for idx, d in self.domain2idx.items()}

        self.loss_type = loss_type
        self.reduction = reduction


    def ce(self, preds, targets):
        return self.ce_loss(preds, targets)

    def kl(self, preds, targets):
        preds = torch.log_softmax(preds, dim=1)
        target = torch.log_softmax(targets, dim=1)
        loss = self.kl_loss(preds, target)

        return loss

    def bce(self, preds, targets, use_reduction=True):
        loss = self.bce_loss(preds, targets)

        if use_reduction:
            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()

        return loss

    def focal_bce(self, preds, targets):
        """Binary focal loss.

            Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
            improvements for alpha.
            :param bce_loss: Binary Cross Entropy loss, a torch tensor.
            :param targets: a torch tensor containing the ground truth, 0s and 1s.
            :param gamma: focal loss power parameter, a float scalar.
            :param alpha: weight of the class indicated by 1, a float scalar.
            """
        bce_loss = self.bce(preds, targets, use_reduction=False)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.fbce_alpha) + targets * (2 * self.fbce_alpha - 1)
        f_loss = alpha_tensor * (1 - p_t) ** self.fbce_gamma * bce_loss

        if self.reduction == "sum":
            f_loss = f_loss.sum()
        else:
            f_loss = f_loss.mean()

        return f_loss

    def forward(self, preds, targets, list_out, domains):

        if self.sim_model_type=="multi":
            preds=preds[0]

        list_preds = torch.argmax(list_out, dim=1)
        list_target_accuracy = torch.eq(list_preds, targets).float()

        # estimate loss based on the type
        if self.loss_type == "ce":
            if self.sim_model_type == "domain":
                doms = [self.domain2idx[d] for d in domains]
                doms = torch.as_tensor(doms).to(device)
                loss = self.ce(preds, doms)
            else:
                loss = self.ce(preds, list_out)
        elif self.loss_type == "kl":
            loss = self.kl(preds, list_out)
        elif self.loss_type == "bce":
            loss = self.bce(preds, list_target_accuracy)
        elif self.loss_type == "fbce":
            loss = self.focal_bce(preds, list_target_accuracy)
        else:
            raise ValueError(f"Loss type {self.loss_type = } is invalid!")


        return loss


class SimLossAdapt(SimLossPretrain):

    def init(self, loss_type, reduction, sim_model_type, alpha, gamma, list_domain, all_domains):
        super(SimLossAdapt, self).__init__(loss_type, reduction, sim_model_type, alpha, gamma, list_domain, all_domains)

    def forward(self, preds, targets, list_out, domains):

        if self.sim_model_type == "multi":
            preds = preds[1]

        list_preds = torch.argmax(list_out, dim=1)
        list_target_accuracy = torch.eq(list_preds, targets).float()

        # when model is binary use as target a vector of ones
        correct_accuracy = torch.ones(list_target_accuracy.size()).to(list_target_accuracy.device)

        # estimate loss based on the type
        if self.loss_type == "ce":
            if self.sim_model_type == "domain":
                # create a vector of listener domain the size of the input
                doms = [self.domain2idx[self.list_domain] for _ in domains]
                doms = torch.as_tensor(doms).to(device)
                loss = self.ce(preds, doms)
            else:
                loss = self.ce(preds, targets)

        elif self.loss_type == "bce":
            loss = self.bce(preds, correct_accuracy)
        elif self.loss_type == "fbce":
            loss = self.focal_bce(preds, correct_accuracy)
        else:
            raise ValueError(f"Loss type {self.loss_type = } is invalid!")


        return loss
