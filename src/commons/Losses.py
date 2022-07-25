import torch
from torch import nn

from src.commons import get_domain_accuracy


class SimLoss(torch.nn.Module):

    def __init__(self, loss_type, reduction, alpha, gamma, list_domain, all_domains):

        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.fbce_alpha=alpha
        self.fbce_gamma=gamma
        self.list_domain=list_domain
        self.all_domains=all_domains

        self.loss_type = loss_type
        self.reduction=reduction

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
            if self.reduction=="sum":
                loss=loss.sum()
            elif self.reduction=="mean":
                loss=loss.mean()

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
        bce_loss=self.bce(preds,targets, use_reduction=False)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.fbce_alpha) + targets * (2 * self.fbce_alpha - 1)
        f_loss = alpha_tensor * (1 - p_t) ** self.fbce_gamma * bce_loss

        if self.reduction=="sum":
            f_loss=f_loss.sum()
        else:
            f_loss=f_loss.mean()

        return f_loss

    def forward(self, preds, targets, list_out, domains):

        list_preds = torch.argmax(list_out, dim=1)
        list_target_accuracy = torch.eq(list_preds, targets).float()

        # estimate loss based on the type
        if self.loss_type == "ce":
            loss = self.ce(preds, list_out)
        elif self.loss_type == "kl":
            loss = self.kl(preds, list_out)
        elif self.loss_type == "bce":
            loss = self.bce(preds, list_target_accuracy)
        elif self.loss_type == "fbce":
            loss = self.focal_bce(preds, list_target_accuracy)
        else:
            raise ValueError(f"Loss type {self.loss_type = } is invalid!")

        # get list accuracy and preds
        targets = targets.squeeze()
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets)

        # with bce the simulator has a binary output,
        # so the making of the aux dict with the accuracies differs
        if self.loss_type in ["bce","fbce"]:

            # sim out is logits, need bool
            sim_preds = torch.sigmoid(preds)
            sim_preds = torch.round(sim_preds)
            sim_preds = sim_preds.bool()

            if sim_preds.ndim != 1:
                sim_preds = sim_preds.squeeze(dim=-1)

            sim_list_accuracy = torch.eq(list_target_accuracy, sim_preds)
            sim_target_accuracy = sim_list_accuracy

            neg_idx = torch.ne(list_preds, targets)
            list_neg_preds = list_target_accuracy[neg_idx]

            pos_idx = torch.eq(list_preds, targets)
            list_pos_preds = list_target_accuracy[pos_idx]

        else:
            sim_preds = torch.argmax(preds.squeeze(dim=-1), dim=1)
            sim_list_accuracy = torch.eq(list_preds, sim_preds)
            sim_target_accuracy = torch.eq(sim_preds, targets)
            neg_idx = torch.ne(list_preds, targets)
            list_neg_preds = list_preds[neg_idx]
            pos_idx = torch.eq(list_preds, targets)
            list_pos_preds = list_preds[pos_idx]

        sim_list_neg_accuracy = torch.eq(
            list_neg_preds,
            sim_preds[neg_idx],
        )
        sim_list_pos_accuracy = torch.eq(
            list_pos_preds,
            sim_preds[pos_idx],
        )

        list_target_accuracy_dom=list_target_accuracy.tolist()
        sim_list_accuracy_dom=sim_list_accuracy.tolist()
        sim_target_accuracy_dom=sim_target_accuracy.tolist()

        d = [domains[idx] for idx in range(len(domains)) if neg_idx[idx]]
        sim_list_neg_accuracy_dom = (sim_list_neg_accuracy, d)

        d = [domains[idx] for idx in range(len(domains)) if pos_idx[idx]]
        sim_list_pos_accuracy_dom = (sim_list_pos_accuracy, d)


        list_target_accuracy = list_target_accuracy.sum()
        sim_list_accuracy=sim_list_accuracy.sum()
        sim_target_accuracy=sim_list_accuracy.sum()
        sim_list_neg_accuracy=sim_list_neg_accuracy.sum()
        sim_list_pos_accuracy=sim_list_pos_accuracy.sum()

        # cast to list
        sim_list_accuracy = sim_list_accuracy.tolist()
        list_target_accuracy = list_target_accuracy.tolist()
        sim_target_accuracy = sim_target_accuracy.tolist()
        sim_list_neg_accuracy = sim_list_neg_accuracy.tolist()
        sim_list_pos_accuracy = sim_list_pos_accuracy.tolist()
        list_preds = list_preds.tolist()
        sim_preds = sim_preds.tolist()

        # build dict
        aux = dict(
            sim_list_accuracy=sim_list_accuracy,
            list_target_accuracy=list_target_accuracy,
            sim_target_accuracy=sim_target_accuracy,
            sim_list_neg_accuracy=sim_list_neg_accuracy,
            sim_list_pos_accuracy=sim_list_pos_accuracy,
            list_preds=list_preds,
            sim_preds=sim_preds,
            neg_pred_len=len(list_neg_preds),
            pos_pred_len=len(list_pos_preds),

            # domain specific
            list_target_accuracy_dom=list_target_accuracy_dom,
            sim_list_accuracy_dom=sim_list_accuracy_dom,
            sim_target_accuracy_dom=sim_target_accuracy_dom,
            sim_list_neg_accuracy_dom=sim_list_neg_accuracy_dom,
            sim_list_pos_accuracy_dom=sim_list_pos_accuracy_dom,
            domains=domains,

        )

        return loss, aux
