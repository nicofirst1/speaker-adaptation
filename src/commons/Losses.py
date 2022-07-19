import torch
from torch import nn


class SimLoss(torch.nn.Module):

    def __init__(self, loss_type, reduction):

        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.fbce_alpha=0.4
        self.fbce_gamma=1

        self.loss_type = loss_type
        self.reduction=reduction

    def ce(self, preds, targets):
        return self.ce_loss(preds, targets)

    def kl(self, preds, targets):
        preds = torch.log_softmax(preds, dim=1)
        target = torch.log_softmax(targets, dim=1)
        loss = self.kl_loss(preds, target)

        return loss

    def bce(self, preds, targets, list_out, use_reduction=True):
        list_preds = torch.argmax(list_out, dim=1)
        list_target_accuracy = torch.eq(list_preds, targets).float()
        loss = self.bce_loss(preds, list_target_accuracy)

        if use_reduction:
            if self.reduction=="sum":
                loss=loss.sum()
            elif self.reduction=="mean":
                loss=loss.mean()

        return loss

    def focal_bce(self, preds, targets, list_out):
        bce_loss=self.bce(preds,targets,list_out, use_reduction=False)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.fbce_alpha) + targets * (2 * self.fbce_alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.fbce_gamma * bce_loss

        if self.reduction=="sum":
            f_loss=f_loss.sum()
        else:
            f_loss=f_loss.mean()

        return f_loss

    def forward(self, preds, targets, list_out):

        # estimate loss based on the type
        if self.loss_type == "ce":
            loss = self.ce(preds, list_out)
        elif self.loss_type == "kl":
            loss = self.kl(preds, list_out)
        elif self.loss_type == "bce":
            loss = self.bce(preds, targets, list_out)
        elif self.loss_type == "fbce":
            loss = self.focal_bce(preds, targets, list_out)
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

            sim_list_accuracy = torch.eq(list_target_accuracy, sim_preds).sum()
            sim_target_accuracy = sim_list_accuracy

            neg_idx = torch.ne(list_preds, targets)
            list_neg_preds = list_target_accuracy[neg_idx]
            sim_list_neg_accuracy = torch.eq(
                list_neg_preds,
                sim_preds[neg_idx],
            ).sum()

        else:
            sim_preds = torch.argmax(preds.squeeze(dim=-1), dim=1)
            sim_list_accuracy = torch.eq(list_preds, sim_preds).sum()
            sim_target_accuracy = torch.eq(sim_preds, targets).sum()
            list_neg_preds = list_preds[torch.ne(list_preds, targets)]
            sim_list_neg_accuracy = torch.eq(
                list_neg_preds,
                sim_preds[torch.ne(list_preds, targets)],
            ).sum()

        list_target_accuracy = list_target_accuracy.sum()

        # cast to list
        sim_list_accuracy = sim_list_accuracy.tolist()
        list_target_accuracy = list_target_accuracy.tolist()
        sim_target_accuracy = sim_target_accuracy.tolist()
        sim_list_neg_accuracy = sim_list_neg_accuracy.tolist()
        list_preds = list_preds.tolist()
        sim_preds = sim_preds.tolist()

        # build dict
        aux = dict(
            sim_list_accuracy=sim_list_accuracy,
            list_target_accuracy=list_target_accuracy,
            sim_target_accuracy=sim_target_accuracy,
            sim_list_neg_accuracy=sim_list_neg_accuracy,
            list_preds=list_preds,
            sim_preds=sim_preds,
            neg_pred_len=len(list_neg_preds),

        )

        return loss, aux