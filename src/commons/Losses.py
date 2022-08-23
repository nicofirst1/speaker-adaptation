from typing import Tuple, Literal, Dict

import numpy as np
import torch
from torch import nn, exp

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

class MTLOptim(nn.Module):
    """
    Implements different types of multi task learning optimization techniques taken from https://arxiv.org/pdf/2004.13379.pdf

    DWA: Dynamic Weight Averaging
    DTP : Dynamic Task Prioritization
    GradNorm: Gradient Norm
    """
    def __init__(self, type:Literal["DWA","DTP","GradNorm","None"], gamma_a:float, gamma_p:float, alpha:float, temp:float):
        super(MTLOptim, self).__init__()
        self.type=type


        self.eps=1e-6
        self.is_train=True

        self.wa=1
        self.wp=1

        #DWA
        # T represents a temperature which controls the softness of task weighting, similar to .
        # A large T results in a more even distribution between different tasks.
        # If T is large enough, we have λi ≈ 1, and tasks are weighted equally
        self.temp=temp
        self.loss_p=[]
        self.loss_a=[]

        # GradNorm
        self.grads_p=[]
        self.grads_a=[]
        # α sets the strength of the restoring force which pulls tasks back to a common training rate .
        # In cases where tasks are very different in their complexity,
        # leading to dramatically different learning dynamics between tasks,
        # a higher value of α should be used to enforce stronger training rate balancing.
        # When tasks are more symmetric (e.g. the synthetic examples in Section 4), a lower value of α is appropriate.
        # Note that α = 0 will always try to pin the norms of backpropagated gradients from each task to be equal at W
        self.alpha=alpha
        self.weights = torch.nn.Parameter(torch.ones(2).float())

        # DTP
        # Further, we define a task-level focusing parameter γi ≥ 0 that allows to adjust the
        # weight at which easy or hard tasks are down-weighted, higher values result tasks being down-weighted.
        self.gamma_a=gamma_a
        self.gamma_p=gamma_p
        self.p_acc=0
        self.a_acc=0




    def update_grads(self, model,p_loss, a_loss):

        if self.type!="GradNorm" or not self.is_train:
            return

        # get layer of shared weights
        W = model.lin_mm
        task_loss=torch.stack((p_loss,a_loss))
        initial_task_loss=torch.stack((self.loss_p[0],self.loss_a[0]))

        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(self.weights[i], gygw[0])))
        norms = torch.stack(norms)
        # print('G_w(t): {}'.format(norms))

        # compute the inverse training rate r_i(t)
        # \curl{L}_i
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss

        # r_i(t)
        inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
        # print('r_i(t): {}'.format(inverse_train_rate))

        # compute the mean norm \tilde{G}_w(t)
        mean_norm = np.mean(norms.data.cpu().numpy())

        # print('tilde G_w(t): {}'.format(mean_norm))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
        # print('Constant term: {}'.format(constant_term))
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        # print('GradNorm loss {}'.format(grad_norm_loss))

        # compute the gradient for the weights
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        self.wp=self.weights[0].detach().cpu()
        self.wa=self.weights[1].detach().cpu()

    def grad_norm(self,p_loss, a_loss):

        p_loss*=self.weights[0]
        a_loss*=self.weights[1]

        return p_loss, a_loss

    def dwa(self,p_loss, a_loss)->Tuple[torch.Tensor, torch.Tensor]:

        if len(self.loss_p)<=1:
            return p_loss, a_loss

        eps=1 if self.loss_a[-2] ==0 else self.loss_a[-2]
        r_a=self.loss_a[-1]/(self.loss_a[-2]+eps)
        eps=1 if self.loss_p[-2] ==0 else self.loss_p[-2]
        r_p=self.loss_p[-1]/(self.loss_p[-2]+eps)

        self.wa=(2*exp(r_a/self.temp))/(exp(r_a/self.temp)+exp(r_p/self.temp))
        self.wp=(2*exp(r_p/self.temp))/(exp(r_a/self.temp)+exp(r_p/self.temp))

        return self.wp*p_loss,self.wa*a_loss

    def update_dtp(self, p_acc, a_acc):

        if not self.is_train:
            return

        btc_size=len(p_acc['list_preds'])


        self.p_acc=p_acc['sim_target_accuracy']/btc_size
        self.a_acc=a_acc['sim_target_accuracy']/btc_size



    def dtp(self, p_loss, a_loss):

        self.wp=-(1-self.p_acc)**self.gamma_p* np.log(self.p_acc + self.eps)
        self.wa=-(1-self.a_acc)**self.gamma_a* np.log(self.a_acc + self.eps)

        return self.wp*p_loss,self.wa*a_loss

    def forward(self, p_loss, a_loss) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.is_train:
            return p_loss, a_loss

        self.loss_a.append(a_loss.detach().cpu())
        self.loss_p.append(p_loss.detach().cpu())

        if self.type=="DWA":
            return self.dwa(p_loss, a_loss)
        elif self.type=="GradNorm":
            return self.grad_norm(p_loss, a_loss)
        elif self.type=="DTP":
            return self.dtp(p_loss, a_loss)
        elif self.type=="None":
            return p_loss, a_loss
        else:
            raise ValueError("Unknown optimization type")


    def get_infos(self)->Dict:
        res=dict(
            pretrain=self.wp.item(),
            adaptive=self.wa.item(),
        )
        res={f"weights/{k}":v for k,v in res.items()}
        return res

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
