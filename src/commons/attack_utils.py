from collections import Counter

import torch
import numpy as np
import wandb
from torch import nn, cosine_similarity


class PGD(nn.Module):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model, eps=0.4, steps=10):
        super(PGD, self).__init__()
        self.eps = eps
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.model = model
        self.targeted = False
        self.loss = nn.CrossEntropyLoss()
        self.most_similar = Counter()
        self.same_sim_len = Counter()
        self.acc_change=Counter()

    def fgsm_attack(self, inputs, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_inp = inputs + self.eps * sign_data_grad
        # Return the perturbed image
        return perturbed_inp

    def forward(self, utterance, inputs, targets):
        r"""
        Overridden.
        """

        orig_utterance = utterance.clone().detach()
        utterance = utterance.clone().detach()
        targets = targets.clone().detach()

        prev_loss = 99999
        prev_correct = 0
        orig_correct = 0
        itx = 1
        epsiolon = self.eps
        most_similar=[]
        same_sim_len=[]
        for st in range(self.steps):
            # Zero all existing gradients
            self.model.zero_grad()

            out = self.model(utterance, *inputs)
            out = out.squeeze(dim=-1)

            preds = torch.argmax(out, dim=-1)
            correct = torch.eq(preds, targets).sum() / len(targets)

            if st == 0:
                orig_correct = correct

            if correct > prev_correct or correct == 1:
                break

            loss = self.loss(out, targets)
            loss.backward(retain_graph=True)

            # Collect datagrad
            data_grad = self.model.embeddings.weight.grad[utterance]
            in_data = self.model.embeddings(utterance)

            # Call FGSM Attack
            ratio = prev_loss / loss
            if ratio <= 1.2:
                epsiolon += itx / 10
                itx += 1
            else:
                itx = 1

            perturbed_emb = self.fgsm_attack(in_data, data_grad)
            perturbed_batch = []

            # get perturbed utterance
            for btc in range(perturbed_emb.shape[0]):
                perturbed_utts = []

                for idx in range(perturbed_emb.shape[1]):
                    embed = perturbed_emb[btc, idx].squeeze()
                    sim = cosine_similarity(embed, self.model.embeddings.weight, dim=1)
                    sorted_sim, sorted_idx=torch.sort(sim,dim=-1, descending=True)
                    max_sim=(sorted_sim==sorted_sim[0]).sum()
                    same_sim_len.append(max_sim.item())
                    sorted_idx=sorted_idx[:max_sim]
                    most_similar+=sorted_idx.tolist()
                    idx3 = torch.randint(0, len(sorted_idx), (1,)).item()
                    idx3 = sorted_idx[idx3]
                    perturbed_utts.append(idx3)
                perturbed_batch.append(perturbed_utts)

            utterance = torch.as_tensor(perturbed_batch)
            utterance = utterance.to(self.model.device)

            prev_loss = loss
            prev_correct = correct

        most_similar = Counter(most_similar)
        same_sim_len = Counter(same_sim_len)
        # update self.most_similar
        self.most_similar.update(most_similar)
        self.same_sim_len.update(same_sim_len)

        new_acc=correct-orig_correct
        self.acc_change.update([new_acc.item()])

        return utterance

    def get_stats(self):
        """convert counters

        """



        res=dict()
        most_similar = dict(self.most_similar)
        same_sim_len = dict(self.same_sim_len)
        acc_change = dict(self.acc_change)

        # get weighted average
        weighted_sim = np.array(list(most_similar.values())) * np.array(list(most_similar.keys()))
        weighted_sim = np.sum(weighted_sim) / np.sum(list(most_similar.values()))
        weighted_sim=weighted_sim/len(self.model.embeddings.weight)

        weighted_sim_len = np.array(list(same_sim_len.values())) * np.array(list(same_sim_len.keys()))
        weighted_sim_len = np.sum(weighted_sim_len) / np.sum(list(same_sim_len.values()))

        weighted_acc_change = np.array(list(acc_change.values())) * np.array(list(acc_change.keys()))
        weighted_acc_change = np.sum(weighted_acc_change) / np.sum(list(acc_change.values()))

        res['weighted_sim'] = weighted_sim
        res['weighted_sim_len'] = weighted_sim_len
        res['weighted_acc_change'] = weighted_acc_change


        # make numpy NumpyHistogram
        hist_sim = np.histogram(list(most_similar.keys()), weights=list(most_similar.values()), bins=100)
        hist_sim_len = np.histogram(list(same_sim_len.keys()), weights=list(same_sim_len.values()))
        hist_acc_change = np.histogram(list(acc_change.keys()), weights=list(acc_change.values()))

        # transform to wandb
        hist_sim = wandb.Histogram(np_histogram=hist_sim)
        hist_sim_len = wandb.Histogram(np_histogram=hist_sim_len)
        hist_acc_change = wandb.Histogram(np_histogram=hist_acc_change)

        # reset counters
        self.most_similar = Counter()
        self.same_sim_len = Counter()
        self.acc_change = Counter()


        res['hist_sim'] = hist_sim
        res['hist_sim_len'] = hist_sim_len
        res['hist_acc_change'] = hist_acc_change

        return res




