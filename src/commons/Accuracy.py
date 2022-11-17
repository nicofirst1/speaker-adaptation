import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AccuracyEstimator(torch.nn.Module):
    def __init__(self, list_domain, all_domains):
        super().__init__()

        self.list_domain = list_domain

        # create an index dict for domains
        self.domain2idx = {d: idx for idx, d in enumerate(sorted(all_domains))}
        self.domain2idx["all"] = len(all_domains)
        self.idx2domain = {idx: d for idx, d in self.domain2idx.items()}
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, preds, targets, list_out, domains, is_adaptive=False):
        preds = preds.to("cpu")
        targets = targets.to("cpu")
        list_out = list_out.to("cpu")

        # get list accuracy and preds
        targets = targets.squeeze()
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets)
        neg_idx = torch.ne(list_preds, targets)
        pos_idx = torch.eq(list_preds, targets)

        # with bce the simulator has a binary output,
        # so the making of the aux dict with the accuracies differs

        sim_preds = torch.argmax(preds.squeeze(dim=-1), dim=1)

        # estimate kl divergence and kolmogorov-smirnov test
        p = preds.squeeze(dim=-1)
        l = list_out.squeeze(dim=-1)
        p = F.log_softmax(p, dim=1)
        l = F.softmax(l, dim=1)
        kl_div = self.kl_div(p, l).detach().cpu().item()

        # get pos and neg predictions
        list_neg_preds = list_preds[neg_idx]
        list_pos_preds = list_preds[pos_idx]

        # simulator is predicting the listener output
        sim_list_accuracy = torch.eq(list_preds, sim_preds)
        sim_target_accuracy = torch.eq(sim_preds, targets)

        sim_list_neg_accuracy = torch.eq(
            list_neg_preds,
            sim_preds[neg_idx],
        )
        sim_list_pos_accuracy = torch.eq(
            list_pos_preds,
            sim_preds[pos_idx],
        )

        list_target_accuracy_dom = list_target_accuracy.tolist()
        sim_list_accuracy_dom = sim_list_accuracy.tolist()
        sim_target_accuracy_dom = sim_target_accuracy.tolist()

        d = [domains[idx] for idx in range(len(domains)) if neg_idx[idx]]
        sim_list_neg_accuracy_dom = (sim_list_neg_accuracy, d)

        d = [domains[idx] for idx in range(len(domains)) if pos_idx[idx]]
        sim_list_pos_accuracy_dom = (sim_list_pos_accuracy, d)

        list_target_accuracy = list_target_accuracy.sum()
        sim_list_accuracy = sim_list_accuracy.sum()
        sim_target_accuracy = sim_target_accuracy.sum()
        sim_list_neg_accuracy = sim_list_neg_accuracy.sum()
        sim_list_pos_accuracy = sim_list_pos_accuracy.sum()

        # cast to list
        sim_list_accuracy = sim_list_accuracy.tolist()
        list_target_accuracy = list_target_accuracy.tolist()
        sim_target_accuracy = sim_target_accuracy.tolist()
        sim_list_neg_accuracy = sim_list_neg_accuracy.tolist()
        sim_list_pos_accuracy = sim_list_pos_accuracy.tolist()
        list_preds = list_preds.tolist()
        sim_preds = sim_preds.tolist()

        list_dist = F.softmax(list_out.squeeze(dim=-1).detach().cpu(), dim=1)
        pred_dist = F.softmax(preds.squeeze(dim=-1).detach().cpu(), dim=1)

        # build dict
        aux = dict(
            # accuracy
            sim_list_accuracy=sim_list_accuracy,
            list_target_accuracy=list_target_accuracy,
            sim_target_accuracy=sim_target_accuracy,
            sim_list_neg_accuracy=sim_list_neg_accuracy,
            sim_list_pos_accuracy=sim_list_pos_accuracy,

            # preds
            list_preds=list_preds,
            sim_preds=sim_preds,
            neg_pred_len=len(list_neg_preds),
            pos_pred_len=len(list_pos_preds),

            # distributions
            list_dist=list_dist,
            sim_dist=pred_dist,

            # domain specific
            list_target_accuracy_dom=list_target_accuracy_dom,
            sim_list_accuracy_dom=sim_list_accuracy_dom,
            sim_target_accuracy_dom=sim_target_accuracy_dom,
            sim_list_neg_accuracy_dom=sim_list_neg_accuracy_dom,
            sim_list_pos_accuracy_dom=sim_list_pos_accuracy_dom,
            domains=domains,
            # divergence stats
            kl_div=kl_div,

        )

        return aux
