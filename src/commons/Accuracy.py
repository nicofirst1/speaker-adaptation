import torch
from scipy.stats import ks_2samp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AccuracyEstimator(torch.nn.Module):
    def __init__(self, list_domain, int_model_type, all_domains):

        super().__init__()

        self.list_domain = list_domain
        self.int_model_type = int_model_type

        # create an index dict for domains
        self.domain2idx = {d: idx for idx, d in enumerate(sorted(all_domains))}
        self.domain2idx["all"] = len(all_domains)
        self.idx2domain = {idx: d for idx, d in self.domain2idx.items()}
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")


    def forward(self, preds, targets, list_out, domains, is_adaptive=False):

        if self.int_model_type == "multi":
            if is_adaptive:
                preds = preds[1]
            else:
                preds = preds[0]

        preds = preds.to("cpu")
        targets = targets.to("cpu")
        list_out = list_out.to("cpu")

        if is_adaptive:
            doms = [self.domain2idx[self.list_domain] for _ in domains]

        else:
            doms = [self.domain2idx[d] for d in domains]
        doms = torch.as_tensor(doms).to("cpu")

        # get list accuracy and preds
        targets = targets.squeeze()
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets)
        neg_idx = torch.ne(list_preds, targets)
        pos_idx = torch.eq(list_preds, targets)

        # with bce the interpreter has a binary output,
        # so the making of the aux dict with the accuracies differs
        if self.int_model_type == "binary":

            # int out is logits, need bool
            int_preds = torch.sigmoid(preds)
            int_preds = torch.round(int_preds)
            int_preds = int_preds.bool()

            if int_preds.ndim != 1:
                int_preds = int_preds.squeeze(dim=-1)

            int_list_accuracy = torch.eq(list_target_accuracy, int_preds)

            if is_adaptive:
                # use vector of ones if adaptive
                int_target_accuracy = torch.eq(
                    int_preds, torch.ones(list_target_accuracy.shape).bool()
                )
            else:
                int_target_accuracy = int_list_accuracy

            list_neg_preds = list_target_accuracy[neg_idx]
            list_pos_preds = list_target_accuracy[pos_idx]
            kl_div=0


        else:
            int_preds = torch.argmax(preds.squeeze(dim=-1), dim=1)

            # estimate kl divergence and kolmogorov-smirnov test
            kl_div=self.kl_div(preds.squeeze(dim=-1),list_out.squeeze(dim=-1)).detach().cpu().item()

            # get pos and neg predictions
            list_neg_preds = list_preds[neg_idx]
            list_pos_preds = list_preds[pos_idx]

            # interpreter is predicting the domain of the target
            if self.int_model_type == "domain":
                int_target_accuracy = torch.eq(int_preds, doms)
                # there is no int_list accuracy when predicting domain, set to -1
                int_list_accuracy = torch.zeros(int_target_accuracy.shape) - 1

            else:
                # int is predicting the listener output
                int_list_accuracy = torch.eq(list_preds, int_preds)
                int_target_accuracy = torch.eq(int_preds, targets)

        int_list_neg_accuracy = torch.eq(
            list_neg_preds,
            int_preds[neg_idx],
        )
        int_list_pos_accuracy = torch.eq(
            list_pos_preds,
            int_preds[pos_idx],
        )

        list_target_accuracy_dom = list_target_accuracy.tolist()
        int_list_accuracy_dom = int_list_accuracy.tolist()
        int_target_accuracy_dom = int_target_accuracy.tolist()

        d = [domains[idx] for idx in range(len(domains)) if neg_idx[idx]]
        int_list_neg_accuracy_dom = (int_list_neg_accuracy, d)

        d = [domains[idx] for idx in range(len(domains)) if pos_idx[idx]]
        int_list_pos_accuracy_dom = (int_list_pos_accuracy, d)

        list_target_accuracy = list_target_accuracy.sum()
        int_list_accuracy = int_list_accuracy.sum()
        int_target_accuracy = int_target_accuracy.sum()
        int_list_neg_accuracy = int_list_neg_accuracy.sum()
        int_list_pos_accuracy = int_list_pos_accuracy.sum()

        # cast to list
        int_list_accuracy = int_list_accuracy.tolist()
        list_target_accuracy = list_target_accuracy.tolist()
        int_target_accuracy = int_target_accuracy.tolist()
        int_list_neg_accuracy = int_list_neg_accuracy.tolist()
        int_list_pos_accuracy = int_list_pos_accuracy.tolist()
        list_preds = list_preds.tolist()
        int_preds = int_preds.tolist()

        # build dict
        aux = dict(
            # accuracy
            int_list_accuracy=int_list_accuracy,
            list_target_accuracy=list_target_accuracy,
            int_target_accuracy=int_target_accuracy,
            int_list_neg_accuracy=int_list_neg_accuracy,
            int_list_pos_accuracy=int_list_pos_accuracy,
            # preds
            list_preds=list_preds,
            int_preds=int_preds,
            neg_pred_len=len(list_neg_preds),
            pos_pred_len=len(list_pos_preds),
            # domain specific
            list_target_accuracy_dom=list_target_accuracy_dom,
            int_list_accuracy_dom=int_list_accuracy_dom,
            int_target_accuracy_dom=int_target_accuracy_dom,
            int_list_neg_accuracy_dom=int_list_neg_accuracy_dom,
            int_list_pos_accuracy_dom=int_list_pos_accuracy_dom,
            domains=domains,
            # divergence stats
            kl_div=kl_div,

        )

        return aux
