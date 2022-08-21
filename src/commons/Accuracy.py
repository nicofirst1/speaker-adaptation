import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AccuracyEstimator(torch.nn.Module):

    def __init__(self, list_domain, sim_model_type,all_domains):

        super().__init__()

        self.list_domain = list_domain
        self.sim_model_type = sim_model_type


        # create an index dict for domains
        self.domain2idx = {d: idx for idx, d in enumerate(sorted(all_domains))}
        self.idx2domain = {idx: d for idx, d in self.domain2idx.items()}
    def forward(self, preds, targets, list_out, domains, is_adaptive=False):

        if self.sim_model_type == "multi":
            if is_adaptive:
                preds=preds[1]
            else:
                preds=preds[0]

        preds=preds.to("cpu")
        targets=targets.to("cpu")
        list_out=list_out.to("cpu")

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

        # with bce the simulator has a binary output,
        # so the making of the aux dict with the accuracies differs
        if self.sim_model_type =="binary":

            # sim out is logits, need bool
            sim_preds = torch.sigmoid(preds)
            sim_preds = torch.round(sim_preds)
            sim_preds = sim_preds.bool()

            if sim_preds.ndim != 1:
                sim_preds = sim_preds.squeeze(dim=-1)

            sim_list_accuracy = torch.eq(list_target_accuracy, sim_preds)

            if is_adaptive:
                # use vector of ones if adaptive
                sim_target_accuracy = torch.eq(sim_preds, torch.ones(list_target_accuracy.shape).bool())
            else:
                sim_target_accuracy = sim_list_accuracy

            list_neg_preds = list_target_accuracy[neg_idx]
            list_pos_preds = list_target_accuracy[pos_idx]

        else:
            sim_preds = torch.argmax(preds.squeeze(dim=-1), dim=1)

            list_neg_preds = list_preds[neg_idx]
            list_pos_preds = list_preds[pos_idx]

            # simulator is predicting the domain of the target
            if self.sim_model_type == "domain":
                sim_target_accuracy = torch.eq(sim_preds, doms)
                # there is no sim_list accuracy when predicting domain, set to -1
                sim_list_accuracy = torch.zeros(sim_target_accuracy.shape) - 1


            else:
                # sim is predicting the listener output
                sim_list_accuracy = torch.eq(list_preds, sim_preds)
                sim_target_accuracy = sim_list_accuracy


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

            # domain specific
            list_target_accuracy_dom=list_target_accuracy_dom,
            sim_list_accuracy_dom=sim_list_accuracy_dom,
            sim_target_accuracy_dom=sim_target_accuracy_dom,
            sim_list_neg_accuracy_dom=sim_list_neg_accuracy_dom,
            sim_list_pos_accuracy_dom=sim_list_pos_accuracy_dom,
            domains=domains,

        )

        return aux

