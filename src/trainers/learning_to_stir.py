import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rich.progress
import spacy as spacy
import torch
import wandb
from numpy import mean
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, AccuracyEstimator,
                         EarlyStopping, IntLossAdapt, IntLossPretrain,
                         MTLOptim, get_dataloaders, get_domain_accuracy,
                         load_wandb_checkpoint, mask_attn, merge_dict,
                         parse_args, save_model, speak2list_vocab,
                         translate_utterance, set_seed, get_int_chk)
from src.data.dataloaders import Vocab
from src.models import get_model
from src.wandb_logging import ListenerLogger
from torch import optim
from torch.utils.data import DataLoader


def make_hypo_table(hypos, list_target_accuracy) -> wandb.Table:
    """
    make a table from the hypotesis list
    """

    # add hypo table
    nlp = spacy.load("en_core_web_lg")
    for idx in range(len(hypos)):
        list_acc = list_target_accuracy[idx][-1]
        h = hypos[idx]
        # estimate disintilarity between first and last hypo
        if len(h) > 1:

            doc1 = nlp(str(h[0]))
            doc2 = nlp(str(h[-1]))
            int = doc1.intilarity(doc2)
        else:
            int = 0

        if len(h) < common_p.s_iter + 1:
            hypos[idx] += ["/"] * (common_p.s_iter + 1 - len(hypos[idx]))
        hypos[idx] += [int, list_acc]

    columns = ["original_hypo"]
    columns += [f"hypo_{i}" for i in range(common_p.s_iter)]
    columns += ["intilarity", "is list correct?"]

    return wandb.Table(columns, data=hypos)


def normalize_aux(aux, data_length, s_iter):
    mean_s = mean([len(x) for x in aux["int_list_accuracy"]])
    accs = sum([x[-1] for x in aux["list_target_accuracy"]]) / data_length
    aux["loss"] = mean([x[-1] for x in aux.pop("loss")])

    if "list_loss" in aux.keys():
        aux["list_loss"] = mean([x[-1] for x in aux.pop("list_loss")])
    if "eq_loss" in aux.keys():
        aux["eq_loss"] = mean([x[-1] for x in aux.pop("eq_loss")])

    aux["accs"] = accs
    aux["mean_s"] = mean_s

    # best attempt to tie accuracy to mean_s
    s_norm = 1 - mean_s / common_p.s_iter
    aux["s_acc"] = s_norm * accs

    # aux["int_loss"] = np.mean(aux["int_loss"])

    # a function to recursively flatten a list of lists into a single list
    def weighted_acc(l: List[List[float]], use_w=True) -> List[float]:
        """
        Return a list of weighted accuracies. Since each elem in the list is a list of len [1,s_iter],
        the shortest the list the better the accuracy, so filter out all the lists with only zeros and
        use the following formula for estimating accuracy for the rest:
        acc=(s_iter -len(lst)+1)/s_iter
        """
        res = []
        for elem in l:
            # no right predictions
            if set(elem) == {0}:
                acc = 0
            else:
                w = (s_iter - len(elem) + 1) / s_iter
                acc = elem[-1]
                if use_w:
                    acc = w * acc

            res.append(acc)
        return res

    # aux["int_list_accuracy_w"] = np.sum(weighted_acc(aux["int_list_accuracy"])) / data_length
    # aux["list_target_accuracy_w"] = np.sum(weighted_acc(aux["list_target_accuracy"])) / data_length
    # aux["int_target_accuracy_w"] = np.sum(weighted_acc(aux["int_target_accuracy"])) / data_length
    # aux["int_list_neg_accuracy_w"] = np.sum(weighted_acc(aux["int_list_neg_accuracy"])) / np.sum(
    #     np.sum(aux["neg_pred_len"]))
    # aux["int_list_pos_accuracy_w"] = np.sum(weighted_acc(aux["int_list_pos_accuracy"])) / np.sum(
    #     np.sum(aux["pos_pred_len"]))

    aux["int_list_accuracy"] = (
        np.sum(weighted_acc(aux["int_list_accuracy"], use_w=False)) / data_length
    )
    aux["list_target_accuracy"] = (
        np.sum(weighted_acc(aux["list_target_accuracy"], use_w=False)) / data_length
    )
    aux["int_target_accuracy"] = (
        np.sum(weighted_acc(aux["int_target_accuracy"], use_w=False)) / data_length
    )

    # aux["int_list_neg_accuracy"] = np.sum(weighted_acc(aux["int_list_neg_accuracy"],use_w=False)) / np.sum(
    #     np.sum(aux["neg_pred_len"]))
    # aux["int_list_pos_accuracy"] = np.sum(weighted_acc(aux["int_list_pos_accuracy"],use_w=False)) / np.sum(
    #     np.sum(aux["pos_pred_len"]))

    def flatten(lst):

        for idx in range(len(lst)):
            lst[idx] = [x for sub in lst[idx] for x in sub]

        lst = [x for sub in lst for x in sub]

        return lst

    domains = [x[-1] for x in aux.pop("domains")]
    domains = [x for sub in domains for x in sub]

    accs = [x[-1] for x in aux.pop("list_target_accuracy_dom")]
    accs = [x for sub in accs for x in sub]
    aux["domain/list_target_acc"] = get_domain_accuracy(accs, domains, logger.domains)

    accs = [x[-1] for x in aux.pop("int_list_accuracy_dom")]
    accs = [x for sub in accs for x in sub]
    aux["domain/int_list_acc"] = get_domain_accuracy(accs, domains, logger.domains)

    accs = [x[-1] for x in aux.pop("int_target_accuracy_dom")]
    accs = [x for sub in accs for x in sub]
    aux["domain/int_target_acc"] = get_domain_accuracy(accs, domains, logger.domains)

    # flatten nested lists
    aux["list_preds"] = flatten(aux["list_preds"])
    aux["int_preds"] = flatten(aux["int_preds"])

    aux.pop("list_preds")
    aux.pop("int_preds")
    aux.pop("int_list_neg_accuracy_dom")
    aux.pop("int_list_pos_accuracy_dom")
    aux.pop("int_list_neg_accuracy")
    aux.pop("int_list_pos_accuracy")
    aux.pop("neg_pred_len")
    aux.pop("pos_pred_len")


def get_predictions(
    data: Dict,
    int_model: torch.nn.Module,
    speak_model: torch.nn.Module,
    list_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    adapt_loss_f: IntLossAdapt,
    acc_estimator: AccuracyEstimator,
    adapt_lr: float,
    s_iter: int,
    list_vocab: Vocab,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/int out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["separate_images"]
    context_concat = data["concat_context"]
    targets = data["target"]
    prev_hist = data["prev_histories"]
    prev_utterance = data["prev_utterance"]
    prev_utt_lengths = data["prev_length"]
    target_img_feats = data["target_img_feats"]
    batch_size = context_concat.size(0)
    device = list_model.device

    max_length_tensor = prev_utterance.shape[1]
    speak_masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

    ################################################
    #   Get results with original hypo
    ################################################
    # generate hypothesis
    utts, logs, decoder_hid = speak_model.generate_hypothesis(
        prev_utterance,
        prev_utt_lengths,
        context_concat,
        target_img_feats,
    )

    translator(utts)
    origin_h = [list_vocab.decode(sent) for sent in utts]

    history_att = logs["history_att"]

    ################################################
    #   Get results with adapted hypo
    ################################################
    h0 = decoder_hid.clone().detach().requires_grad_(True)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["params"] = [h0]
    else:
        optimizer.add_param_group({"params": h0, "lr": adapt_lr})

    losses = []
    hypos = []

    # perform loop
    i = 0
    infos = []
    while i < s_iter:

        set_seed(seed)

        # get modified hypo
        utts = speak_model.nucleus_sampling(h0, history_att, speak_masks)
        translator(utts)

        hypo = [list_vocab.decode(sent) for sent in utts]

        hypos.append(hypo)

        # get lengths
        lengths = []

        for idx in range(batch_size):
            nz = torch.nonzero(utts[idx])
            if len(nz) > 1:
                nz = torch.max(nz) + 1
            else:
                nz = 1
            lengths.append(nz)

        max_len = max(lengths)

        utts = utts.to(device)
        masks = mask_attn(lengths, max_len, device)

        if batch_size == 1:
            masks = masks.squeeze(1)

        list_out = list_model(utts, context_separate, context_concat, prev_hist, masks)

        int_out = int_model(h0, context_separate, context_concat, prev_hist, masks)

        # compute loss for pretraining
        loss = adapt_loss_f(int_out, targets, list_out, data["domain"])


        # compute loss for adaptation
        loss.backward(retain_graph=True)


        # params = {f"int/{k}":v for k, v in dict(list(int_model.named_parameters())).items()}
        # params.update({f"list/{k}":v for k, v in dict(list(list_model.named_parameters())).items()})
        # params.update({f"speak/{k}":v for k, v in dict(list(speak_model.named_parameters())).items()})
        # draw_grad_graph(params, h0, a_loss, file_name="./adaptive_grad.png")

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().cpu().item())

        # get  accuracy

        info = acc_estimator(
            int_out, targets, list_out, data["domain"], is_adaptive=True
        )

        # add loss
        info["loss"] = loss.detach().cpu().item()
        # append to list
        infos.append(info)

        # break if int gets it right

        if info["list_target_accuracy"] :
            break
        i += 1

    aux = dict(
        hypos=hypos,
        loss=losses,
    )
    infos = merge_dict(infos)
    aux["info"] = infos

    return mean(losses), aux


def process_epoch(
    data_loader: DataLoader,
    int_model: torch.nn.Module,
    speaker_model: torch.nn.Module,
    list_model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer],
    list_vocab: Vocab,
    adapt_loss_f: torch.nn.Module,
    acc_estimator: torch.nn.Module,
    split: str,
    common_p,
) -> Dict:
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """
    auxs = []
    for i, data in rich.progress.track(
        enumerate(data_loader),
        total=len(data_loader),
        description=f"{split} epoch {epoch}",
    ):
        # get datapoints
        loss, aux = get_predictions(
            data,
            int_model,
            speaker_model,
            list_model,
            optimizer,
            adapt_loss_f,
            acc_estimator,
            common_p.adapt_lr,
            common_p.s_iter,
            list_vocab,
        )

        auxs.append(aux)

    aux = merge_dict(auxs)

    infos = aux.pop("info")
    infos = merge_dict(infos)
    # aux[f"hypo_table_epoch{epoch}"] = make_hypo_table(aux.pop('hypos'), infos['list_target_accuracy'])
    normalize_aux(infos, len(data_loader.dataset.data), s_iter=common_p.s_iter)

    aux.update({f"adaptive/{k}": v for k, v in infos.items()})
    aux["loss"] = mean([x[-1] for x in aux.pop("loss")])


    return aux


if __name__ == "__main__":

    img_dim = 2048

    common_p = parse_args("int")
    domain = common_p.train_domain
    device = common_p.device

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]

    speak_vocab = Vocab(parse_args("speak").vocab_file, is_speaker=True)

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="learning2stir",
    )

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(
        LISTENER_CHK_DICT[domain],
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

    # for reproducibility
    seed = list_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    model = get_model("list", list_args.model_type)
    list_model = model(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    for param in list_model.parameters():
        param.requires_grad = False

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK,
        device,
    )  # datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1]))
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    # init speak model and load state

    model = get_model("speak", speak_p.model_type)
    speaker_model = model(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.beam_size,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
        use_beam=common_speak_p.use_beam,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # Interpreter
    ##########################

    if common_p.force_resume_url == "":
        check = get_int_chk(common_p.model_type, common_p.pretrain_loss, domain)
    else:
        check = common_p.force_resume_url
    int_check, _ = load_wandb_checkpoint(check, device)

    model = get_model("int", common_p.model_type)
    int_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.train_domain,
        common_p.device,
    ).to(device)

    if common_p.type_of_int != "untrained":
        int_model.load_state_dict(int_check["model_state_dict"])

    int_model = int_model.to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################


    optimizer = optim.Adam(
        list(int_model.parameters()) , lr=common_p.learning_rate
    )

    adapt_loss_f = IntLossAdapt(
        common_p.adaptive_loss,
        common_p.reduction,
        common_p.model_type,
        alpha=common_p.focal_alpha,
        gamma=common_p.focal_gamma,
        list_domain=domain,
        all_domains=logger.domains,
    )

    acc_estimator = AccuracyEstimator(
        domain, common_p.model_type, all_domains=logger.domains
    )

    ###################################
    ## RESUME
    ###################################


    metric = common_p.metric

    if metric == "loss":

        es = EarlyStopping(common_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(common_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    # logger.watch_model([int_model])

    ###################################
    ##  Get speaker dataloader
    ###################################

    data_domain = common_p.data_domain

    training_loader, test_loader, val_loader = get_dataloaders(
        common_p, speak_vocab, "all"
    )
    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)
    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(common_p.epochs):

        print("Epoch : ", epoch)

        auxs = []
        data = {}

        int_model.train()
        int_model.use_batchnorm(False)

        ###################################
        ##  TRAIN LOOP
        ###################################
        for param in int_model.parameters():
            param.requires_grad = True

        with torch.set_grad_enabled(True):
            aux = process_epoch(
                training_loader,
                int_model,
                speaker_model,
                list_model,
                optimizer,
                list_vocab,
                adapt_loss_f,
                acc_estimator,
                "train",
                common_p,
            )

        logger.on_eval_end(
            aux,
            list_domain=training_loader.dataset.domain,
            modality="train",
            commit=True,
        )

        print(
            f"Train loss {mean(aux['loss']):.6f}, accuracy {mean(aux['adaptive/accs']):.4f} "
        )

        ###################################
        ##  EVAL LOOP
        ###################################

        int_model.eval()

        for param in int_model.parameters():
            param.requires_grad = False

        print(f"\nEvaluation")

        aux = process_epoch(
            val_loader,
            int_model,
            speaker_model,
            list_model,
            optimizer,
            list_vocab,
            adapt_loss_f,
            acc_estimator,
            "eval",
            common_p,
        )

        logger.on_eval_end(
            aux,
            list_domain=val_loader.dataset.domain,
            modality="eval",
            commit=True,
        )
        eval_loss = mean(aux["loss"])
        eval_accuracy = mean(aux["adaptive/accs"])
        print(f"Eval loss {eval_loss:.6f}, accuracy {eval_accuracy:.4f} ")

        save_model(
            model=int_model,
            model_type="Interpreter",
            epoch=epoch,
            accuracy=eval_accuracy,
            optimizer=optimizer,
            args=common_p,
            timestamp=timestamp,
            logger=logger,
            loss=eval_loss,
        )

        # check for early stopping
        metric_val = eval_loss if common_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
