import datetime
from typing import List

import numpy as np
import pandas as pd
import rich.progress
import torch
from torch import nn

import wandb
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, AccuracyEstimator,
                         IntLossAdapt, get_dataloaders, get_int_chk,
                         load_wandb_checkpoint, mask_attn, parse_args,
                         set_seed, mask_oov_embeds, speak2list_vocab, translate_utterance)
from src.data.dataloaders import Vocab
from src.evals.adaptive_speaker import generate_ood_table
from src.models import InterpreterModel_no_hist, ListenerModel_hist, get_model
from src.models.speaker.model_speaker_hist import SpeakerModel_hist
from src.wandb_logging import ListenerLogger
from torch.utils.data import DataLoader


def evaluate(
        data_loader: DataLoader,
        speak_model: SpeakerModel_hist,
        int_model: InterpreterModel_no_hist,
        list_model: ListenerModel_hist,
        list_vocab: Vocab,
        adaptation_criterion: IntLossAdapt,
        continual_criterion: torch.nn.Module,
        continual_optimizer: torch.optim.Optimizer,
        acc_estimator: AccuracyEstimator,
        split: str,
        lr: float = 0.1,
        s: int = 1,
) -> pd.DataFrame:
    """
    Perform evaluation of given split
    Parameters
    ----------
    data_loader
    speak_model
    int_model
    list_model
    adaptation_criterion
    split
    lr
    s

    Returns
    -------
    dataloader for analysis

    """
    adapted_list_outs = []
    adapted_int_outs = []

    original_accs = []
    golden_accs = []
    adapted_accs = []
    int_accs = []
    int_list_accs = []

    original_hypos = []
    golden_hypos = []
    modified_hypos = []
    losses_adapt = []
    losses_continual = []
    h0s = []
    grads = []

    in_domain = list_model.domain == data_loader.dataset.domain

    csv_data = []

    target_domain = []
    for ii, data in rich.progress.track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"Evaluating on split {split}",
    ):

        # filter out indomain data points
        # if not in_domain:
        #     if data["domain"][0] == list_model.domain:
        #         continue

        ## extract data
        context_separate = data["separate_images"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        context_concat = data["concat_context"]
        target_img_feats = data["target_img_feats"]
        targets = data["target"]
        prev_hist = data["prev_histories"]
        golden_utt_ids = data["utterance"]

        max_length_tensor = prev_utterance.shape[1]
        speak_masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

        ##################################
        # Get results for golden captions
        ##################################
        lengths = [golden_utt_ids.shape[1]]
        max_length_tensor = golden_utt_ids.shape[1]
        masks = mask_attn(lengths, max_length_tensor, device)

        golden_list_out = list_model(
            golden_utt_ids, context_separate, context_concat, prev_hist, masks
        )
        golden_list_out.squeeze(dim=0)
        golden_acc = torch.argmax(golden_list_out.squeeze(dim=-1), dim=1)
        golden_acc = torch.eq(golden_acc, targets.squeeze()).double().item()
        golden_accs.append(golden_acc)

        golden_hypos += data["orig_utterance"]

        ################################################
        #   Get results with original hypo
        ################################################
        # generate hypothesis
        utterance, logs, decoder_hid = speak_model.generate_hypothesis(
            prev_utterance,
            prev_utt_lengths,
            context_concat,
            target_img_feats,
        )

        translator(utterance)

        origin_hypo = [list_vocab.decode(sent) for sent in utterance]

        original_hypos.append(origin_hypo)
        history_att = logs["history_att"]

        # translate utt to ids and feed to listener
        lengths = [utterance.shape[1]]
        max_length_tensor = utterance.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        list_out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks
        )
        original_list_out = list_out.squeeze(dim=0)

        # get  accuracy
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
        original_accs.append(list_target_accuracy)

        ################################################
        #   Get results with adapted hypo
        ################################################
        # decoder_hid = normalize(decoder_hid)
        h0 = decoder_hid.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([h0], lr=lr)
        optimizer.zero_grad()
        continual_optimizer.zero_grad()

        # repeat for s interations
        s_hypo = ["" for _ in range(s)]
        s_accs = [-1 for _ in range(s)]
        s_h0 = [-1 for _ in range(s)]
        s_adapted_list_outs = [-1 for _ in range(s)]
        s_adapted_int_outs = [-1 for _ in range(s)]
        s_loss_adapt = [-1 for _ in range(s)]
        s_loss_continual = [-1 for _ in range(s)]
        s_grad = [-1 for _ in range(s)]
        int_accuracy = [-1 for _ in range(s)]
        int_list_acc = [-1 for _ in range(s)]


        # perform loop
        i = 0
        while i < s:
            set_seed(seed)

            int_out = int_model(h0, context_separate, context_concat, prev_hist, masks)
            s_adapted_int_outs[i] = int_out.squeeze(dim=0).tolist()

            # compute loss and perform backprop
            loss_adapted = adaptation_criterion(int_out, targets, list_out, data["domain"])
            # aux = acc_estimator(
            #     int_out, targets, list_out, data["domain"], is_adaptive=True
            # )
            loss_adapted.backward(retain_graph=True)
            optimizer.step()

            s_loss_adapt[i] = loss_adapted.detach().item()
            s_h0[i] = h0[0].clone().detach().tolist()
            s_grad[i] = h0.grad[0].clone().detach().tolist()

            # get modified hypo
            utterance = speak_model.nucleus_sampling(h0, history_att, speak_masks)
            translator(utterance)
            hypo = [list_vocab.decode(sent) for sent in utterance]

            s_hypo[i] = hypo[0]
            # generate utt for list
            # translate utt to ids and feed to listener
            lengths = [utterance.shape[1]]
            max_length_tensor = utterance.shape[1]
            masks = mask_attn(lengths, max_length_tensor, device)

            list_out = list_model(
                utterance, context_separate, context_concat, prev_hist, masks
            )
            s_adapted_list_outs[i] = list_out.squeeze(dim=0).tolist()

            # get  accuracy
            list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
            list_target_accuracy = (
                torch.eq(list_preds, targets.squeeze()).double().item()
            )
            s_accs[i] = list_target_accuracy

            aux = acc_estimator(
                int_out, targets, list_out, data["domain"], is_adaptive=True
            )

            # update on int weights
            c_loss = continual_criterion(int_out, list_out)
            c_loss.backward()
            continual_optimizer.step()

            int_accuracy[i] = aux["int_target_accuracy"]
            int_list_acc[i] = aux["int_list_accuracy"]
            s_accs[i] = aux["list_target_accuracy"]

            s_loss_adapt[i] = loss_adapted.detach().item()
            s_loss_continual[i] = c_loss.detach().item()
            s_h0[i] = h0[0].clone().detach().tolist()
            s_grad[i] = h0.grad[0].clone().detach().tolist()

            # break if listener gets it right
            if aux["int_target_accuracy"]:
                break
            i += 1

        adapted_int_outs.append(s_adapted_int_outs)
        adapted_list_outs.append(s_adapted_list_outs)
        modified_hypos.append(s_hypo)
        adapted_accs.append(s_accs)
        int_accs.append(int_accuracy)
        int_list_accs.append(int_list_acc)
        h0s.append([decoder_hid] + s_h0)
        target_domain.append(data["domain"][0])
        losses_adapt.append(s_loss_adapt)
        losses_continual.append(s_loss_continual)
        grads.append(s_grad)

        ##########################
        # CSV generation
        ##########################

        # -----------------
        # extract info from datapoint
        # -----------------

        target_img_idx = data["target"][0].item()
        # remove target from set
        distractors_img_path = data["image_set"][0]
        distractors_img_path = [str(x) for x in distractors_img_path]
        distractors_img_path = [logger.img_id2path[x] for x in distractors_img_path]

        # -----------------
        # fill in rows
        # -----------------
        # the informations to log are:
        # 1. target domain X 1
        # 2. list domain X 1
        # 3. interpreter domain X 1
        # 4. target index X 1
        # 5. distractor+target img paths X 6
        # 6. golden captions X 1
        # 7. original hypo  X 1
        # 8. adapted utterances (s) x s
        # 9. original h0 (decoder_hid) x 1
        # 10. each h0 after a backprop (s h0s) x s
        # 11. each loss after a backprop (s h0s) x s
        # 11. each grad after a backprop (s h0s) x s
        # 12. the listener's output distribution given the golden caption x1
        # 13. the listener's output distribution given the non-adapted/original utterance x1
        # 14. the listener's output distribution given the adapted utterance (for each backprop step) xs
        # 15. the interpreter's output distribution given h0 x1
        # 16. the interpreter's output distribution given h0' (for each backprop step) xs
        # 17. whether the listener makes a correct guess given the caption x1
        # 18. whether the listener makes a correct guess given the original utterance x1
        # 19. whether the interpreter makes a correct guess given the adapted utterance (for each backprop step) x(s-1)
        # 20. whether the listener makes a correct guess given the adapted utterance (for each backprop step) x(s-1)
        # 20. Simulator accuracy on listener (for each backprop step) x(s-1)
        # size formula : 16+7s

        row = [data["domain"][0], list_model.domain, int_model.domain, target_img_idx]
        row += distractors_img_path
        row += [data["orig_utterance"][0], origin_hypo]
        row += s_hypo
        row += [decoder_hid[0].tolist()]
        row += s_h0
        row += s_loss_adapt
        row += s_grad
        row += [golden_list_out.squeeze(dim=0).tolist()]
        row += [original_list_out.tolist()]
        row += s_adapted_list_outs
        row += s_adapted_int_outs
        row += [golden_acc]
        row += [original_accs[-1]]
        row += int_accuracy
        row += s_accs
        row += int_list_acc

        csv_data.append(row)

    ## csv columns
    columns = [
        "target domain",
        "listener domain",
        "interpreter domain",
        "target img idx",
    ]
    columns += [f"img path #{x}" for x in range(6)]
    columns += ["golden utt", "original utt"]
    columns += [f"adapted utt s{i}" for i in range(s)]
    columns += ["original h0"]
    columns += [f"adapted h0 s{i}" for i in range(s)]
    columns += [f"loss s{i}" for i in range(s)]
    columns += [f"grad s{i}" for i in range(s)]
    columns += ["golden_list_out", "original_list_out"]
    columns += [f"adapted_list_out_s{i}" for i in range(s)]
    columns += ["original_int_out"]
    columns += [f"adapted_int_out_s{i}" for i in range(s - 1)]
    columns += [f"golden_acc"]
    columns += [f"original_acc"]
    columns += [f"int_acc_s{i}" for i in range(s)]
    columns += [f"adapted_acc_s{i}" for i in range(s)]
    columns += [f"int_list_acc{i}" for i in range(s)]

    df = pd.DataFrame(columns=columns, data=csv_data)

    ##############################
    # WANDB TABLE
    ##############################
    table_data = list(
        list(
            zip(
                golden_hypos,
                golden_accs,
                original_hypos,
                original_accs,
                modified_hypos,
                adapted_accs,
                int_accs,
                int_list_accs,
                [[x != y1 for y1 in y] for x, y in zip(original_accs, adapted_accs)],
            )
        )
    )

    # hypo_table = generate_hypo_table(table_data, target_domain, s)
    ood_table = generate_ood_table(df, s)

    ##############################
    # METRICS
    ##############################

    original_accs = np.mean(original_accs)
    golden_accs = np.mean(golden_accs)
    adapted_accs = [[y for y in x if y != -1] for x in adapted_accs]
    mean_s = np.mean([len(x) for x in adapted_accs])
    adapted_accs = np.mean([x[-1] for x in adapted_accs])

    int_accs = [[y for y in x if y != -1] for x in int_accs]
    int_accs = np.mean([x[-1] for x in int_accs])

    int_list_accs = [[y for y in x if y != -1] for x in int_list_accs]
    int_list_accs = [x for x in int_list_accs if len(x)]
    int_list_accs = np.mean([x[-1] for x in int_list_accs])

    loss_adapted = [[y for y in x if y != -1] for x in losses_adapt]
    loss_adapted = np.mean(loss_adapted)

    loss_continual = [[y for y in x if y != -1] for x in losses_continual]
    loss_continual = np.mean(loss_continual)

    metrics = dict(
        original_accs=original_accs,
        adapted_accs=adapted_accs,
        int_accs=int_accs,
        golden_accs=golden_accs,
        # hypo_table=hypo_table,
        ood_table=ood_table,
        loss_adapted=loss_adapted,
        loss_continual=loss_continual,
        mean_s=mean_s,
        int_list_accs=int_list_accs,
    )

    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=split)

    return df


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("int")
    domain = common_p.train_domain

    ##########################
    # LISTENER
    ##########################

    list_checkpoint, _ = load_wandb_checkpoint(LISTENER_CHK_DICT[domain], device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.device = device
    list_args.reset_paths()

    # for debug
    list_args.subset_size = common_p.subset_size
    list_args.debug = common_p.debug

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # update paths
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

    # mask OOV words in the vocab

    with torch.no_grad():
        list_model.embeddings = mask_oov_embeds(list_model.embeddings, list_vocab, domain,
                                                replace_token=common_p.mask_oov_embed, data_path=common_p.data_path)

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

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
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # INTERPRETER
    ##########################

    if common_p.force_resume_url == "":
        check = get_int_chk(common_p.model_type, common_p.pretrain_loss, common_p.int_domain)
    else:
        check = common_p.force_resume_url
    int_check, _ = load_wandb_checkpoint(check, device)

    # load args
    int_p = int_check["args"]
    common_p.train_domain = domain
    common_p.device = device

    # for debug
    # common_p.subset_size = common_p.subset_size
    # common_p.debug = common_p.debug
    # common_p.s_iter = common_p.s_iter
    # common_p.adapt_lr=common_p.adapt_lr
    # common_p.learning_rate = common_p.learning_rate
    # common_p.type_of_int = common_p.type_of_int
    # common_p.seed = seed
    # common_p.test_split = common_p.test_split
    #
    # common_p.reset_paths()

    model = get_model("int", common_p.model_type)
    int_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.int_domain,
        common_p.device,
    ).to(device)

    if common_p.type_of_int != "untrained":
        int_model.load_state_dict(int_check["model_state_dict"])

    int_model = int_model.to(device)

    ###################################
    ##  LOGGER
    ###################################

    flag = common_p.type_of_int

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        project=f"continual-adaptation-{common_p.type_of_int}",
        tags=common_p.tags,
    )

    metric = common_p.metric
    sweep_config = wandb.config

    logger.watch_model([int_model], log_freq=10)

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = common_p.batch_size
    # need batchsize =1 for generating hypothesis
    int_p.batch_size = 1
    common_p.batch_size = 1
    train_dl_dom, test_dl_dom, val_dl_dom = get_dataloaders(common_p, speak_vocab, domain)
    train_dl_all, test_dl_all, val_dl_all = get_dataloaders(
        common_p, speak_vocab, domain="all"
    )

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator = translate_utterance(speak2list_v, device)

    ###################################
    ##  LOSS
    ###################################

    loss_f = IntLossAdapt(
        int_p.adaptive_loss,
        int_p.reduction,
        int_p.model_type,
        alpha=int_p.focal_alpha,
        gamma=int_p.focal_gamma,
        list_domain=domain,
        all_domains=logger.domains,
    )

    continual_criterion = torch.nn.CrossEntropyLoss(reduction=common_p.reduction)
    continual_optimizer = torch.optim.Adam(int_model.parameters(), lr=common_p.learning_rate)

    acc_estimator = AccuracyEstimator(
        domain, int_p.model_type, all_domains=logger.domains
    )

    ###################################
    ##  EVAL LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    int_model.train()

    ##################
    # OOD TEST
    ##################
    print(f"\nTest split for domain all")
    df = evaluate(
        test_dl_all,
        speaker_model,
        int_model,
        list_model,
        list_vocab,
        adaptation_criterion=loss_f,
        continual_criterion=continual_criterion,
        continual_optimizer=continual_optimizer,
        acc_estimator=acc_estimator,
        split="out_domain_test",
        lr=common_p.adapt_lr,
        s=common_p.s_iter,
    )

    ### saving df
    file_name = "tmp.csv"
    df.to_csv(file_name)

    logger.log_artifact(
        file_name,
        f"adaptive_speak_test_out_domain_{domain}",
        "csv",
        metadata=int_p,
    )

    logger.on_train_end({}, epoch_id=0)
