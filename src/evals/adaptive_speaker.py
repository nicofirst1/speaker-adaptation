import datetime
from typing import List

import numpy as np
import pandas as pd
import rich.progress
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from src.commons import (
    LISTENER_CHK_DICT,
    SIM_DOMAIN_CHK,
    SPEAKER_CHK,
    get_dataloaders,
    hypo2utterance,
    load_wandb_checkpoint,
    mask_attn,
    parse_args,
)
from src.data.dataloaders import Vocab
from src.models import ListenerModel_hist, SimulatorModel_hist, get_model
from src.models.speaker.model_speaker_hist import SpeakerModel_hist
from src.wandb_logging import ListenerLogger


def generate_table(data: List, target_domain: List, s: int) -> wandb.Table:
    # unflatten inner lists
    new_data = []
    for row, tg in zip(data, target_domain):
        new_row = [tg]
        for elem in row:
            if isinstance(elem, list):
                new_row += elem
            else:
                new_row.append(elem)
        new_data.append(new_row)
    data = new_data

    table_columns = [
        "target domain",
        "original hypo",
        "original guess",
    ]
    table_columns += [f"adapted hypo {i}" for i in range(s)]
    table_columns += [f"adapted guess {i}" for i in range(s)]
    table_columns += [f"diff {i}" for i in range(s)]

    table = wandb.Table(columns=table_columns, data=data)
    return table


def evaluate(
        data_loader: DataLoader,
        speak_model: SpeakerModel_hist,
        sim_model: SimulatorModel_hist,
        list_model: ListenerModel_hist,
        list_vocab:Vocab,
        criterion,
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
    sim_model
    list_model
    criterion
    split
    lr
    s

    Returns
    -------
    dataloader for analysis

    """

    adapted_list_outs = []
    adapted_sim_outs = []

    original_accs = []
    adapted_accs = []

    original_hypos = []
    modified_hypos = []

    h0s = []
    csv_data = []

    target_domain = []
    for ii, data in rich.progress.track(
            enumerate(data_loader),
            total=len(data_loader),
            description=f"Evaluating on split {split}",
    ):

        ## extract data
        context_separate = data["separate_images"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        context_concat = data["concat_context"]
        target_img_feats = data["target_img_feats"]
        targets = data["target"]
        prev_hist = data["prev_histories"]
        original_utt_ids = data['utterance']

        max_length_tensor = prev_utterance.shape[1]
        speak_masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

        # generate hypothesis
        origin_hypo, logs, decoder_hid = speak_model.generate_hypothesis(
            prev_utterance,
            prev_utt_lengths,
            context_concat,
            target_img_feats,
        )

        original_hypos.append(origin_hypo)
        history_att = logs["history_att"]

        ##################
        # Analysis only
        ##################
        lengths = [original_utt_ids.shape[1]]
        max_length_tensor = original_utt_ids.shape[1]
        masks = mask_attn(lengths, max_length_tensor, device)

        golden_list_out = list_model(
            original_utt_ids, context_separate, context_concat, prev_hist, masks
        )
        golden_list_out.squeeze(dim=0)
        golden_list_acc = torch.argmax(golden_list_out.squeeze(dim=-1), dim=1)
        golden_list_acc = torch.eq(golden_list_acc, targets.squeeze()).double().item()

        ##################
        # Analysis only
        ##################

        ################################################
        #   Get results with original hypo
        ################################################
        # translate utt to ids and feed to listener
        utterance = hypo2utterance(origin_hypo, list_vocab)
        lengths = [utterance.shape[1]]
        max_length_tensor = utterance.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        list_out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks
        )
        original_list_out=list_out.squeeze(dim=0)

        # get  accuracy
        list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
        list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
        original_accs.append(list_target_accuracy)

        ################################################
        #   Get results with adapted hypo
        ################################################
        h0 = decoder_hid.clone().detach().requires_grad_(True)
        #todo: move optim out of loop (?)
        optimizer = torch.optim.Adam([h0], lr=lr)

        # repeat for s interations
        s_hypo = []
        s_accs = []
        s_h0 = []
        s_adapted_list_outs = []
        s_adapted_sim_outs = []

        # perform loop
        for i in range(s):

            sim_out = sim_model(h0, context_separate, context_concat, prev_hist, masks)
            s_adapted_sim_outs.append(sim_out.squeeze(dim=0).tolist())

            # compute loss and perform backprop
            loss = criterion(sim_out, targets)
            loss.backward()
            optimizer.step()
            s_h0.append(h0[0].clone().detach().tolist())


            # get modified hypo
            hypo = speak_model.beam_serach(h0, history_att, speak_masks)
            s_hypo.append(hypo)

            # translate utt to ids and feed to listener
            utterance = hypo2utterance(hypo, list_vocab)
            lengths = [utterance.shape[1]]
            max_length_tensor = utterance.shape[1]

            masks = mask_attn(lengths, max_length_tensor, device)

            list_out = list_model(
                utterance, context_separate, context_concat, prev_hist, masks
            )
            s_adapted_list_outs.append(list_out.squeeze(dim=0).tolist())

            # get  accuracy
            list_preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
            list_target_accuracy = torch.eq(list_preds, targets.squeeze()).double().item()
            s_accs.append(list_target_accuracy)

        adapted_sim_outs.append(s_adapted_sim_outs)
        adapted_list_outs.append(s_adapted_list_outs)
        modified_hypos.append(s_hypo)
        adapted_accs.append(s_accs)
        h0s.append([decoder_hid] + s_h0)
        target_domain.append(data['domain'][0])

        ##########################
        # CSV generation
        ##########################

        # -----------------
        # extract info from datapoint
        # -----------------

        target_img_idx = data['target'][0].item()
        # remove target from set
        distractors_img_path = data['image_set'][0]
        distractors_img_path = [str(x) for x in distractors_img_path]
        distractors_img_path = [logger.img_id2path[x] for x in distractors_img_path]

        # -----------------
        # fill in rows
        # -----------------
        # the informations to log are:
        # 1. target domain X 1
        # 2. list domain X 1
        # 3. simulator domain X 1
        # 4. target index X 1
        # 5. distractor+target img paths X 6
        # 6. golden captions X 1
        # 7. original hypo  X 1
        # 8. adapted utterances (s) x s
        # 9. original h0 (decoder_hid) x 1
        # 10. each h0 after a backprop (s h0s) x s
        # 11. the listener's output distribution given the golden caption x1
        # 12. the listener's output distribution given the non-adapted/original utterance x1
        # 14. the listener's output distribution given the adapted utterance (for each backprop step) xs
        # 15. the simulator's output distribution given h0 x1
        # 16. the simulator's output distribution given h0' (for each backprop step) xs
        # 17. whether the listener makes a correct guess given the original utterance x1
        # 18. whether the listener makes a correct guess given the adapted utterance (for each backprop step) x(s-1)

        # size formula : 16+5s

        row = [data['domain'][0], list_model.domain, sim_model.domain, target_img_idx]
        row += distractors_img_path
        row += [data['orig_utterance'][0], origin_hypo]
        row += s_hypo
        row += [decoder_hid[0].tolist()]
        row += s_h0
        row += [golden_list_out.squeeze(dim=0).tolist()]
        row += [original_list_out.tolist()]
        row += s_adapted_list_outs
        row += s_adapted_sim_outs
        row += [golden_list_acc]
        row += s_accs

        csv_data.append(row)

    # Generate data for wandb table
    table_data = list(
        list(
            zip(
                original_hypos,
                original_accs,
                modified_hypos,
                adapted_accs,
                [[x != y1 for y1 in y] for x, y in zip(original_accs, adapted_accs)],
            )
        )
    )

    table = generate_table(table_data, target_domain, s)

    ## csv columns
    columns = ["target domain", "listener domain", "simulator domain", "target img idx"]
    columns += [f"img path #{x}" for x in range(6)]
    columns += ["golden utt", "original utt"]
    columns += [f"adapted utt s{i}" for i in range(s)]
    columns += ["original h0"]
    columns += [f"adapted h0 s{i}" for i in range(s)]
    columns += ["golden_list_out", "original_list_out"]
    columns += [f"adapted_list_out_s{i}" for i in range(s)]
    columns += ["original_sim_out"]
    columns += [f"adapted_sim_out_s{i}" for i in range(s-1)]
    columns += [f"original_acc"]
    columns += [f"adapted_accuracy_s{i}" for i in range(s)]

    df = pd.DataFrame(columns=columns, data=csv_data)

    original_accs = np.mean(original_accs)
    adapted_accs = np.mean([x[-1] for x in adapted_accs])

    metrics = dict(
        original_accs=original_accs,
        modified_accs=adapted_accs,
        hypo_table=table,
    )

    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality=split)

    return df


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_dim = 2048

    common_p = parse_args("sim")
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
    seed = list_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        use_beam=common_speak_p.use_beam
    )


    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################
    sim_check, _ = load_wandb_checkpoint(SIM_DOMAIN_CHK[domain], device)
    # load args
    sim_p = sim_check["args"]
    sim_p.train_domain = domain
    sim_p.device = device

    # for debug
    sim_p.subset_size = common_p.subset_size
    sim_p.debug = common_p.debug
    sim_p.s_iter = common_p.s_iter
    sim_p.alpha = common_p.alpha

    sim_p.reset_paths()

    model = get_model("sim", sim_p.model_type)
    sim_model = model(
        len(list_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        sim_p.dropout_prob,
        sim_p.train_domain,
        sim_p.device,
    ).to(device)

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(sim_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="speaker-influence",
    )

    metric = sim_p.metric
    sweep_config = wandb.config

    ###################################
    ##  Get speaker dataloader
    ###################################
    bs = sim_p.batch_size
    # need batchsize =1 for generating hypothesis
    sim_p.batch_size = 1
    train_dl_dom, _, val_dl_dom = get_dataloaders(sim_p, speak_vocab, domain)
    train_dl_all, _, val_dl_all = get_dataloaders(sim_p, speak_vocab, domain="all")

    ###################################
    ##  LOSS
    ###################################

    cel = nn.CrossEntropyLoss(reduction=sim_p.reduction)

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    ###################################
    ##  EVAL LOOP
    ###################################

    sim_model.eval()

    if common_p.log_train:
        print(f"\nEvaluation on train for domain {domain}")
        df = evaluate(
            train_dl_dom,
            speaker_model,
            sim_model,
            list_model,
            list_vocab,
            criterion=cel,
            split="in_domain_train",
            lr=sweep_config.learning_rate,
            s=sweep_config.s_iter,
        )

        ### saving df
        file_name = "tmp.csv"
        df.to_csv(file_name)

        logger.log_artifact(file_name,
                            f"adaptive_speak_train_in_domain_{domain}",
                            "csv",
                            metadata=sim_p, )

    print(f"\nEvaluation on val for domain {domain}")
    df = evaluate(
        val_dl_dom,
        speaker_model,
        sim_model,
        list_model,
        list_vocab,
        criterion=cel,
        split="in_domain_val",
        lr=sweep_config.learning_rate,
        s=sweep_config.s_iter,
    )

    ### saving df
    file_name = "tmp.csv"
    df.to_csv(file_name)

    logger.log_artifact(file_name,
                        f"adaptive_speak_eval_in_domain_{domain}",
                        "csv",
                        metadata=sim_p, )

    if common_p.log_train:
        print(f"\nEvaluation on train for domain all")
        df = evaluate(
            train_dl_all,
            speaker_model,
            sim_model,
            list_model,
            list_vocab,
            criterion=cel,
            split="out_domain_train",
            lr=sweep_config.learning_rate,
            s=sweep_config.s_iter,
        )

        ### saving df
        file_name = "tmp.csv"
        df.to_csv(file_name)

        logger.log_artifact(file_name,
                            f"adaptive_speak_train_out_domain_{domain}",
                            "csv",
                            metadata=sim_p, )

    print(f"\nEvaluation on val for domain all")
    df = evaluate(
        val_dl_all,
        speaker_model,
        sim_model,
        list_model,
        list_vocab,
        criterion=cel,
        split="out_domain_val",
        lr=sweep_config.learning_rate,
        s=sweep_config.s_iter,
    )

    ### saving df
    file_name = "tmp.csv"
    df.to_csv(file_name)

    logger.log_artifact(file_name,
                        f"adaptive_speak_eval_out_domain_{domain}",
                        "csv",
                        metadata=sim_p, )

    logger.on_train_end({}, epoch_id=0)
