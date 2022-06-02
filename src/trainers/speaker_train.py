import copy
import operator
import os
import sys
from os.path import abspath, dirname

import numpy as np
import rich.progress
import torch.utils.data
from bert_score import score
from nlgeval import NLGEval
from torch import nn, optim

from src.commons import (get_dataloaders, load_wandb_checkpoint, mask_attn,
                         parse_args, save_model, EarlyStopping)
from src.data.dataloaders import Vocab
from src.models import SpeakerModelHistAtt
from src.wandb_logging import SpeakerLogger

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")

if not os.path.isdir("speaker_outputs"):
    os.mkdir("speaker_outputs")


# beam search
# topk
# topp

# built via modifying https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py


def eval_beam_histatt(
        split_data_loader,
        model,
        args,
        beam_size,
        max_len,
        nlgeval_obj,
        logger,
):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = []
    hypotheses = []

    for i, data in enumerate(split_data_loader):
        # print(i)

        completed_sentences = []
        completed_scores = []

        beam_k = beam_size

        ref = data["reference_chain"][
            0
        ]  # batch size 1  # full set of references for a single instance

        hypo, model_params = model.generate_hypothesis(data, beam_k, max_len)
        references.append(ref)
        hypotheses.append(hypo)

    # Calculate scores
    metrics_dict = nlgeval_obj.compute_metrics(references, hypotheses)
    print(metrics_dict)

    (P, R, Fs), hashname = score(
        hypotheses,
        references,
        lang="en",
        return_hash=True,
        model_type="bert-base-uncased",
    )
    print(
        f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}"
    )

    ##########################################
    # Logging objects
    ##########################################

    # metrics
    logs = copy.deepcopy(metrics_dict)
    logs["precision"] = P.mean().numpy()
    logs["recal"] = R.mean().numpy()
    logs["Fscore"] = Fs.mean().numpy()

    model_out = dict(
        hypotheses=hypo,
    )

    logger.on_eval_end(logs, model_params, model_out, data)

    if args.metric == "cider":
        selected_metric_score = metrics_dict["CIDEr"]
        print(round(selected_metric_score, 5))

    elif args.metric == "bert":
        selected_metric_score = Fs.mean().item()
        print(round(selected_metric_score, 5))

    return selected_metric_score, metrics_dict


if __name__ == "__main__":

    t = datetime.datetime.now()
    timestamp = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )
    print("code starts", timestamp)

    speak_p = parse_args("speak")
    print(speak_p)

    model_type = speak_p.model_type

    # for reproducibility
    seed = speak_p.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ###################################
    ##  DATA
    ###################################

    print("Loading the vocab...")
    vocab = Vocab(speak_p.vocab_file)
    vocab.index2word[len(vocab)] = "<nohs>"  # special token placeholder for no prev utt
    vocab.word2index["<nohs>"] = len(vocab)  # len(vocab) updated (depends on w2i)

    training_loader, test_loader, val_loader, training_beam_loader = get_dataloaders(
        speak_p, vocab
    )

    max_len = 30  # for beam search

    img_dim = 2048

    embedding_dim = speak_p.embedding_dim
    hidden_dim = speak_p.hidden_dim
    att_dim = speak_p.attention_dim

    dropout_prob = speak_p.dropout_prob
    beam_size = speak_p.beam_size

    metric = speak_p.metric
    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    shuffle = speak_p.shuffle
    normalize = speak_p.normalize

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if speak_p.debug or speak_p.subset_size != -1:
        tags = ["debug"]

    logger = SpeakerLogger(
        vocab=vocab,
        opts=vars(speak_p),
        train_logging_step=20,
        val_logging_step=1,
        resume=speak_p.resume_train != "",
    )

    ###################################
    ##  MODEL
    ###################################

    # depending on the selected model type, we will have a different architecture
    if model_type == "hist_att":  # attention over prev utterance

        model = SpeakerModelHistAtt(
            vocab, embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim, speak_p.device
        ).to(speak_p.device)

    ###################################
    ##  LOSS
    ###################################

    reduction_method = speak_p.reduction
    criterion = nn.CrossEntropyLoss(
        reduction=reduction_method, ignore_index=0
    )  # reduction

    learning_rate = speak_p.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ###################################
    ##  RESTORE MODEL
    ###################################

    if speak_p.resume_train != "":
        checkpoint, file = load_wandb_checkpoint(speak_p.resume_train, speak_p.device)
        # logger.run.restore(file)

        model.load_state_dict(checkpoint["model_state_dict"])
        speaker_model = model.to(speak_p.device)
        epoch = checkpoint["epoch"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed run at epoch {epoch}")

    logger.watch_model([model])

    ###################################
    ##  TRAIN STARTS
    ###################################

    patience = 10  # when to stop if there is no improvement

    if metric == "cider":

        es = EarlyStopping(patience, operator.ge)
    elif metric == "bert":
        es = EarlyStopping(patience, operator.ge)
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    t = datetime.datetime.now()
    timestamp_tr = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    print("training starts", timestamp_tr)

    for epoch in range(speak_p.epochs):

        print("Epoch", epoch)

        losses = []

        model.train()
        torch.enable_grad()

        for i, data in rich.progress.track(
                enumerate(training_loader),
                total=len(training_loader),
                description=f"Train epoch {epoch}",
        ):
            # load infos from datapoint
            utterances_text_ids = data["utterance"]
            prev_utterance_ids = data["prev_utterance"]
            prev_lengths = data["prev_length"]
            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            target_img_feats = data["target_img_feats"]
            lengths = data["length"]
            targets = data["target"]
            prev_hist = data["prev_histories"]
            prev_hist_lens = data["prev_history_lengths"]


            max_length_tensor = prev_utterance_ids.shape[1]
            masks = mask_attn(prev_lengths, max_length_tensor, speak_p.device)


            out = model(
                utterance=utterances_text_ids,
                lengths=lengths,
                prev_utterance=prev_utterance_ids,
                prev_utt_lengths=prev_lengths,
                visual_context_sep=context_separate,
                visual_context=context_concat,
                target_img_feats=target_img_feats,
                targets=targets,
                prev_hist=prev_hist,
                prev_hist_len=prev_hist_lens,
                normalize=normalize,
                masks=masks,
            )

            model.zero_grad()

            """ https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398/2
            ptrblck Nov '18
            Try to permute your output and target so that the batch dimension is in dim0,
            i.e. your output should be [1, number_of_classes, seq_length], while your target 
            should be [1, seq_length]."""

            # out is [batch_size, seq_length, number of classes]
            out = out.permute(0, 2, 1)
            # out is now [batch_size, number_of_classes, seq_length]

            # utterances_text_ids is already [batch_size, seq_length]
            # except SOS: 1:
            target_utterances_text_ids = utterances_text_ids[:, 1:]

            loss = criterion(out, target_utterances_text_ids)

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            ##################
            # logging
            ##################
            aux = dict(out=out, target_utt_ids=utterances_text_ids)
            logger.on_batch_end(
                loss,
                data_point=data,
                modality="train",
                batch_id=i,
                aux=aux,
            )

        print(
            "Train loss", round(np.sum(losses), 3)
        )  # sum all the batches for this epoch

        logger.log_datapoint(data, preds=out, modality="train")

        # evaluation
        with torch.no_grad():
            model.eval()

            isValidation = True
            isTest = False
            print("\nEVALUATION\n")

            current_score, metrics_dict = eval_beam_histatt(
                split_data_loader=val_loader,
                model=model,
                args=speak_p,
                beam_size=beam_size,
                max_len=max_len,
                nlgeval_obj=nlge,
                logger=logger,
            )

            save_model(
                model=model,
                model_type=model_type,
                epoch=epoch,
                accuracy=current_score,
                optimizer=optimizer,
                args=speak_p,
                timestamp=timestamp,
                logger=logger,
            )

            ################################################################
            # Early stopping
            ################################################################
            # check for early stopping
            if es.should_stop(current_score): break

        logger.on_train_end({}, epoch_id=epoch)