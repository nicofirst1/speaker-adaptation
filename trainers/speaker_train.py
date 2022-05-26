import os
import sys
from os.path import abspath, dirname

import numpy as np
import rich.progress
import torch.utils.data
from nlgeval import NLGEval
from torch import nn, optim

from data.dataloaders import Vocab, get_dataloaders
from evals.speaker_eval import eval_beam_histatt
from models.speaker.model_speaker_hist_att import SpeakerModelHistAtt
from trainers.parsers import parse_args
from trainers.utils import mask_attn
from wandb_logging import save_model, load_wandb_checkpoint,SpeakerLogger

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

if not os.path.isdir("saved_models"):
    os.mkdir("saved_models")

if not os.path.isdir("speaker_outputs"):
    os.mkdir("speaker_outputs")

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

    # depending on the selected model type, we will have a different architecture

    if model_type == "hist_att":  # attention over prev utterance

        model = SpeakerModelHistAtt(
            vocab, embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim
        ).to(speak_p.device)

    learning_rate = speak_p.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if speak_p.resume_train != "":
        checkpoint, file = load_wandb_checkpoint(speak_p.resume_train, speak_p.device)
        #logger.run.restore(file)

        model.load_state_dict(checkpoint["model_state_dict"])
        speaker_model = model.to(speak_p.device)
        epoch=checkpoint["epoch"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed run at epoch {epoch}")

    logger.watch_model([model])

    reduction_method = speak_p.reduction
    criterion = nn.CrossEntropyLoss(
        reduction=reduction_method, ignore_index=0
    )  # reduction

    batch_size = speak_p.batch_size

    epochs = speak_p.epochs
    patience = 50  # when to stop if there is no improvement
    patience_counter = 0

    # best_loss = float('inf')
    best_score = -1

    # prev_loss = float('inf')
    prev_score = -1

    best_epoch = -1

    t = datetime.datetime.now()
    timestamp_tr = (
            str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    print("training starts", timestamp_tr)

    for epoch in range(epochs):

        print("Epoch", epoch)
        print("Train")

        losses = []

        model.train()
        torch.enable_grad()

        count = 0
        logger.on_train_end({}, epoch_id=epoch)

        for i, data in rich.progress.track(
                enumerate(training_loader),
                total=len(training_loader),
                description=f"Train epoch {epoch}",
        ):


            # print(count)
            count += 1

            utterances_text_ids = data["utterance"]
            prev_utterance_ids = data["prev_utterance"]
            prev_lengths = data["prev_length"]

            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            target_img_feats = data["target_img_feats"]

            lengths = data["length"]
            targets = data["target"]  # image target

            max_length_tensor = prev_utterance_ids.shape[1]

            masks = mask_attn(prev_lengths, max_length_tensor, speak_p.device)

            prev_hist = data["prev_histories"]
            prev_hist_lens = data["prev_history_lengths"]

            out = model(
                utterances_text_ids,
                lengths,
                prev_utterance_ids,
                prev_lengths,
                context_separate,
                context_concat,
                target_img_feats,
                targets,
                prev_hist,
                prev_hist_lens,
                normalize,
                masks,
                speak_p.device,
            )

            model.zero_grad()

            # ignoring 0 index in criterion
            #
            # get_predictions(out, utterances_text_ids, vocab)

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
            print("\nVal Eval")

            # THIS IS val EVAL_BEAM
            print("beam")

            if model_type == "hist_att":
                (
                    best_score,
                    current_score,
                    metrics_dict,
                    has_best_score,
                ) = eval_beam_histatt(
                    val_loader,
                    model,
                    speak_p,
                    best_score,
                    beam_size,
                    max_len,
                    vocab,
                    mask_attn,
                    nlge,
                    isValidation,
                    timestamp,
                    isTest,
                    logger,
                )

            ################################################################
            # Early stopping
            ################################################################

            if metric == "cider":

                if has_best_score:  # comes from beam eval
                    # current_score > best_score
                    best_epoch = epoch
                    patience_counter = 0
                    save_model(
                        model=model,
                        model_type=model_type,
                        epoch=best_epoch,
                        accuracy=current_score,
                        optimizer=optimizer,
                        args=speak_p,
                        timestamp=timestamp,
                        logger=logger,

                    )


                else:
                    # best_score >= current_score:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("model ending duration", duration)

                        break

            elif metric == "bert":

                if has_best_score:  # comes from beam eval
                    # current_score > best_score
                    best_epoch = epoch
                    patience_counter = 0

                    save_model(
                        model=model,
                        model_type=model_type,
                        epoch=best_epoch,
                        accuracy=current_score,
                        optimizer=optimizer,
                        args=speak_p,
                        timestamp=timestamp,
                        logger=logger,

                    )


                else:
                    # best_score >= current_score:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("model ending duration", duration)

                        break

            prev_score = current_score  # not using, stopping based on best score

            print(
                "\nBest", round(best_score, 5), "epoch", best_epoch
            )  # , best_loss)  #validset
            print()
