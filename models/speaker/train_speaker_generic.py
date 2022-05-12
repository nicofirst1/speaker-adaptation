import os
import sys
from os.path import abspath, dirname

import numpy as np
import rich.progress
import torch.utils.data
from evals import  eval_beam_histatt
from nlgeval import NLGEval
from torch import nn, optim

from models.model_speaker_hist_att import SpeakerModelHistAtt
from models.speaker.utils import (get_args, get_dataloaders, get_predictions,
                                  mask_attn)
from wandb_logging.SpeakerLogger import SpeakerLogger
from wandb_logging.utils import save_model

sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datetime

from data.Vocab import Vocab

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

    args = get_args()

    print(args)

    model_type = args.model_type

    # for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.speaker_data, args.vocab_file))
    vocab.index2word[len(vocab)] = "<nohs>"  # special token placeholder for no prev utt
    vocab.word2index["<nohs>"] = len(vocab)  # len(vocab) updated (depends on w2i)

    training_loader, test_loader, val_loader, training_beam_loader = get_dataloaders(
        args, vocab
    )

    max_len = 30  # for beam search

    img_dim = 2048

    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    att_dim = args.attention_dim

    dropout_prob = args.dropout_prob
    beam_size = args.beam_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    metric = args.metric
    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    shuffle = args.shuffle
    normalize = args.normalize
    breaking = args.breaking

    print_gen = args.print

    # add debug label
    tags = []
    if args.debug or args.subset_size != -1:
        tags = ["debug"]

    logger = SpeakerLogger(
        vocab=vocab,
        opts=vars(args),
        train_logging_step=20,
        val_logging_step=1,
    )

    # depending on the selected model type, we will have a different architecture

    if model_type == "hist_att":  # attention over prev utterance

        model = SpeakerModelHistAtt(
            len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim
        ).to(device)

    logger.watch_model([model])

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reduction_method = args.reduction
    criterion = nn.CrossEntropyLoss(
        reduction=reduction_method, ignore_index=0
    )  # reduction

    batch_size = args.batch_size

    epochs = args.epochs
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

            if breaking and count == 5:
                break

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

            masks = mask_attn(prev_lengths, max_length_tensor, device)

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
                device,
            )

            model.zero_grad()

            # ignoring 0 index in criterion
            #
            #get_predictions(out, utterances_text_ids, vocab)

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

            # TRAINSET TAKES TOO LONG WITH BEAM
            # isValidation = False
            # print('\nTrain Eval')
            #
            # if model_type == 'base':
            #     eval_beam_base(training_beam_loader, model, args, best_score, print_gen, device,
            #                    beam_size, max_len, vocab, nlge, isValidation, timestamp)
            #     #
            #     # eval_top_k_top_p_base(training_beam_loader, model, args, best_score, print_gen, device,
            #     #                 40, 0.9, 'topk', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40
            #     #
            #     #
            #     # eval_top_k_top_p_base(training_beam_loader, model, args, best_score, print_gen, device,
            #     #                 40, 0.9, 'topp', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40
            #
            # elif model_type == 'hist_att':
            #     eval_beam_histatt(training_beam_loader, model, args, best_score, print_gen, device,
            #                       beam_size, max_len, vocab, mask_attn, nlge, isValidation, timestamp)
            #     #
            #     # eval_top_k_top_p_histatt(training_beam_loader, model, args, best_score, print_gen, device,
            #     #                       40, 0.9, 'topk', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40
            #     #
            #     # eval_top_k_top_p_histatt(training_beam_loader, model, args, best_score, print_gen, device,
            #     #                       40, 0.9, 'topp', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40

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
                    args,
                    best_score,
                    print_gen,
                    device,
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
                        args=args,
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
                        args=args,
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
