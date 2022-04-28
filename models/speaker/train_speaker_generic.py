import torch
import numpy as np

from torch import nn
from torch import optim
import torch.utils.data

from models.model_speaker_hist_att import SpeakerModelHistAtt

from evals import eval_beam_base, eval_beam_histatt

from nlgeval import NLGEval

import os
import argparse

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.SpeakerDataset import SpeakerDataset
from utils.Vocab import Vocab
import datetime

if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')

if not os.path.isdir('speaker_outputs'):
    os.mkdir('speaker_outputs')


def print_predictions(predicted, expected, vocab):

    selected_tokens = torch.argmax(predicted, dim=2)

    for b in range(selected_tokens.shape[0]):

        # reference
        reference = expected[b].data

        reference_string = ''

        for r in range(len(reference)):

            reference_string += vocab.index2word[reference[r].item()]

            if r < len(reference) - 1:
                reference_string += ' '

        print('***REF***', reference_string)

        generation = selected_tokens[b].data

        generation_string = ''

        for g in range(len(generation)):

            generation_string += vocab.index2word[generation[g].item()]

            if g < len(generation) - 1:
                generation_string += ' '

        print('***GEN***', generation_string)


def mask_attn(actual_num_tokens, max_num_tokens, device):

    masks = []

    for n in range(len(actual_num_tokens)):

        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def save_model(model, model_type, epoch, accuracy, optimizer, args, metric, timestamp, seed, t):

    file_name = 'saved_models/model_speaker_' + model_type + '_' + str(seed) + '_' + metric + '_' + timestamp + '.pkl'

    print(file_name)

    duration = datetime.datetime.now() - t

    print('model saving duration', duration)

    torch.save({
        'accuracy': accuracy,
        'args': args, # more detailed info, metric, model_type etc
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../../data/speaker")
    parser.add_argument("-utterances_file", type=str, default="ids_utterances.pickle")
    parser.add_argument("-chains_file", type=str, default="text_chains.json")
    parser.add_argument('-orig_ref_file', type=str, default='text_utterances.pickle')
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-model_type", type=str, default='hist_att')
    parser.add_argument("-subset_size", type=int, default=-1)  # -1 is the full dataset, if you put 10, it will only use 10 chains
    parser.add_argument("-shuffle", action='store_true')
    parser.add_argument("-normalize", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-embedding_dim", type=int, default=512)
    parser.add_argument("-hidden_dim", type=int, default=512)
    parser.add_argument("-attention_dim", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-print", action='store_true')
    parser.add_argument("-metric", type=str, default='cider')  # some metric or loss
    parser.add_argument("-dropout_prob", type=float, default=0.0)
    parser.add_argument("-reduction", type=str, default='sum')  # reduction for crossentropy loss
    parser.add_argument("-beam_size", type=int, default=5)

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)
    print('code starts', timestamp)

    args = parser.parse_args()

    print(args)

    model_type = args.model_type

    # for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file))
    vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
    vocab.word2index['<nohs>'] = len(vocab)  # len(vocab) updated (depends on w2i)

    trainset = SpeakerDataset(
        data_dir=args.data_path,
        utterances_file='train_' + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file='train_' + args.chains_file,
        orig_ref_file='train_' + args.orig_ref_file,
        split='train',
        subset_size=args.subset_size
    )

    testset = SpeakerDataset(
        data_dir=args.data_path,
        utterances_file='test_' + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file='test_' + args.chains_file,
        orig_ref_file='test_' + args.orig_ref_file,
        split='test',
        subset_size=args.subset_size
    )

    valset = SpeakerDataset(
        data_dir=args.data_path,
        utterances_file='val_' + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file='val_' + args.chains_file,
        orig_ref_file='val_' + args.orig_ref_file,
        split='val',
        subset_size=args.subset_size
    )

    print('vocab len', len(vocab))
    print('train len', len(trainset), 'longest sentence', trainset.max_len)
    print('test len', len(testset), 'longest sentence', testset.max_len)
    print('val len', len(valset), 'longest sentence', valset.max_len)

    max_len = 30 # for beam search

    img_dim = 2048

    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    att_dim = args.attention_dim

    dropout_prob = args.dropout_prob
    beam_size = args.beam_size

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metric = args.metric
    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    shuffle = args.shuffle
    normalize = args.normalize
    breaking = args.breaking

    print_gen = args.print

    # depending on the selected model type, we will have a different architecture

    if model_type == 'hist_att':  # attention over prev utterance

        model = SpeakerModelHistAtt(len(vocab),embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim).to(device)

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reduction_method= args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method, ignore_index=0) # reduction

    batch_size = args.batch_size

    load_params = {'batch_size':batch_size, 'shuffle': args.shuffle,
                   'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<sos>'], vocab['<eos>'], vocab['<nohs>'])}

    load_params_test = {'batch_size': 1, 'shuffle': False,
                        'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<sos>'], vocab['<eos>'], vocab['<nohs>'])}

    training_loader = torch.utils.data.DataLoader(trainset, **load_params)
    training_beam_loader = torch.utils.data.DataLoader(trainset, **load_params_test)

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

    epochs = 100
    patience = 50 # when to stop if there is no improvement
    patience_counter = 0

    #best_loss = float('inf')
    best_score = -1

    #prev_loss = float('inf')
    prev_score = -1

    best_epoch = -1

    t = datetime.datetime.now()
    timestamp_tr = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

    print('training starts', timestamp_tr)

    for epoch in range(epochs):

        print('Epoch', epoch)
        print('Train')

        losses = []

        model.train()
        torch.enable_grad()

        count = 0

        for i, data in enumerate(training_loader):

            if i % 200 == 0:
                print(i)

            if breaking and count == 5:
                break

            #print(count)
            count += 1

            utterances_text_ids = data['utterance']
            prev_utterance_ids = data['prev_utterance']
            prev_lengths = data['prev_length']

            context_separate = data['separate_images']
            context_concat = data['concat_context']
            target_img_feats = data['target_img_feats']

            lengths = data['length']
            targets = data['target']  # image target

            max_length_tensor = prev_utterance_ids.shape[1]

            masks = mask_attn(prev_lengths, max_length_tensor, device)

            prev_hist = data['prev_histories']
            prev_hist_lens = data['prev_history_lengths']

            out = model(utterances_text_ids, lengths, prev_utterance_ids, prev_lengths, context_separate, context_concat,
                        target_img_feats, targets, prev_hist, prev_hist_lens, normalize, masks, device)

            model.zero_grad()

            # ignoring 0 index in criterion
            #
            if print_gen:

                print_predictions(out, utterances_text_ids, vocab)

            ''' https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398/2
            ptrblck Nov '18
            Try to permute your output and target so that the batch dimension is in dim0,
            i.e. your output should be [1, number_of_classes, seq_length], while your target 
            should be [1, seq_length].'''

            # out is [batch_size, seq_length, number of classes]
            out = out.permute(0,2,1)
            # out is now [batch_size, number_of_classes, seq_length]

            # utterances_text_ids is already [batch_size, seq_length]
            # except SOS: 1:
            target_utterances_text_ids = utterances_text_ids[:,1:]

            loss = criterion(out, target_utterances_text_ids)

            losses.append(loss.item())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        print('Train loss', round(np.sum(losses), 3))  # sum all the batches for this epoch

        #evaluation
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
            print('\nVal Eval')

            # THIS IS val EVAL_BEAM
            print('beam')

            if model_type == 'base':
                best_score, current_score, metrics_dict, has_best_score = \
                    eval_beam_base(val_loader, model, args, best_score, print_gen, device,
                                   beam_size, max_len, vocab, nlge, isValidation, timestamp, isTest)
                #
                # best_score_topk, current_score_topk, metrics_dict_topk, has_best_score_topk = \
                #     eval_top_k_top_p_base(val_loader, model, args, best_score, print_gen, device,
                #                     40, 0.9, 'topk', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40
                #
                # best_score_topp, current_score_topp, metrics_dict_topp, has_best_score_topp =\
                #     eval_top_k_top_p_base(val_loader, model, args, best_score, print_gen, device,
                #                     40, 0.9, 'topp', max_len, vocab, nlge, isValidation, timestamp)  # k_size 40

            elif model_type == 'hist_att':
                best_score, current_score, metrics_dict, has_best_score = \
                eval_beam_histatt(val_loader, model, args, best_score, print_gen, device,
                                  beam_size, max_len, vocab, mask_attn, nlge, isValidation, timestamp, isTest)
                #
                # best_score_topk, current_score_topk, metrics_dict_topk, has_best_score_topk = \
                #     eval_top_k_top_p_histatt(val_loader, model, args, best_score, print_gen, device,
                #                     40, 0.9, 'topk', max_len, vocab, nlge, isValidation, timestamp)  # k_size 10
                #
                # best_score_topp, current_score_topp, metrics_dict_topp, has_best_score_topp =\
                #     eval_top_k_top_p_histatt(val_loader, model, args, best_score, print_gen, device,
                #                     40, 0.9, 'topp', max_len, vocab, nlge, isValidation, timestamp)  # k_size 10

            if metric == 'cider':

                if has_best_score:  # comes from beam eval
                    # current_score > best_score
                    best_epoch = epoch
                    patience_counter = 0

                    save_model(model, model_type, epoch, current_score, optimizer, args, metric, timestamp,
                               seed, t)

                else:
                    #best_score >= current_score:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print('model ending duration', duration)

                        break

            elif metric == 'bert':

                if has_best_score:  # comes from beam eval
                    # current_score > best_score
                    best_epoch = epoch
                    patience_counter = 0

                    save_model(model, model_type, epoch, current_score, optimizer, args, metric, timestamp,
                               seed, t)

                else:
                    # best_score >= current_score:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print('model ending duration', duration)

                        break

            prev_score = current_score # not using, stopping based on best score

            print('\nBest', round(best_score,5), 'epoch', best_epoch)  # , best_loss)  #validset
            print()

