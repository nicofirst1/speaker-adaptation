import torch
import numpy as np

from models.model_speaker_base import SpeakerModelBase
from models.model_speaker_hist_att import SpeakerModelHistAtt

from utils.SpeakerDataset import SpeakerDataset
from utils.Vocab import Vocab

from evals import eval_beam_base, eval_beam_histatt

from nlgeval import NLGEval

import os

import datetime

def mask_attn(actual_num_tokens, max_num_tokens, device):

    masks = []

    for n in range(len(actual_num_tokens)):

        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    speaker_files = ['saved_models/model_speaker_hist_att_42_bert_2020-05-21-15-13-22.pkl',
'saved_models/model_speaker_hist_att_1_bert_2020-05-22-16-40-11.pkl',
'saved_models/model_speaker_hist_att_2_bert_2020-05-22-16-41-12.pkl',
'saved_models/model_speaker_hist_att_3_bert_2020-05-22-16-42-13.pkl',
'saved_models/model_speaker_hist_att_4_bert_2020-05-22-16-43-13.pkl']

    for speaker_file in speaker_files:

        seed = 28

        # for reproducibility
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(speaker_file)

        checkpoint = torch.load(speaker_file, map_location=device)

        args = checkpoint['args']

        model_type = args.model_type

        print("Loading the vocab...")
        vocab = Vocab(os.path.join(args.data_path, args.vocab_file))
        vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
        vocab.word2index['<nohs>'] = len(vocab)  # len(vocab) updated (depends on w2i)

        testset = SpeakerDataset(
            data_dir=args.data_path,
            utterances_file='test_' + args.utterances_file,
            vectors_file=args.vectors_file,
            chain_file='test_' + args.chains_file,
            orig_ref_file='test_' + args.orig_ref_file,
            split='test',
            subset_size=args.subset_size
        )

        print('vocab len', len(vocab))
        print('test len', len(testset), 'longest sentence', testset.max_len)

        max_len = 30  # for beam search

        img_dim = 2048

        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        att_dim = args.attention_dim

        dropout_prob = args.dropout_prob
        beam_size = args.beam_size

        metric = args.metric

        shuffle = args.shuffle
        normalize = args.normalize
        breaking = args.breaking

        print_gen = args.print

        # depending on the selected model type, we will have a different architecture

        if model_type == 'base':  # base speaker

            model = SpeakerModelBase(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob).to(device)

        elif model_type == 'hist_att':  # base speaker + vis context

            model = SpeakerModelHistAtt(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim).to(device)

        batch_size = 1

        load_params_test = {'batch_size': 1, 'shuffle': False,
                            'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<sos>'], vocab['<eos>'],
                                                                        vocab['<nohs>'])}

        test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        with torch.no_grad():
            model.eval()

            isValidation = False
            isTest = True
            print('\nTest Eval')

            # THIS IS test EVAL_BEAM
            print('beam')

            # best_score and timestamp not so necessary here
            best_score = checkpoint['accuracy']  # cider or bert
            t = datetime.datetime.now()
            timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

            if model_type == 'base':
                eval_beam_base(test_loader, model, args, best_score, print_gen, device,
                                   beam_size, max_len, vocab, nlge, isValidation, timestamp, isTest)

            elif model_type == 'hist_att':
                eval_beam_histatt(test_loader, model, args, best_score, print_gen, device,
                                      beam_size, max_len, vocab, mask_attn, nlge, isValidation, timestamp, isTest)
