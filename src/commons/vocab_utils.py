import torch
import pickle


def mask_oov_embeds(current_embeds, full_vocab, domain, replace_token):

    domain_vocab = []

    with open('../../dataset/chains-domain-specific/' + domain + '/train_ids_utterances.pickle', 'rb') as f:
        domain_utts = pickle.load(f)

    for utt in domain_utts:
        utt_str = domain_utts[utt]['utterance']
        domain_vocab.extend(utt_str)

    domain_vocab = set(domain_vocab)
    domain_oov_set = set(full_vocab.index2word.keys()) - domain_vocab

    # pad unk sos eos
    domain_oov_set = domain_oov_set - {0, 1, 2, 3}

    unk_i = full_vocab.word2index['<unk>']

    for w in domain_oov_set:
        if replace_token == 'zero':
            current_embeds.weight[w] = torch.zeros_like(current_embeds.weight[0])
        elif replace_token == 'unk':
            current_embeds.weight[w] = current_embeds.weight[unk_i]

    return current_embeds
