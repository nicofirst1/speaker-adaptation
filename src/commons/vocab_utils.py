from typing import Literal

import torch
import pickle

from src.data.dataloaders import Vocab


def mask_oov_embeds(current_embeds: torch.nn.Embedding, full_vocab: Vocab, domain: str,
                    replace_token: Literal["none", "zero", "unk"], data_path:str) -> torch.nn.Embedding:


    if replace_token == "none":
        return current_embeds

    domain_vocab = []

    # get domain specific vocab
    file=f"{data_path}/chains-domain-specific/{domain}//train_ids_utterances.pickle"
    with open(file, 'rb') as f:
        domain_utts = pickle.load(f)

    # extract utts
    for utt in domain_utts:
        utt_str = domain_utts[utt]['utterance']
        domain_vocab.extend(utt_str)

    # find difference between domain and full vocab
    domain_vocab = set(domain_vocab)
    domain_oov_set = set(full_vocab.index2word.keys()) - domain_vocab

    # pad unk sos eos
    domain_oov_set = domain_oov_set - {0, 1, 2, 3}

    unk_i = full_vocab.word2index['<unk>']

    # get replacement based on replace_token
    replacement=current_embeds.weight[unk_i] if replace_token == "unk" else torch.zeros(current_embeds.weight.shape[1])

    # mask or zero out oovs
    for w in domain_oov_set:
        current_embeds.weight[w]=replacement


    return current_embeds
