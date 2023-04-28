from typing import Dict

import torch

from src.data.dataloaders import Vocab


def speak2list_vocab(speak_v: Vocab, list_v: Vocab) -> Dict:
    res = {}
    for k, v in speak_v.word2index.items():
        if k in list_v.word2index.keys():
            res[v] = list_v[k]

    return res


def translate_utterance(speak2list_v, device):
    def translate(utterance):
        # copy the utterance
        utterance = utterance.clone()
        for idx in range(len(utterance)):
            new_utt = [speak2list_v[x.item()] for x in utterance[idx]]
            utterance[idx] = torch.as_tensor(new_utt).to(device)
        return utterance

    return translate


class Translator:
    def __init__(self, speak_vocab, list_vocab, device):
        self.speak_vocab = speak_vocab
        self.list_vocab = list_vocab
        self.device = device

        speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
        self._s2l = translate_utterance(speak2list_v, device)

        list2speak_v = speak2list_vocab(list_vocab, speak_vocab)
        self._l2s = translate_utterance(list2speak_v, device)

    def s2l(self, utterance):
        return self._s2l(utterance)

    def l2s(self, utterance):
        return self._l2s(utterance)
