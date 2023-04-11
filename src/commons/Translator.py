from src.commons import speak2list_vocab, translate_utterance


class Translator:


    def __init__(self, speak_vocab, list_vocab, device):
        self.speak_vocab = speak_vocab
        self.list_vocab = list_vocab
        self.device = device

        speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
        self._s2l=translate_utterance(speak2list_v, device)

        list2speak_v = speak2list_vocab(list_vocab, speak_vocab)
        self._l2s=translate_utterance(list2speak_v, device)


    def s2l(self, utterance):
        self._s2l(utterance)

    def l2s(self, utterance):
        self._l2s(utterance)