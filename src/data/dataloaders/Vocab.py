import csv
import re
from typing import List

import torch
from nltk import TweetTokenizer


class Vocab:
    def __init__(self, file, is_speaker: bool):
        print("Initialising vocab from file.")

        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.tokenizer = TweetTokenizer(preserve_case=False)


        for t in ["<pad>", "<unk>", "<sos>", "<eos>"]:
            self.index2word[len(self.word2index)] = t
            self.word2index[t] = len(self.word2index)

        with open(file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in reader:
                w, c = row[0], int(row[1])
                self.word2index[w] = len(self.word2index)
                self.index2word[self.word2index[w]] = w
                self.word2count[w] = c

        if is_speaker:
            self.index2word[
                len(self)
            ] = "<nohs>"  # special token placeholder for no prev utt
            self.word2index["<nohs>"] = len(self)  # len(vocab) updated (depends on w2i)


    def encode(self, text: List[str], add_special_tokens=False) -> torch.Tensor:
        """
        Encode a list of strings to a list of indices.
        Parameters
        ----------
        text : List[str] - list of strings to encode
        add_special_tokens : bool - whether to add special tokens to the beginning and end of the sequence

        Returns - torch.Tensor - tensor of indices
        -------
        """

        # remove empty strings
        text = [t for t in text if t]

        encoded=[]
        # tokenize and convert to indices
        for sent in text:
            tok= self.tokenizer.tokenize(sent)
            tok=[self.word2index[t] for t in tok]
            if add_special_tokens:
                tok = [self.word2index["<sos>"]] + tok + [self.word2index["<eos>"]]
            tok=torch.as_tensor(tok)
            encoded.append(tok)

        # add padding
        max_len = max([len(e) for e in encoded])
        padded = torch.zeros(len(encoded), max_len, dtype=torch.long)

        for i, e in enumerate(encoded):
            padded[i, : len(e)] = e


        return padded

    def batch_decode(self, encoded_ids: torch.Tensor) -> List[str]:

        batch_size = encoded_ids.shape[0]

        return [self.decode(encoded_ids[i]) for i in range(batch_size)]

    def decode(self, encoded_ids: torch.Tensor) -> str:


        if len(encoded_ids.shape) == 0:
            encoded_ids = encoded_ids.unsqueeze(0)

        decodes = " ".join([self.index2word[t.item()] for t in encoded_ids])
        rg = r"<[a-z]+>"
        decodes = re.sub(rg, "", decodes).strip()
        return decodes

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, q):
        if isinstance(q, str):
            return self.word2index.get(q, self.word2index["<unk>"])
        elif isinstance(q, int):
            return self.index2word.get(q, "<unk>")
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))
