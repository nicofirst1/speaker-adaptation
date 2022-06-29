import csv
import re
from typing import List

import torch


class Vocab:
    def __init__(self, file, is_speaker: bool):
        print("Initialising vocab from file.")

        self.word2index = {}
        self.index2word = {}
        self.word2count = {}

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

        text = [t for t in text if t]
        encoded = [self.word2index[t] for t in text]

        if add_special_tokens:
            encoded.append(self.word2index["<eos>"])
            encoded.insert(
                0,
                self.word2index["<sos>"],
            )

        encoded = torch.as_tensor(encoded)

        return encoded

    def decode(self, encoded_ids: torch.Tensor) -> str:

        decodes = " ".join([self.index2word[t.item()] for t in encoded_ids])
        # for i in range(encoded_ids.shape[0]):
        #     batch=encoded_ids[i]
        #     decodes.append(" ".join([self.index2word[t.item()] for t in batch]))

        rg = r" <[a-z]+>"
        decodes = re.sub(rg, "", decodes)
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
