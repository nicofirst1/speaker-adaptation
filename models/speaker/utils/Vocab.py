import csv

class Vocab():

    def __init__(self, file):
        print("Initialising vocab from file.")

        self.word2index = {}
        self.index2word = {}
        self.word2count = {}

        for t in ['<pad>', '<unk>', '<sos>', '<eos>']:
            self.index2word[len(self.word2index)] = t
            self.word2index[t] = len(self.word2index)

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                w, c = row[0], int(row[1])
                self.word2index[w] = len(self.word2index)
                self.index2word[self.word2index[w]] = w
                self.word2count[w] = c

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, q):
        if isinstance(q, str):
            return self.word2index.get(q, self.word2index['<unk>'])
        elif isinstance(q, int):
            return self.index2word.get(q, '<unk>')
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))
