import os
import json
import pickle
import torch
from collections import defaultdict
from torch.utils.data import Dataset

import numpy as np

class ListenerDataset(Dataset):
    def __init__(self, split, data_dir, chain_file, utterances_file,
                 vectors_file, representations_file, subset_size):

        self.data_dir = data_dir
        self.split = split

        # Load a PhotoBook utterance chain dataset
        with open(os.path.join(self.data_dir, chain_file), 'r') as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue utterance dataset
        with open(os.path.join(self.data_dir, utterances_file), 'rb') as file:
            self.utterances = pickle.load(file)

        # Load BERT representations for the tokens in the utterances
        with open(os.path.join(self.data_dir, representations_file), 'rb') as file:
            self.representations = pickle.load(file)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

        self.img_dim = 2048
        self.img_count = 6  # images in the context

        self.data = dict()

        self.img2chain = defaultdict(dict)

        for chain in self.chains:

            self.img2chain[chain['target']][chain['game_id']] = chain['utterances']

        if subset_size == -1:
            self.subset_size = len(self.chains)
        else:
            self.subset_size = subset_size

        print('processing',self.split)
        for chain in self.chains[:self.subset_size]:

            chain_utterances = chain['utterances']
            game_id = chain['game_id']

            for s in range(len(chain_utterances)):

                prev_chains = defaultdict(list)
                prev_lengths = defaultdict(int)

                utterance_id = tuple(chain_utterances[s]) # utterance_id = (game_id, round_nr, messsage_nr, img_id)
                round_nr = utterance_id[1]
                message_nr = utterance_id[2]

                cur_utterance_obj = self.utterances[utterance_id]
                cur_utterance_text = cur_utterance_obj['utterance']

                cur_utterance_reps = self.representations[(game_id, round_nr, message_nr)].squeeze(dim=0)

                length = cur_utterance_obj['length']

                assert len(cur_utterance_text) != 2 # cls and sep # no empty utt and no empty chain

                images = cur_utterance_obj['image_set']
                target = cur_utterance_obj['target']  # index of correct img

                target_image = images[target[0]]

                images = list(np.random.permutation(images))
                target = [images.index(target_image)]

                context_separate = torch.zeros(self.img_count, self.img_dim)

                im_counter = 0

                for im in images:

                    context_separate[im_counter] = torch.tensor(self.image_features[im])
                    im_counter += 1

                    if game_id in self.img2chain[im]:  # was there a linguistic chain for this image in this game
                        temp_chain = self.img2chain[im][game_id]

                        hist_utterances = []

                        for t in range(len(temp_chain)):

                            _, t_round, t_message, _ = temp_chain[t] #(game_id, round_nr, messsage_nr, img_id)

                            if t_round < round_nr:
                                hist_utterances.append((game_id, t_round, t_message))

                            elif t_round == round_nr:

                                if t_message < message_nr:
                                    hist_utterances.append((game_id, t_round, t_message))

                        if len(hist_utterances) > 0:

                            # ONLY THE MOST RECENT history
                            for hu in [hist_utterances[-1]]:
                                prev_chains[im].extend(self.representations[hu].squeeze(dim=0))

                        else:
                            # no prev reference to that image
                            prev_chains[im] = []

                    else:
                        # image is in the game but never referred to
                        prev_chains[im] = []

                    prev_lengths[im] = len(prev_chains[im])  # shape[0]

                # ALWAYS 6 IMAGES IN THE CONTEXT

                context_concat = context_separate.reshape(self.img_count * self.img_dim)

                self.data[len(self.data)] = {'utterance': cur_utterance_text,
                                             'representations': cur_utterance_reps,
                                             'image_set': images,
                                             'concat_context': context_concat,
                                             'separate_images': context_separate,
                                             'target':target,
                                             'length': length,
                                             'prev_histories': prev_chains,
                                             'prev_history_lengths': prev_lengths
                                             }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            max_src_length = max(d['length'] for d in data)

            batch = defaultdict(list)

            for sample in data:

                for key in data[0].keys():

                    if key == 'utterance':
                        padded = sample['utterance']

                        # I am not using these actually, we can remove them from here

                        # print('utt', padded)

                    elif key == 'representations':

                        pad_rep = torch.zeros(max_src_length-sample['length'],sample['representations'].shape[1])

                        padded = torch.cat((sample['representations'], pad_rep), dim=0)

                        # print('rep', padded)

                    elif key == 'image_set':

                        padded = [int(img) for img in sample['image_set']]

                        # print('img', padded)

                    elif key == 'prev_histories':

                        histories_per_img = []

                        for k in range(len(sample['image_set'])):
                            #keep the order of imgs
                            img_id = sample['image_set'][k]
                            history = sample[key][img_id]

                            histories_per_img.append(history)

                        padded = histories_per_img

                    elif key == 'prev_history_lengths':

                        histlens_per_img = []

                        for k in range(len(sample['image_set'])):
                            #keep the order of imgs
                            img_id = sample['image_set'][k]
                            history_length = sample[key][img_id]

                            histlens_per_img.append(history_length)

                        padded = histlens_per_img

                    else:
                        # length of utterance in number of words
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                if key not in ['prev_histories', 'utterance', 'representations',
                               'separate_images', 'concat_context']:
                    # print(key)
                    batch[key] = torch.Tensor(batch[key]).long().to(device)

                    # for instance targets can be long and sent to device immediately

                elif key in ['representations', 'separate_images', 'concat_context']:

                    batch[key] = torch.stack(batch[key]).to(device)  # float

            return batch

        return collate_fn

