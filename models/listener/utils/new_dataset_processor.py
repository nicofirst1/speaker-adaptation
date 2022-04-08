import json
import pickle

from nltk import TweetTokenizer
from collections import Counter, defaultdict
import csv

from Vocab import Vocab

tweet_tokenizer = TweetTokenizer(preserve_case=False)
min_freq = 2


def process_data(data, split, min_freq=2):

    if split == 'train':

        # for the creation of the vocab from the train set
        train_full_vocab = []
        vocab_csv_path = '../data/vocab.csv'

    chain_dataset = []
    utterance_dataset = defaultdict()

    chains_path = '../data/' + split + '_text_chains.json'
    utterances_path = '../data/' + split + '_text_utterances.pickle'

    chain_count = 0
    utterance_count = 0

    for img_file in sorted(data):

        img_id = str(int(img_file.split('/')[1].split('.')[0].split('_')[2]))
        # img path in the form of 'person_bed/COCO_train2014_000000318646.jpg'
        # but also like 'bowl_dining_table/COCO_train2014_000000086285.jpg'

        chains4img = data[img_file]

        for game_id in sorted(chains4img):

            chain_data = chains4img[game_id]

            utt_ids = []  # pointers for the utterances in this chain
            utterance_lengths = []  # lengths of utterances in this chain

            for m in range(len(chain_data)):

                utterance_data = chain_data[m]
                message = utterance_data['Message_Text']
                message_nr = utterance_data['Message_Nr']
                round_nr = utterance_data['Round_Nr']

                tokenized_message = tweet_tokenizer.tokenize(message)

                if split == 'train':
                    train_full_vocab.extend(tokenized_message)

                # - / * AND SO ON punctuation marks are they in the dataset? vocab of bert?
                # INCLUDES * AND STUFF FOR CENSORING

                speaker = utterance_data['Message_Speaker']

                # visual context for the listener is the round images of the person who uttered the message
                if speaker == 'A':

                    visual_context = utterance_data['Round_Images_A']

                elif speaker == 'B':

                    visual_context = utterance_data['Round_Images_B']

                visual_context_ids = []

                for v in visual_context:

                    v_id = str(int(v.split('/')[1].split('.')[0].split('_')[2]))

                    visual_context_ids.append(v_id)

                visual_context_ids = sorted(visual_context_ids)  # SORTED VISUAL CONTEXT

                utt_length = len(tokenized_message) + 2  # WARNING!! ALREADY INCLUDING sos eos into the length
                # utterance information
                utterance = {'utterance': tokenized_message, 'image_set': visual_context_ids,
                             'target': [visual_context_ids.index(img_id)], 'length': utt_length, 'game_id': game_id,
                             'round_nr': round_nr, 'message_nr': message_nr}

                utterance_dataset[(game_id, round_nr, message_nr, img_id)] = utterance # add to the full dataset

                utterance_lengths.append(utt_length)
                utt_ids.append((game_id, round_nr, message_nr, img_id))
                utterance_count += 1

                if utterance_count % 500 == 0:
                    print(utterance_count)

            # chain information
            chain = {'game_id': game_id, 'chain_id': chain_count, 'utterances': utt_ids, 'target': img_id,
                     'lengths': utterance_lengths}  # utterance lengths

            chain_dataset.append(chain)
            chain_count += 1

    # dump the text versions of the chains and utterances

    with open(chains_path, 'w') as f:
        json.dump(chain_dataset, f)

    with open(utterances_path, 'wb') as f:
        pickle.dump(utterance_dataset, f)

    if split == 'train':
        # vocabulary from the train set
        # ordered in terms of frequency (descending)
        vocab_ordered = Counter(train_full_vocab).most_common()

        truncated_word_list = []

        for word, freq in vocab_ordered:

            if freq < min_freq:
                break
            else:
                truncated_word_list.append((word, freq))

        with open(vocab_csv_path, "w") as f:

            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerows(truncated_word_list)


with open('../data/train.json', 'r') as f:
    train = json.load(f)

with open('../data/val.json', 'r') as f:
    val = json.load(f)

with open('../data/test.json', 'r') as f:
    test = json.load(f)

print(len(train))
print(len(val))
print(len(test))

print('processing train...')
process_data(train, 'train', min_freq)

print('processing val...')
process_data(val, 'val')

print('processing test...')
process_data(test, 'test')


# Convert from text to IDs

with open('../data/train_text_utterances.pickle', 'rb') as f:
    train_utterances = pickle.load(f)

with open('../data/val_text_utterances.pickle', 'rb') as f:
    val_utterances = pickle.load(f)

with open('../data/test_text_utterances.pickle', 'rb') as f:
    test_utterances = pickle.load(f)

vocab = Vocab('../data/vocab.csv')


def convert2indices(dataset, vocab, split):

    for tup in dataset:

        utt = dataset[tup]

        text = utt['utterance']

        ids = [vocab['<sos>']] + [vocab[t] for t in text] + [vocab['<eos>']]

        utt['utterance'] = ids  # length was already +2 so ne need to add again

    new_file_name = '../data/' + split + '_ids_utterances.pickle'

    with open(new_file_name, 'wb') as f:
        pickle.dump(dataset, f)


print('converting train...')
convert2indices(train_utterances, vocab, 'train')

print('converting val...')
convert2indices(val_utterances, vocab, 'val')

print('converting test...')
convert2indices(test_utterances, vocab, 'test')
