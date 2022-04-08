import json
import pickle

from nltk import TweetTokenizer
from collections import Counter, defaultdict
import csv

from Vocab import Vocab

tweet_tokenizer = TweetTokenizer(preserve_case=False)
min_freq = 0

full_vocab = []
vocab_csv_path = '../../../data/vocab.csv'


def process_data(data, domain_path, split):

    chain_dataset = []
    utterance_dataset = defaultdict()

    chains_path = domain_path + '/' + split + '_text_chains.json'
    utterances_path = domain_path + '/' + split + '_text_utterances.pickle'

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

                full_vocab.extend(tokenized_message)

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

                utt_length = len(tokenized_message)
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


domains = ['indoor', 'outdoor', 'food', 'vehicles', 'appliances', 'speaker']

for domain in domains:

    print(domain)

    domain_path = '../../../data/chains-domain-specific/{}/'.format(domain)

    with open('{}train.json'.format(domain_path), 'r') as f:
        train = json.load(f)

    with open('{}val.json'.format(domain_path), 'r') as f:
        val = json.load(f)

    with open('{}test.json'.format(domain_path), 'r') as f:
        test = json.load(f)

    with open('{}test_seen.json'.format(domain_path), 'r') as f:
        test_seen = json.load(f)

    with open('{}test_unseen.json'.format(domain_path), 'r') as f:
        test_unseen = json.load(f)

    print(len(train))
    print(len(val))
    print(len(test))

    print('processing train...')
    process_data(train, domain_path, 'train')

    print('processing val...')
    process_data(val, domain_path, 'val')

    print('processing test...')
    process_data(test, domain_path, 'test')

    print('processing test seen...')
    process_data(test_seen, domain_path, 'test_seen')

    print('processing test unseen...')
    process_data(test_unseen, domain_path, 'test_unseen')

vocab_ordered = Counter(full_vocab).most_common()

truncated_word_list = []

for word, freq in vocab_ordered:

    if freq < min_freq:
        break
    else:
        truncated_word_list.append((word, freq))

with open(vocab_csv_path, "w") as f:

    writer = csv.writer(f, delimiter=',', quotechar='|')
    writer.writerows(truncated_word_list)

vocab = Vocab(vocab_csv_path)

# Convert from text to IDs
print('converting test to IDs')


def convert2indices(dataset, vocab, split, id_path):

    for tup in dataset:

        utt = dataset[tup]

        text = utt['utterance']

        ids = [vocab[t] for t in text] # + [vocab['<eos>']]

        utt['utterance'] = ids  # length was already +2 so ne need to add again

    new_file_name = id_path + '/' + split + '_ids_utterances.pickle'

    with open(new_file_name, 'wb') as f:
        pickle.dump(dataset, f)


for domain in domains:

    print(domain)

    domain_path = '../../../data/chains-domain-specific/{}/'.format(domain)

    with open('{}train_text_utterances.pickle'.format(domain_path), 'rb') as f:
        train_utterances = pickle.load(f)

    with open('{}val_text_utterances.pickle'.format(domain_path), 'rb') as f:
        val_utterances = pickle.load(f)

    with open('{}test_text_utterances.pickle'.format(domain_path), 'rb') as f:
        test_utterances = pickle.load(f)

    print('converting train...')
    convert2indices(train_utterances, vocab, 'train', domain_path)

    print('converting val...')
    convert2indices(val_utterances, vocab, 'val', domain_path)

    print('converting test...')
    convert2indices(test_utterances, vocab, 'test', domain_path)
