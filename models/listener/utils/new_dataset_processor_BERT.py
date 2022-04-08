import json
import pickle
from collections import defaultdict

# FROM https://huggingface.co/transformers/quickstart.html
#

import torch
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # ALREADY do_lower_case=True

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()
model.to(device)

def get_bert_outputs(text, model, tokenizer):

    input_tensors = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    # same as adding special tokens then tokenize + convert_tokens_to_ids

    tokenized_text = tokenizer.tokenize('[CLS]' + text + '[SEP]')
    # input tensors the same as tokenizer.convert_tokens_to_ids(tokenized_text)

    #print(input_tensors)

    # just one segment
    segments_ids = [0] * input_tensors.shape[1]
    segments_tensors = torch.tensor([segments_ids])

    input_tensors = input_tensors.to(device)
    segments_tensors = segments_tensors.to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(input_tensors, token_type_ids=segments_tensors)

        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model

        encoded_layers = outputs[0]
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    assert tuple(encoded_layers.shape) == (1,  input_tensors.shape[1], model.config.hidden_size)
    assert len(tokenized_text) == input_tensors.shape[1]

    return encoded_layers, tokenized_text


with open('../data/train.json', 'r') as f:
    train = json.load(f)

with open('../data/val.json', 'r') as f:
    val = json.load(f)

with open('../data/test.json', 'r') as f:
    test = json.load(f)


def process4bert(data, split, model, tokenizer):

    # chain dataset - lengths are going to be different
    chain_dataset = []
    chains_path = '../data/' + split + '_BERTNEW_chains.json'

    chain_count = 0

    utterance_dataset = defaultdict()
    utterances_path = '../data/' + split + '_BERTNEW_utterances.pickle'

    utterance_count = 0

    representation_dataset = defaultdict()
    representations_path = '../data/' + split + '_BERTNEW_representations.pickle'  # for torch tensors

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

                encoded_utt, tokenized_message = get_bert_outputs(message, model, tokenizer)

                representation_dataset[(game_id, round_nr, message_nr)] = encoded_utt

                utt_length = len(tokenized_message)  # including CLS and SEP and wordpiece count

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

                # utterance information
                utterance = {'utterance': tokenized_message, 'image_set': visual_context_ids,
                             'target': [visual_context_ids.index(img_id)], 'length': utt_length, 'game_id': game_id,
                             'round_nr': round_nr, 'message_nr': message_nr}

                utterance_dataset[(game_id, round_nr, message_nr, img_id)] = utterance # add to the full dataset

                utterance_lengths.append(utterance['length'])

                utt_ids.append((game_id, round_nr, message_nr, img_id))
                utterance_count += 1

                if utterance_count % 500 == 0:
                    print(utterance_count)

            # chain information
            chain = {'game_id': game_id, 'chain_id': chain_count, 'utterances': utt_ids, 'target': img_id,
                     'lengths': utterance_lengths}  # utterance lengths

            chain_dataset.append(chain)
            chain_count += 1

    with open(chains_path, 'w') as f:
        json.dump(chain_dataset, f)

    # save the bert texts of words in utterances
    with open(utterances_path, 'wb') as f:
        pickle.dump(utterance_dataset, f)

    # save the bert representations of words in utterances
    with open(representations_path, 'wb') as f:
        pickle.dump(representation_dataset, f)


print('processing train...')
process4bert(train, 'train', model, tokenizer)

print('processing val...')
process4bert(val, 'val', model, tokenizer)

print('processing test...')
process4bert(test, 'test', model, tokenizer)
