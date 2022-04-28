import torch
import numpy as np
import json

from listener.models.model_bert_att_ctx_hist import ListenerModelBertAttCtxHist

from listener.utils.ListenerDatasetBert import ListenerDataset

from transformers import BertTokenizer, BertModel


def mask_attn(actual_num_tokens, max_num_tokens, device):

    masks = []

    for n in range(len(actual_num_tokens)):

        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])
        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


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


def evaluate_trained_model(split_data_loader, split_dataset, model, device, hyps, model_bert, tokenizer):

    accuracies = []
    ranks = []

    for ii, data in enumerate(split_data_loader):

        if ii % 500 == 0:
            print(ii)

        hypothesis = ' '.join([w for w in hyps[ii].split() if w != '<eos>'])
        utterances_BERT = get_bert_outputs(hypothesis, model_bert, tokenizer)[0]

        context_separate = data['separate_images']
        context_concat = data['concat_context']

        lengths = [utterances_BERT.shape[1]]
        targets = data['target']

        max_length_tensor = utterances_BERT.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        prev_hist = data['prev_histories']

        out = model(utterances_BERT, lengths, context_separate, context_concat, prev_hist, masks, device)

        preds = torch.argmax(out, dim=1)

        correct = torch.eq(preds, targets).sum()
        accuracies.append(float(correct))

        scores_ranked, images_ranked = torch.sort(out.squeeze(), descending=True)

        if out.shape[0] > 1:
            for s in range(out.shape[0]):
                # WARNING - assumes batch size > 1
                rank_target = images_ranked[s].tolist().index(targets[s].item())
                ranks.append(rank_target + 1)  # no 0

        else:
            rank_target = images_ranked.tolist().index(targets.item())
            ranks.append(rank_target + 1)  # no 0

    sum_accuracy = np.sum(accuracies)

    current_accuracy = sum_accuracy / len(split_dataset)

    MRR = np.sum([1/r for r in ranks])/len(ranks)

    print(int(sum_accuracy), round(current_accuracy, 5), 'MRR', round(MRR, 5))

    return current_accuracy, MRR


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # ALREADY do_lower_case=True

    # Load pre-trained model (weights)
    model_bert = BertModel.from_pretrained('bert-base-uncased')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model_bert.eval()
    model_bert.to(device)

    model_file = 'listener/saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_3_accs_2020-05-19-0-38-21.pkl'
    print(model_file)

    checkpoint = torch.load(model_file, map_location=device)

    print(checkpoint['accuracy'])

    args = checkpoint['args']
    print(args)

    model_type = args.model_type
    bert_type = args.bert_type

    all_hyps_files = ['FINAL_SPEAKERS/TEST_JSONS/histatt/hyps_hist_att_bert_test_2020-05-26-21-17-19.json',
                      'FINAL_SPEAKERS/TEST_JSONS/histatt/hyps_hist_att_bert_test_2020-05-26-21-21-18.json',
                      'FINAL_SPEAKERS/TEST_JSONS/histatt/hyps_hist_att_bert_test_2020-05-26-21-24-48.json',
                      'FINAL_SPEAKERS/TEST_JSONS/histatt/hyps_hist_att_bert_test_2020-05-26-21-28-2.json',
                      'FINAL_SPEAKERS/TEST_JSONS/histatt/hyps_hist_att_bert_test_2020-05-26-21-31-21.json']

    for hyps_file_path in all_hyps_files:

        for seed in [28]:

            print('seed', seed)
            # for reproducibilty
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            print(hyps_file_path)

            with open(hyps_file_path, 'r') as f:
                hyps = json.load(f)

            testset = ListenerDataset(
                data_dir='listener/data',
                utterances_file='test_BERTNEW_utterances.pickle',
                representations_file='test_BERTNEW_representations.pickle',
                vectors_file='vectors.json',
                chain_file='test_BERTNEW_chains.json',
                split='test',
                subset_size=-1
            )

            load_params_test = {'batch_size': 1, 'shuffle': False,
                                'collate_fn': ListenerDataset.get_collate_fn(device)}

            test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

            print('test len', len(testset))

            img_dim = 2048
            hidden_dim = args.hidden_dim
            att_dim = args.attention_dim

            dropout_prob = args.dropout_prob

            if bert_type == 'base':

                embedding_dim = 768

            elif bert_type == 'large':

                embedding_dim = 1024

            # depending on the selected model type, we will have a different architecture

            if model_type == 'bert_att_ctx_hist': # BERT as embeds, vis context given with BERT embeds together, history added to the target side?

                model = ListenerModelBertAttCtxHist(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob) #.to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            with torch.no_grad():

                model.eval()

                print('test')
                current_accuracy, MRR = evaluate_trained_model(test_loader, testset, model, device, hyps, model_bert, tokenizer)
                print('Accuracy', current_accuracy, 'MRR', MRR)
