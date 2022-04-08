import torch
import numpy as np

from train_listener_bert_CE_generic import mask_attn

from models.model_bert_att_ctx_hist import ListenerModelBertAttCtxHist
from models.model_bert_att_ctx import ListenerModelBertAttCtx

from utils.ListenerDatasetBert import ListenerDataset


def evaluate_trained_model(split_data_loader, split_dataset, model, device):

    accuracies = []
    ranks = []

    for ii, data in enumerate(split_data_loader):

        # if ii % 500 == 0:
        #     print(ii)

        utterances_BERT = data['representations']

        context_separate = data['separate_images']
        context_concat = data['concat_context']

        lengths = data['length']
        targets = data['target']

        max_length_tensor = utterances_BERT.shape[1]

        masks = mask_attn(data['length'], max_length_tensor, device)

        prev_hist = data['prev_histories']

        out = model(utterances_BERT, lengths, context_separate, context_concat, prev_hist, masks, device)

        preds = torch.argmax(out, dim=1)

        correct = torch.eq(preds, targets).sum()
        accuracies.append(float(correct))

        scores_ranked, images_ranked = torch.sort(out.squeeze().squeeze(), descending=True)

        if out.shape[0] > 1:
            for s in range(out.shape[0]):
                # WARNING - assumes batch size > 1
                rank_target = images_ranked[s].tolist().index(targets[s].item())
                ranks.append(rank_target + 1)  # no 0

        else:
            rank_target = images_ranked.tolist().index(targets.item())
            ranks.append(rank_target + 1)  # no 0

    sum_accuracy = np.sum(accuracies)

    current_accuracy = sum_accuracy / len(accuracies)

    MRR = np.sum([1/r for r in ranks])/len(ranks)

    print(int(sum_accuracy), len(accuracies), round(current_accuracy, 5), 'MRR', round(MRR, 5))

    return current_accuracy, MRR


if __name__ == '__main__':

    seed = 28

    print('seed', seed)

    # for reproducibilty
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bert_type = 'base'

    if bert_type == 'base':
        utterances_file = 'BERTNEW_utterances.pickle'
        representations_file = 'BERTNEW_representations.pickle'
        chains_file = 'BERTNEW_chains.json'
        embedding_dim = 768

    valset = ListenerDataset(
        data_dir='./data',
        utterances_file='val_' + utterances_file,
        representations_file='val_' + representations_file,
        vectors_file='vectors.json',
        chain_file='val_' + chains_file,
        split='val',
        subset_size=-1
    )

    print('val len', len(valset))

    load_params_eval = {'batch_size': 1, 'shuffle': False,
                        'collate_fn': ListenerDataset.get_collate_fn(device)}

    val_loader = torch.utils.data.DataLoader(valset, **load_params_eval)

    model_files = ['saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_42_accs_2020-05-16-14-18-44.pkl',
'saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_1_accs_2020-05-19-0-36-21.pkl',
'saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_2_accs_2020-05-19-0-37-19.pkl',
'saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_3_accs_2020-05-19-0-38-21.pkl',
'saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_4_accs_2020-05-19-0-39-19.pkl']

    for model_file in model_files:
        print(model_file)

        checkpoint = torch.load(model_file, map_location=device)

        print(checkpoint['accuracy'])

        args = checkpoint['args']
        print(args)

        model_type = args.model_type

        img_dim = 2048
        hidden_dim = args.hidden_dim
        att_dim = args.attention_dim

        dropout_prob = args.dropout_prob

        # depending on the selected model type, we will have a different architecture

        if model_type ==  'bert_att_ctx_hist': # BERT as embeds, vis context given with BERT embeds together, history added to the target side?

            model = ListenerModelBertAttCtxHist(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob) #.to(device)

        elif model_type ==  'bert_att_ctx': # BERT as embeds, vis context given with BERT embeds together, history added to the target side?

            model = ListenerModelBertAttCtx(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob) #.to(device)

        else:
            print('WRONG MODEL TYPE')

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print('val')
        current_accuracy, MRR = evaluate_trained_model(val_loader, valset, model, device)
        print('Accuracy', current_accuracy, 'MRR', MRR)

