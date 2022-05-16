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

    current_accuracy = sum_accuracy / len(split_dataset)

    MRR = np.sum([1/r for r in ranks])/len(ranks)

    print(int(sum_accuracy), round(current_accuracy, 5), 'MRR', round(MRR, 5))

    return current_accuracy, MRR


if __name__ == '__main__':

    seeds = [1,2,3,4,5]

    for seed in seeds:
        print('seed', seed)

        # for reproducibilty
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model_file_att_ctx_hist = 'saved_models/model_listener_NEW_bert_att_ctx_hist_base_CE_3_accs_2020-05-19-0-38-21.pkl'
        model_file_att_ctx = 'saved_models/model_listener_NEW_bert_att_ctx_base_CE_3_accs_2020-05-19-20-36-46.pkl'

        for model_file in [model_file_att_ctx, model_file_att_ctx_hist]:
            print(model_file)

            checkpoint = torch.load(model_file, map_location=device)

            print(checkpoint['accuracy'])

            args = checkpoint['args']
            print(args)

            model_type = args.model_type
            bert_type = args.bert_type

            if bert_type == 'base':

                utterances_file = 'BERTNEW_utterances.pickle'
                representations_file = 'BERTNEW_representations.pickle'
                chains_file = 'BERTNEW_chains.json'
                embedding_dim = 768

            testset = ListenerDataset(
                data_dir=args.data_path,
                utterances_file='test_' + utterances_file,
                representations_file='test_' + representations_file,
                vectors_file=args.vectors_file,
                chain_file='test_' + chains_file,
                split='test',
                subset_size=args.subset_size
            )

            valset = ListenerDataset(
                data_dir=args.data_path,
                utterances_file='val_' + utterances_file,
                representations_file='val_' + representations_file,
                vectors_file=args.vectors_file,
                chain_file='val_' + chains_file,
                split='val',
                subset_size=args.subset_size
            )
            #
            # print('train len', len(trainset))
            print('test len', len(testset))
            print('val len', len(valset))

            img_dim = 2048
            hidden_dim = args.hidden_dim
            att_dim = args.attention_dim

            dropout_prob = args.dropout_prob

            # depending on the selected model type, we will have a different architecture

            if model_type ==  'bert_att_ctx_hist': # BERT as embeds, vis context given with BERT embeds together, history added to the target side?

                model = ListenerModelBertAttCtxHist(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob) #.to(device)

            elif model_type ==  'bert_att_ctx': # BERT as embeds, vis context given with BERT embeds together, history added to the target side?

                model = ListenerModelBertAttCtx(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob) #.to(device)

            load_params_eval = {'batch_size': 1, 'shuffle': False,
                                'collate_fn': ListenerDataset.get_collate_fn(device)}

            # training_loader = torch.utils.data.DataLoader(trainset, **load_params_eval)

            test_loader = torch.utils.data.DataLoader(testset, **load_params_eval)

            val_loader = torch.utils.data.DataLoader(valset, **load_params_eval)

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            print('val')
            current_accuracy, MRR = evaluate_trained_model(val_loader, valset, model, device)
            print('Accuracy', current_accuracy, 'MRR', MRR)

            print('test')
            current_accuracy, MRR = evaluate_trained_model(test_loader, testset, model, device)
            print('Accuracy', current_accuracy, 'MRR', MRR)

