import torch
import numpy as np

from torch import nn
from torch import optim
import torch.utils.data

from models.model_bert_att_ctx import ListenerModelBertAttCtx
from models.model_bert_att_ctx_hist import ListenerModelBertAttCtxHist

import argparse

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.ListenerDatasetBert import ListenerDataset

import datetime

import os

if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')


def mask_attn(actual_num_tokens, max_num_tokens, device):

    masks = []

    for n in range(len(actual_num_tokens)):

        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])
        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def save_model(model, model_type, epoch, accuracy, loss, mrr, optimizer, args, metric, timestamp, seed, t):

    file_name = 'saved_models/model_listener_NEW_' + model_type + '_' + args.bert_type + '_CE_' + str(seed) + '_' + metric + '_' + timestamp + '.pkl'

    print(file_name)

    duration = datetime.datetime.now() - t

    print('model saving duration', duration)

    torch.save({
        'accuracy': accuracy,
        'mrr': mrr,
        'args': args, # more detailed info, metric, model_type etc
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'seed': seed
    }, file_name)


def evaluate(split_data_loader, split_dataset, breaking, model, isValidation, device):

    losses_eval = []
    accuracies = []
    ranks = []

    count = 0

    for ii, data in enumerate(split_data_loader):
        # print(i)

        if breaking and count == 5:
            break

        count += 1

        utterances_BERT = data['representations']

        context_separate = data['separate_images']
        context_concat = data['concat_context']

        lengths = data['length']
        targets = data['target']

        max_length_tensor = utterances_BERT.shape[1]

        masks = mask_attn(data['length'], max_length_tensor, device)

        prev_hist = data['prev_histories']

        out = model(utterances_BERT, lengths, context_separate, context_concat, prev_hist, masks, device)
        
        loss = criterion(out, targets)
        losses_eval.append(loss.item())

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

    sum_loss = np.sum(losses_eval)
    sum_accuracy = np.sum(accuracies)

    current_accuracy = sum_accuracy/len(split_dataset)

    #avg_rank = np.sum(ranks) / len(split_dataset)

    MRR = np.sum([1/r for r in ranks])/len(ranks)

    print(int(sum_accuracy), 'Acc', round(current_accuracy, 5), 'Loss', round(sum_loss, 5), 'MRR', round(MRR, 5))

    if isValidation:

        return current_accuracy, sum_loss, MRR


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data")
    parser.add_argument("-bert_type", type=str, default='base')  # type of bert embeds to use
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-model_type", type=str, default='bert_att')
    parser.add_argument("-subset_size", type=int, default=-1)  # -1 is the full dataset, if you put 10, it will only use 10 chains
    parser.add_argument("-shuffle", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-attention_dim", type=int, default=512)
    parser.add_argument("-hidden_dim", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-metric", type=str, default='accs')  # accs or loss
    parser.add_argument("-dropout_prob", type=float, default=0.0)
    parser.add_argument("-reduction", type=str, default='sum')  # reduction for crossentropy loss

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)
    print('code starts', timestamp)

    args = parser.parse_args()

    print(args)

    model_type = args.model_type

    # for reproducibilty
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    bert_type = args.bert_type

    if bert_type == 'base':

        utterances_file = 'BERTNEW_utterances.pickle'
        representations_file = 'BERTNEW_representations.pickle'
        chains_file = 'BERTNEW_chains.json'
        embedding_dim = 768

    # elif bert_type == 'base_hid':
    #
    #     utterances_file = 'BERTHID_utterances.json'
    #     representations_file = 'BERTHID_representations.pickle'
    #     chains_file = 'BERTHID_chains.json'
    #     embedding_dim = 768

    elif bert_type == 'large':

        utterances_file = 'LARGEBERT_utterances.pickle'
        representations_file = 'LARGEBERT_representations.pickle'
        chains_file = 'LARGEBERT_chains.json'
        embedding_dim = 1024
    #
    # elif bert_type == 'large_hid':
    #
    #     utterances_file = 'BERTLGHID_utterances.json'
    #     representations_file = 'BERTLGHID_representations.pickle'
    #     chains_file = 'BERTLGHID_chains.json'
    #     embedding_dim = 1024

    trainset = ListenerDataset(
        data_dir=args.data_path,
        utterances_file='train_' + utterances_file,
        representations_file='train_' + representations_file,
        vectors_file=args.vectors_file,
        chain_file='train_' + chains_file,
        split='train',
        subset_size=args.subset_size
    )

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

    print('train len', len(trainset))
    print('test len', len(testset))
    print('val len', len(valset))

    img_dim = 2048
    hidden_dim = args.hidden_dim
    att_dim = args.attention_dim

    dropout_prob = args.dropout_prob

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metric = args.metric
    shuffle = args.shuffle
    breaking = args.breaking

    # depending on the selected model type, we will have a different architecture

    if model_type == 'bert_att_ctx':  # BERT as embeds, vis context given with BERT embeds together, targets visual

        model = ListenerModelBertAttCtx(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob).to(device)

    elif model_type ==  'bert_att_ctx_hist': # BERT as embeds, vis context given with BERT embeds together,
        # history added to the target image side

        model = ListenerModelBertAttCtxHist(embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob).to(device)

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reduction_method= args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)

    batch_size = args.batch_size

    load_params = {'batch_size':batch_size, 'shuffle': shuffle,
                   'collate_fn': ListenerDataset.get_collate_fn(device)}

    load_params_test = {'batch_size': batch_size, 'shuffle': False,
                        'collate_fn': ListenerDataset.get_collate_fn(device)}

    training_loader = torch.utils.data.DataLoader(trainset, **load_params)

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

    epochs = 100
    patience = 50 # when to stop if there is no improvement
    patience_counter = 0

    best_loss = float('inf')
    best_accuracy = -1
    best_mrr = -1

    prev_loss = float('inf')
    prev_accuracy = -1
    prev_mrr = -1

    best_epoch = -1

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

    print('training starts', timestamp)

    for epoch in range(epochs):

        print('Epoch', epoch)
        print('Train')

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases

            trainset = ListenerDataset(
                data_dir=args.data_path,
                utterances_file='train_' + utterances_file,
                representations_file='train_' + representations_file,
                vectors_file=args.vectors_file,
                chain_file='train_' + chains_file,
                split='train',
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

            training_loader = torch.utils.data.DataLoader(trainset, **load_params)

            val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

        losses = []

        model.train()
        torch.enable_grad()

        count = 0

        for i, data in enumerate(training_loader):

            if i % 200 == 0:
                print(i)

            if breaking and count == 5:
                break

            #print(count)
            count += 1

            utterances_BERT = data['representations']

            context_separate = data['separate_images']
            context_concat = data['concat_context']

            lengths = data['length']
            targets = data['target']

            max_length_tensor = utterances_BERT.shape[1]

            masks = mask_attn(data['length'], max_length_tensor, device)

            prev_hist = data['prev_histories']

            out = model(utterances_BERT, lengths, context_separate, context_concat, prev_hist, masks, device)

            model.zero_grad()

            # targets = torch.tensor([[torch.argmax(tg)] for tg in targets]).to(device)
            # TARGETS SUITABLE FOR CROSS-ENTROPY LOSS

            loss = criterion(out, targets)

            losses.append(loss.item())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        print('Train loss sum', round(np.sum(losses), 5))  # sum all the batches for this epoch

        #evaluation
        with torch.no_grad():
            model.eval()

            isValidation = False
            print('\nTrain Eval')
            evaluate(training_loader, trainset, breaking, model, isValidation, device)

            isValidation = True
            print('\nVal Eval')

            current_accuracy, current_loss, current_MRR = evaluate(val_loader, valset, breaking, model, isValidation, device)

            if metric == 'loss':

                if best_loss <= current_loss:

                    patience_counter += 1

                    if patience_counter == patience:

                        duration = datetime.datetime.now() - t

                        print('training ending duration', duration)

                        break
                else:

                    patience_counter = 0

                    best_loss = current_loss
                    best_epoch = epoch

                    save_model(model, model_type, best_epoch, current_accuracy, current_loss, current_MRR, optimizer, args, 'loss',
                               timestamp, seed, t)


                print('patience', patience_counter, '\n')

                print('\nBest', best_epoch, round(best_loss, 5), metric)  # validset
                print()

            elif metric == 'accs':

                if best_accuracy >= current_accuracy:

                    patience_counter += 1

                    if patience_counter == patience:

                        duration = datetime.datetime.now() - t

                        print('training ending duration', duration)

                        break
                else:

                    patience_counter = 0

                    best_accuracy = current_accuracy
                    best_epoch = epoch

                    save_model(model, model_type, best_epoch, current_accuracy, current_loss, current_MRR, optimizer, args, 'accs',
                               timestamp, seed, t)

                print('patience', patience_counter)

                print('\nBest', best_epoch, round(best_accuracy, 5), metric, '\n')  # validset

            elif metric == 'mrr':

                if best_mrr >= current_MRR:

                    patience_counter += 1

                    if patience_counter == patience:

                        duration = datetime.datetime.now() - t

                        print('training ending duration', duration)

                        break
                else:

                    patience_counter = 0

                    best_mrr = current_MRR
                    best_epoch = epoch

                    save_model(model, model_type, best_epoch, current_accuracy, current_loss, current_MRR, optimizer, args, 'mrr',
                               timestamp, seed, t)

                print('patience', patience_counter)

                print('\nBest', best_epoch, round(best_mrr, 5), metric, '\n')  # validset

            prev_accuracy = current_accuracy
            prev_loss = current_loss
            prev_mrr = current_MRR
