import json
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import wandb

from transformers import BertTokenizer, BertModel

from data.dataloaders.ListenerDataset import get_data_loaders
from data.dataloaders.Vocab import Vocab
from models.listener.model_listener import ListenerModel
from models.speaker.model_speaker_hist_att import SpeakerModelHistAtt
from trainers.utils import mask_attn


def get_bert_outputs(text, model, tokenizer):
    input_tensors = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    # same as adding special tokens then tokenize + convert_tokens_to_ids

    tokenized_text = tokenizer.tokenize('[CLS]' + text + '[SEP]')
    # input tensors the same as tokenizer.convert_tokens_to_ids(tokenized_text)

    # print(input_tensors)

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
    assert tuple(encoded_layers.shape) == (1, input_tensors.shape[1], model.config.hidden_size)
    assert len(tokenized_text) == input_tensors.shape[1]

    return encoded_layers, tokenized_text


def evaluate_trained_model(dataloader, speak_model, list_model, device, model_bert, tokenizer):
    accuracies = []
    ranks = []
    beam_k=5
    max_len=30

    for ii, data in enumerate(dataloader):


        hypo, _= speak_model.generate_hypothesis(data,beam_k,max_len,device)

        utterances_BERT = get_bert_outputs(hypo, model_bert, tokenizer)[0]

        context_separate = data['separate_images']
        context_concat = data['concat_context']

        lengths = [utterances_BERT.shape[1]]
        targets = data['target']

        max_length_tensor = utterances_BERT.shape[1]

        masks = mask_attn(lengths, max_length_tensor, device)

        prev_hist = data['prev_histories']

        out = list_model(utterances_BERT, lengths, context_separate, context_concat, prev_hist, masks, device)

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

    current_accuracy = sum_accuracy / len(0)

    MRR = np.sum([1 / r for r in ranks]) / len(ranks)

    print(int(sum_accuracy), round(current_accuracy, 5), 'MRR', round(MRR, 5))

    return current_accuracy, MRR


def load_wandb_checkpoint(url):
    api = wandb.Api()
    artifact = api.artifact(url)

    datadir = artifact.download()

    files=[f for f in listdir(datadir) if isfile(join(datadir, f))]

    if len(files)>1:
        raise FileExistsError(f"More than one checkpoint found in {datadir}!")

    checkpoint = torch.load(join(datadir,files[0]), map_location=device)

    return checkpoint


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # ALREADY do_lower_case=True

    # Load pre-trained model (weights)
    model_bert = BertModel.from_pretrained('bert-base-uncased')

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model_bert.eval()
    model_bert.to(device)

    speaker_url = 'adaptive-speaker/speaker/epoch_9_speaker:v0'

    speak_check = load_wandb_checkpoint(speaker_url)

    speak_p = speak_check['args']
    print(speak_p)

    # for reproducibility
    seed = speak_p.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(speak_p.vocab_file)
    #vocab = Vocab("/Users/giulia/Desktop/pb_speaker_adaptation/dataset/vocab.csv")
    vocab.index2word[len(vocab)] = "<nohs>"  # special token placeholder for no prev utt
    vocab.word2index["<nohs>"] = len(vocab)  # len(vocab) updated (depends on w2i)

    img_dim = 2048

    speaker_model = SpeakerModelHistAtt(
        vocab, speak_p.embedding_dim, speak_p.hidden_dim, img_dim, speak_p.dropout_prob, speak_p.att_dim
    ).to(speak_p.device)

    speaker_model.load_state_dict(speak_check['model_state_dict'])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    listener_dict = dict(
        indoor="adaptive-speaker/listener/epoch_4_speaker:v1"
    )

    for dom,url in listener_dict.items():

        list_checkpoint=load_wandb_checkpoint(url)

        list_args=list_checkpoint['args']

        model = ListenerModel(
            len(vocab), list_args.embedding_dim, list_args.hidden_dim, img_dim, list_args.att_dim, list_args.dropout_prob
        ).to(device)

        training_loader, test_loader, val_loader = get_data_loaders(list_args, list_args['train_domain'], img_dim)


        list_model.load_state_dict(list_checkpoint['model_state_dict'])
        list_model = list_model.to(device)

        with torch.no_grad():

            list_model.eval()

            print('test')
            current_accuracy, MRR = evaluate_trained_model(test_loader, speaker_model, list_model, device, model_bert, tokenizer)
            print('Accuracy', current_accuracy, 'MRR', MRR)
