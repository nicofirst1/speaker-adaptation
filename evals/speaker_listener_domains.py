import os
from collections import Counter
from typing import Optional

import numpy as np
import rich.progress
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel

from data.dataloaders.Vocab import Vocab
from data.dataloaders.utils import get_dataloaders
from models.listener.model_listener import ListenerModel
from models.speaker.model_speaker_hist_att import SpeakerModelHistAtt
from trainers.parsers import parse_args
from trainers.utils import mask_attn
from wandb_logging import ListenerLogger, WandbLogger, load_wandb_checkpoint


def get_bert_outputs(text, model, tokenizer):
    input_tensors = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    # same as adding special tokens then tokenize + convert_tokens_to_ids

    tokenized_text = tokenizer.tokenize("[CLS]" + text + "[SEP]")
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
    assert tuple(encoded_layers.shape) == (
        1,
        input_tensors.shape[1],
        model.config.hidden_size,
    )
    assert len(tokenized_text) == input_tensors.shape[1]

    return encoded_layers, tokenized_text


def get_domain_accuracy(accuracy, domains, all_domains):
    assert len(accuracy) == len(domains)

    domain_accs = {d: 0 for d in all_domains}
    domain_accs["all"] = 0

    for idx in range(len(domains)):
        if accuracy[idx]:
            dom = domains[idx]
            domain_accs[dom] += 1
            domain_accs["all"] += 1

    c = Counter(domains)

    for k, v in c.items():
        domain_accs[k] /= v

    domain_accs["all"] /= len(accuracy)

    return domain_accs


def evaluate_trained_model(
        dataloader: Dataset,
        list_model: torch.nn.Module,
        device, vocab: Vocab, domain: str,
        logger: WandbLogger,
        speak_model: Optional[torch.nn.Module] = None,
        tokenizer=None,

):
    accuracies = []
    ranks = []
    domains = []
    beam_k = 5
    max_len = 30
    fake_loss = torch.as_tensor([0])
    in_domain = domain == dataloader.dataset.domain

    if in_domain:
        modality = "in_domain"
    else:
        modality = "out_domain"

    if speak_model is None:
        modality += "_golden"
    else:
        modality += "_generated"

    for ii, data in rich.progress.track(
            enumerate(dataloader),
            total=len(dataloader),
            description=f"Eval on domain '{domain}' with '{modality}' modality",
    ):

        if speak_model is not None:

            # generate hypo with speaker
            hypo, _ = speak_model.generate_hypothesis(data, beam_k, max_len, device)

            # encode with list vocab
            utterance = tokenizer.tokenize(hypo)

            if any(["#" in t for t in utterance]):
                # idk why byt surfboard is tokenized as 'surf' '##board' that raise an error, so skip
                continue

            utterance = vocab.encode(utterance, add_special_tokens=True)
            utterance = utterance.unsqueeze(dim=0)
        else:
            utterance = data['utterance']
            hypo = data['origin_caption']

        # get datapoints
        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        lengths = [utterance.shape[1]]
        targets = data["target"]
        prev_hist = data["prev_histories"]

        max_length_tensor = utterance.shape[1]
        masks = mask_attn(lengths, max_length_tensor, device)

        # get listener output
        out = list_model(
            utterance, context_separate, context_concat, prev_hist, masks, device
        )

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

        aux = dict(
            preds=preds.squeeze(),
            ranks=ranks,
            scores_ranked=scores_ranked,
            images_ranked=images_ranked,
            correct=correct / preds.shape[0],
            hypo=hypo
        )

        logger.on_batch_end(fake_loss, data, aux, batch_id=ii, modality=modality)
        domains += data['domain']

    current_accuracy = np.mean(accuracies)

    # normalize based on batches
    domain_accuracy = get_domain_accuracy(accuracies, domains, logger.domains)
    #domain_accuracy = {k: v / len(accuracies) for k, v in domain_accuracy.items()}
    accuracy = np.mean(accuracies)
    MRR = np.sum([1 / r for r in ranks]) / len(ranks)
    metrics = dict(mrr=MRR, domain_accuracy=domain_accuracy, loss=fake_loss, accuracy=accuracy)

    logger.on_eval_end(metrics, list_domain=dataloader.dataset.domain, modality=modality)

    return current_accuracy, MRR


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    common_args = parse_args("speak")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # ALREADY do_lower_case=True

    # Load pre-trained model (weights)
    model_bert = BertModel.from_pretrained("bert-base-uncased")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model_bert.eval()
    model_bert.to(device)

    speaker_url = "adaptive-speaker/speaker/SpeakerModelHistAtt:v6"

    speak_check,_ = load_wandb_checkpoint(speaker_url, device)

    # load args
    speak_p = speak_check["args"]
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    print(speak_p)

    # for reproducibility
    seed = speak_p.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(speak_p.vocab_file)
    vocab.index2word[len(vocab)] = "<nohs>"  # special token placeholder for no prev utt
    vocab.word2index["<nohs>"] = len(vocab)  # len(vocab) updated (depends on w2i)

    img_dim = 2048

    # init speak model and load state
    speaker_model = SpeakerModelHistAtt(
        vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    # listener dict
    listener_dict = dict(
        all="adaptive-speaker/listener/ListenerModel_all:v20",
        appliances="adaptive-speaker/listener/ListenerModel_appliances:v20",
        food="adaptive-speaker/listener/ListenerModel_food:v20",
        indoor="adaptive-speaker/listener/ListenerModel_indoor:v20",
        outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v20",
        vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v20",
    )

    # for every listener
    for dom, url in listener_dict.items():
        list_checkpoint,_ = load_wandb_checkpoint(url, device)
        list_args = list_checkpoint["args"]

        # update list args
        list_args.batch_size = 1 # hypotesis generation does not support batch
        list_args.vocab_file = "vocab.csv"
        list_args.vectors_file = os.path.basename(list_args.vectors_file)
        list_args.device=device

        # for debug
        #list_args.subset_size = 10

        # update paths
        #list_args.__parse_args()
        list_args.__post_init__()
        vocab = Vocab(list_args.vocab_file)

        list_model = ListenerModel(
            len(vocab),
            list_args.embed_dim,
            list_args.hidden_dim,
            img_dim,
            list_args.attention_dim,
            list_args.dropout_prob,
        ).to(device)

        list_model.load_state_dict(list_checkpoint["model_state_dict"])
        list_model = list_model.to(device)

        # add debug label
        tags = []
        if list_args.debug or list_args.subset_size != -1:
            tags = ["debug"]

        logger = ListenerLogger(
            vocab=vocab,
            opts=vars(list_args),
            group=list_args.train_domain,
            train_logging_step=1,
            val_logging_step=1,
            tags=tags,
            project="speaker-list-dom"
        )

        with torch.no_grad():
            list_model.eval()

            print(f"Eval on '{list_args.train_domain}' domain")
            _, _, val_loader, _ = get_dataloaders(
                list_args, vocab, list_args.train_domain
            )

            evaluate_trained_model(
                dataloader=val_loader, speak_model=speaker_model, list_model=list_model, device=device, vocab=vocab,
                tokenizer=tokenizer, domain=dom, logger=logger
            )

            print(f"Eval on '{list_args.train_domain}' domain with golden caption ")
            evaluate_trained_model(
                dataloader=val_loader, list_model=list_model, device=device, vocab=vocab,
                domain=dom, logger=logger
            )

            print(f"Eval on 'all' domain")
            _, _, val_loader, _ = get_dataloaders(
                list_args, vocab, "all"
            )

            evaluate_trained_model(
                dataloader=val_loader, speak_model=speaker_model, list_model=list_model, device=device, vocab=vocab,
                tokenizer=tokenizer, domain=dom, logger=logger
            )

            print(f"Eval on 'all' domain with golden caption")
            evaluate_trained_model(
                dataloader=val_loader, list_model=list_model, device=device, vocab=vocab,
                domain=dom, logger=logger
            )

        logger.wandb_close()
