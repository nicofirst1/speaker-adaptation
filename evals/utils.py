from collections import Counter

from wandb.wandb_torch import torch


def hypo2utterance(hypo,tokenizer, vocab):
    # encode with list vocab
    utterance = tokenizer.tokenize(hypo)

    if any(["#" in t for t in utterance]):
        # idk why byt surfboard is tokenized as 'surf' '##board' that raise an error, so skip
        raise ValueError()

    utterance = vocab.encode(utterance, add_special_tokens=True)
    utterance = utterance.unsqueeze(dim=0)

    return utterance

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
