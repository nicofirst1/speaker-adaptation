import torch
from transformers import HfArgumentParser


def get_predictions(predicted, expected, vocab, return_str=False):
    selected_tokens = torch.argmax(predicted, dim=2)

    for b in range(selected_tokens.shape[0]):

        # reference
        reference = expected[b].data

        reference_string = ""

        for r in range(len(reference)):

            reference_string += vocab.index2word[reference[r].item()]

            if r < len(reference) - 1:
                reference_string += " "

        # print('***REF***', reference_string)

        generation = selected_tokens[b].data

        generation_string = ""

        for g in range(len(generation)):

            generation_string += vocab.index2word[generation[g].item()]

            if g < len(generation) - 1:
                generation_string += " "

        # print('***GEN***', generation_string)

        return reference_string, generation_string


def mask_attn(actual_num_tokens, max_num_tokens, device):
    masks = []

    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (
                max_num_tokens - actual_num_tokens[n]
        )

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks

