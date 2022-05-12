import argparse
import datetime
from os.path import join

import torch

from models.speaker.data.SpeakerDataset import SpeakerDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../../data")
    parser.add_argument("-utterances_file", type=str, default="ids_utterances.pickle")
    parser.add_argument("-chains_file", type=str, default="text_chains.json")
    parser.add_argument("-orig_ref_file", type=str, default="text_utterances.pickle")
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-model_type", type=str, default="hist_att")
    parser.add_argument(
        "-subset_size", type=int, default=-1
    ) # -1 is the full dataset, if you put 10, it will only use 10 chains
    parser.add_argument( "-epochs", type=int, default=100)
    parser.add_argument("-shuffle", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-normalize", action="store_true")
    parser.add_argument("-breaking", action="store_true")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-embedding_dim", type=int, default=512)
    parser.add_argument("-hidden_dim", type=int, default=512)
    parser.add_argument("-attention_dim", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-print", action="store_true")
    parser.add_argument("-metric", type=str, default="cider")  # some metric or loss
    parser.add_argument("-dropout_prob", type=float, default=0.0)
    parser.add_argument(
        "-reduction", type=str, default="sum"
    )  # reduction for crossentropy loss
    parser.add_argument("-beam_size", type=int, default=5)
    parser.add_argument("-device")
    args = parser.parse_args()

    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    args.speaker_data = join(args.data_path, "speaker")


    return args


def get_dataloaders(args, vocab):
    trainset = SpeakerDataset(
        data_dir=args.speaker_data,
        utterances_file="train_" + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file="train_" + args.chains_file,
        orig_ref_file="train_" + args.orig_ref_file,
        split="train",
        subset_size=args.subset_size,
    )

    testset = SpeakerDataset(
        data_dir=args.speaker_data,
        utterances_file="test_" + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file="test_" + args.chains_file,
        orig_ref_file="test_" + args.orig_ref_file,
        split="test",
        subset_size=args.subset_size,
    )

    valset = SpeakerDataset(
        data_dir=args.speaker_data,
        utterances_file="val_" + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file="val_" + args.chains_file,
        orig_ref_file="val_" + args.orig_ref_file,
        split="val",
        subset_size=args.subset_size,
    )

    print("vocab len", len(vocab))
    print("train len", len(trainset), "longest sentence", trainset.max_len)
    print("test len", len(testset), "longest sentence", testset.max_len)
    print("val len", len(valset), "longest sentence", valset.max_len)

    load_params = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "collate_fn": SpeakerDataset.get_collate_fn(
            args.device, vocab["<sos>"], vocab["<eos>"], vocab["<nohs>"]
        ),
    }

    load_params_test = {
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": SpeakerDataset.get_collate_fn(
            args.device, vocab["<sos>"], vocab["<eos>"], vocab["<nohs>"]
        ),
    }

    training_loader = torch.utils.data.DataLoader(trainset, **load_params)
    training_beam_loader = torch.utils.data.DataLoader(trainset, **load_params_test)

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

    return training_loader, test_loader, val_loader, training_beam_loader


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

