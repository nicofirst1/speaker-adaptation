import argparse
import glob
import os
import pickle
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from rich.progress import track

from src.data.dataloaders import Vocab


def show_images(
    image_list, data_path, target_idx, captions, domain_words, domains, adapt_step
):
    plt.clf()
    image_list = [x.split("dataset")[1] for x in image_list]
    image_list = [f"{data_path}{x}" for x in image_list]
    gu, ou, au = captions

    od, td = domains
    # show images side by side in one figure, color target in green
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    for i, ax in enumerate(axs.flat):
        im = Image.open(image_list[i])
        ax.imshow(im)
        ax.set_axis_off()
        if i == target_idx:
            ax.set_title(image_list[i], color="green")
        else:
            ax.set_title(image_list[i])

    def color_string(string):
        """
        Color words based on their appearance in the domain words. If od bold, if td italic.
        """

        string = string.split(" ")
        new_string = []
        for word in string:
            if len(word) <=1:
                continue
            if word in domain_words[od]:
                new_string.append(r"$\bf{}$".format(word))
            elif word in domain_words[td]:
                new_string.append(r"$\cal{}$".format(word))
            else:
                new_string.append(word)

        return " ".join(new_string)

    # add gu and ou captions
    fig.text(
        0.5,
        0.09,
        f"Golden utterance: {color_string(gu)}",
        ha="center",
        va="center",
        fontsize=20,
    )
    fig.text(
        0.5,
        0.06,
        f"Original utterance: {color_string(ou)}",
        ha="center",
        va="center",
        fontsize=20,
    )

    # add au caption with bold words from dsws
    fig.text(
        0.5,
        0.03,
        f"Adapted utterance: {color_string(au)}",
        ha="center",
        va="center",
        fontsize=20,
    )

    # at the top add od -> td
    fig.text(
        0.5,
        0.95,
        r"$\cal{}$ -> $\bf{}$ (step {})".format(td,od,adapt_step),
        ha="center",
        va="center",
        fontsize=20,
    )

    return fig


def get_domain_words(domains: List[str], data_path: str, list_vocab) -> Dict[str, str]:
    # get domain specific vocab
    domain_vocabs = {}
    for d in domains:
        domain_vocab = []
        file = f"{data_path}/chains-domain-specific/{d}/train_ids_utterances.pickle"
        with open(file, "rb") as f:
            domain_utts = pickle.load(f)

        for utt in domain_utts:
            utt_str = domain_utts[utt]["utterance"]
            domain_vocab.extend(utt_str)

        domain_vocabs[d] = set(domain_vocab)

    # difference all domain vocabs
    new_domain_vocab = {}
    for d in domains:
        d_vocab = domain_vocabs[d]

        # get all other vocabs
        other_vocabs = [domain_vocabs[d2] for d2 in domains if d2 != d]
        other_vocabs = set.union(*other_vocabs)

        new_domain_vocab[d] = d_vocab.difference(other_vocabs)

    # translate from utt to words
    new_domain_vocab = {
        d: [list_vocab.index2word[x] for x in v] for d, v in new_domain_vocab.items()
    }

    return new_domain_vocab


def main(data_path, csv_file):
    # get  domain words
    domains = {"outdoor", "indoor", "appliances", "vehicles", "food"}

    vocab_file = os.path.join(data_path, "vocab.csv")
    list_vocab = Vocab(vocab_file, is_speaker=False)

    domain_words = get_domain_words(domains, data_path, list_vocab)

    # open csv with pandas
    df = pd.read_csv(csv_file)

    list_domain = df["listener domain"][0]

    # make dirs
    os.makedirs(f"plots/{list_domain}", exist_ok=True)

    # drop all rows where golden/original acc is not 0
    df = df[df["golden_acc"] == 0]
    df = df[df["original_acc"] == 0]

    # drop all rows where target domain is the same as original domain
    df = df[df["target domain"] != list_domain]

    s_iter = 22

    for i in track(range(s_iter), f"[bold green]Processing {list_domain}..."):
        curr_df = df[df[f"adapted_acc_s{i}"] == 1]

        if len(curr_df) == 0:
            continue

        curr_target_domain = curr_df["target domain"]

        golden_utt = curr_df["golden utt"]
        original_utt = curr_df["original utt"]
        adapted_utt = curr_df[f"adapted utt s{i}"]

        dir_path=f"plots/{list_domain}/"

        # get all the image in the dir
        image_list = glob.glob(dir_path+"*.png")
        image_list = [x.split("plot_row_")[1] for x in image_list]
        image_list = [x.split(".png")[0] for x in image_list]
        image_list = [int(x) for x in image_list]
        max_idx = max(image_list)


        for j in range(len(golden_utt)):
            gu = golden_utt.iloc[j]
            ou = original_utt.iloc[j]
            au = adapted_utt.iloc[j]
            ctd = curr_target_domain.iloc[j]

            # get row idx
            row_idx = curr_df.index[j]

            # skip already treated images
            if row_idx < max_idx:
                continue

            file_name=f"plots/{list_domain}/plot_row_{row_idx}.png"
            # if filename is already there, skip
            if os.path.exists(file_name):
                continue

            au_dsw = [x in domain_words[list_domain] for x in au.split(" ")]
            ou_dsw = [x in domain_words[list_domain] for x in ou.split(" ")]
            gu_dsw = [x in domain_words[list_domain] for x in gu.split(" ")]

            # get images
            row = curr_df.iloc[j]
            target_img = row["target img idx"]
            images = [row[f"img path #{i}"] for i in range(6)]

            # # adaptation has introduced a domain specific word?
            # if any(au_dsw) and not any(ou_dsw) and not any(gu_dsw):
            #
            #     if "^" in au :
            #         # skip this case since it breaks the plt
            #         continue

            fig = show_images(
                images,
                data_path,
                target_img,
                (gu, ou, au),
                domain_words,
                (list_domain, ctd),
                i,
            )

            try:
                # save plot
                fig.savefig(
                    file_name,
                    bbox_inches="tight",
                    pad_inches=0,
                    # lower dpi to make smaller images
                    dpi=100,

                )
            except AttributeError:
                print(f"Error in row {row_idx}")
                continue

            plt.close(fig)


if __name__ == "__main__":
    # read csv path from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/processed/processed.csv")
    parser.add_argument("--data_path", type=str, default="../../dataset")

    args = parser.parse_args()

    csv_paths = [
        "/Users/giulia/Downloads/outdoor.csv",
        "/Users/giulia/Downloads/appliances.csv",
        "/Users/giulia/Downloads/vehicles.csv",
        "/Users/giulia/Downloads/food.csv",
        "/Users/giulia/Downloads/indoor.csv",

    ]

    for csv_path in csv_paths:
        main(args.data_path, csv_path)
