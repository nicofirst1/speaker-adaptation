import copy
import pickle
from typing import Literal, List, Dict

import matplotlib
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA

matplotlib.use('GTK3Agg')

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from src.commons import (load_wandb_checkpoint,
                         parse_args,
                         get_listener_check)
from src.data.dataloaders import Vocab

global common_p
tsne_params = dict(perplexity=50, n_components=2, init='pca', n_iter=2500, random_state=23, verbose=1)
umap_params = dict(n_neighbors=5, random_state=42, min_dist=1)

def domain_color_list(df, vocab, colors_dict):
    color_list = []
    domain_list = []
    for word in df.index:
        if word in vocab.keys():
            d = vocab[word]
            color_list.append(colors_dict[d])
            domain_list.append(d)
        else:
            color_list.append('black')
            domain_list.append('common')
    return color_list



def join_clustering(mode,all_embeds):
    if mode == "tsne":
        tsne_model = TSNE(**tsne_params)
        all_embeds = tsne_model.fit_transform(all_embeds)
    elif mode == "pca":
        pca = PCA(n_components=2)
        all_embeds = pca.fit_transform(all_embeds)

    elif mode == "umap":
        umap_model = umap.UMAP(**umap_params)
        all_embeds = umap_model.fit_transform(all_embeds)

    print("fitting done")

    return all_embeds


def split_clustering(mode, all_embeds, shape0):
    main_embeds = all_embeds[:shape0]
    unk_embeds = all_embeds[shape0:2 * shape0]
    zero_embeds = all_embeds[2 * shape0:]

    # fitting for all embeddings
    if mode == "tsne":

        tsne_model = TSNE(**tsne_params)
        main_embeds = tsne_model.fit_transform(main_embeds)

        tsne_model = TSNE(**tsne_params)
        unk_embeds = tsne_model.fit_transform(unk_embeds)

        tsne_model = TSNE(**tsne_params)
        zero_embeds = tsne_model.fit_transform(zero_embeds)

        all_embeds = np.concatenate((main_embeds, unk_embeds, zero_embeds), axis=0)

    elif mode == "pca":
        pca = PCA(n_components=2)

        # split into 3 shape chunk

        main_embeds = pca.fit_transform(main_embeds)

        domain_embeds = pca.transform(unk_embeds)

        common_embeds = pca.transform(zero_embeds)

        all_embeds = np.concatenate((main_embeds, domain_embeds, common_embeds), axis=0)

    elif mode == "umap":
        umap_model = umap.UMAP(**umap_params)
        main_embeds = umap_model.fit_transform(main_embeds)

        umap_model = umap.UMAP(**umap_params)
        unk_embeds = umap_model.fit_transform(unk_embeds)

        umap_model = umap.UMAP(**umap_params)
        zero_embeds = umap_model.fit_transform(zero_embeds)

        all_embeds = np.concatenate((main_embeds, unk_embeds, zero_embeds), axis=0)


    print("fitting done")

    return all_embeds

def clustering(clustering_mode, clustering_type, all_embeds, shape0):

    if clustering_mode == "join":
        all_embeds = join_clustering(clustering_type,all_embeds)
    elif clustering_mode == "split":
        all_embeds = split_clustering(clustering_type, all_embeds, shape0)

    return all_embeds

def plot_embeds(main_df, unk_df, zero_df, domain_words):
    """
    Creates and TSNE model and plots it

    :param main_df: df with all embeddings
    :param unk_df: df with unk embeddings
    :param zero_df: df with zero embeddings
    :param name: name of the path to save the plot
    :param domain_words: list of domain words
    """

    #remove first 3 rows from all dataframes
    to_remove=4
    main_df = main_df.iloc[to_remove:]
    unk_df = unk_df.iloc[to_remove:]
    zero_df = zero_df.iloc[to_remove:]

    all_embeds = np.concatenate((main_df.values, unk_df.values, zero_df.values), axis=0)

    clustering_type = "umap"
    clustering_mode = "join"
    all_embeds=clustering(clustering_mode, clustering_type, all_embeds, main_df.shape[0])

    # split back to main and secondary
    main_embeds = all_embeds[:main_df.shape[0]]
    unk_embeds = all_embeds[main_df.shape[0]:main_df.shape[0] + unk_df.shape[0]]
    zero_embeds = all_embeds[main_df.shape[0] + unk_df.shape[0]:]

    markers = ['o', 'x', 's']
    size = 15
    alpha = 0.4

    # define custom colors for domain words
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'grey', 'black']
    sorted_domains = sorted(domain_words.keys())
    domain_colors = {d: colors[i] for i, d in sorted(enumerate(sorted_domains))}
    domain_colors['common'] = 'black'
    patches = []
    for d, c in domain_colors.items():
        patches.append(matplotlib.patches.Patch(color=c, label=d))

    inverted_vocab = {}
    for d in domain_words.keys():
        for w in domain_words[d]:
            inverted_vocab[w] = d

    # get colors for domain words
    main_colors = domain_color_list(main_df, inverted_vocab, domain_colors)
    unk_colors = domain_color_list(unk_df, inverted_vocab, domain_colors)
    zero_colors = domain_color_list(zero_df, inverted_vocab, domain_colors)

    colors = [main_colors, unk_colors, zero_colors]

    # add marker patches
    patches.append(matplotlib.lines.Line2D([], [], marker=markers[0], label='main', markevery=0))
    patches.append(matplotlib.lines.Line2D([], [], marker=markers[1], label='unk', markevery=0))
    patches.append(matplotlib.lines.Line2D([], [], marker=markers[2], label='zero', markevery=0))

    plt.figure(figsize=(10, 8))
    plt.legend(handles=patches)

    plt.scatter(main_embeds[:, 0], main_embeds[:, 1], c=colors[0], alpha=alpha, s=size, marker=markers[0])
    plt.scatter(unk_embeds[:, 0], unk_embeds[:, 1], c=colors[1], alpha=alpha, s=size, marker=markers[1])
    plt.scatter(zero_embeds[:, 0], zero_embeds[:, 1], c=colors[2], alpha=alpha, s=size, marker=markers[2])

    plt.title("all embeddings")
    plt.savefig(f"{clustering_mode}_all_embeddings_{clustering_type}.png")
    plt.clf()

    # plot for main

    # remove last thre patches
    patches = patches[:-3]

    plt.figure(figsize=(10, 8))
    plt.legend(handles=patches)

    plt.scatter(main_embeds[:, 0], main_embeds[:, 1], c=colors[0], alpha=alpha, label="original", s=size,
                marker=markers[0])

    plt.title("Main Embeddings")
    plt.savefig(f"{clustering_mode}_main_embeddings_{clustering_type}.png")
    plt.clf()

    # plot for unk
    plt.figure(figsize=(10, 8))
    plt.legend(handles=patches)

    plt.scatter(unk_embeds[:, 0], unk_embeds[:, 1], c=colors[1], alpha=alpha, label="unk", s=size, marker=markers[1])

    plt.title("UNK Embeddings")
    plt.savefig(f"{clustering_mode}_unk_embeddings_{clustering_type}.png")
    plt.clf()

    # plot for zero
    plt.figure(figsize=(10, 8))
    plt.legend(handles=patches)

    plt.scatter(zero_embeds[:, 0], zero_embeds[:, 1], c=colors[2], alpha=alpha, label="zero", s=size, marker=markers[2])

    plt.title("Zero Embeddings")
    plt.savefig(f"{clustering_mode}_zero_embeddings_{clustering_type}.png")
    plt.clf()


def get_domain_words(domains: List[str], data_path: str, list_vocab) -> Dict[str, str]:
    # get domain specific vocab
    domain_vocabs = {}
    for d in domains:
        domain_vocab = []
        file = f"{data_path}/chains-domain-specific/{d}/train_ids_utterances.pickle"
        with open(file, 'rb') as f:
            domain_utts = pickle.load(f)

        for utt in domain_utts:
            utt_str = domain_utts[utt]['utterance']
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
    new_domain_vocab = {d: [list_vocab.index2word[x] for x in v] for d, v in new_domain_vocab.items()}

    return new_domain_vocab


def mask_oov_embeds_custom(current_embeds: torch.nn.Embedding, full_vocab: Vocab, domain: str,
                           replace_token: Literal["none", "zero", "unk"], data_path: str) -> torch.nn.Embedding:
    if replace_token == "none":
        return current_embeds

    domain_vocab = []

    # todo: maybe add eva/test as utterances

    # get domain specific vocab
    file = f"{data_path}/chains-domain-specific/{domain}/train_ids_utterances.pickle"
    with open(file, 'rb') as f:
        domain_utts = pickle.load(f)

    # extract utts
    for utt in domain_utts:
        utt_str = domain_utts[utt]['utterance']
        domain_vocab.extend(utt_str)

    # find difference between domain and full vocab
    domain_vocab = set(domain_vocab)
    domain_oov_set = set(full_vocab.index2word.keys()) - domain_vocab

    # pad unk sos eos
    domain_oov_set = domain_oov_set - {0, 1, 2, 3}

    unk_i = full_vocab.word2index['<unk>']

    # get replacement based on replace_token
    replacement = current_embeds[unk_i] if replace_token == "unk" else torch.zeros(
        current_embeds.shape[1])

    # mask or zero out oovs
    for w in domain_oov_set:
        current_embeds[w] = replacement

    return current_embeds


def main():
    common_p = parse_args("int")
    domain = common_p.train_domain
    domains = {'outdoor', 'indoor', 'appliances', 'vehicles', 'food'}

    ##########################
    # LISTENER
    ##########################

    list_check = get_listener_check(domain, common_p.golden_data_perc)

    list_checkpoint, _ = load_wandb_checkpoint(
        list_check,
        "cpu",
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.reset_paths()

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    # get embeddings

    list_full_embeddings = list_checkpoint["model_state_dict"]["embeddings.weight"].cpu().numpy()

    list_tokens = list(list_vocab.word2index.keys())

    with torch.no_grad():
        unk_embeddings = mask_oov_embeds_custom(copy.deepcopy(list_full_embeddings), list_vocab, domain,
                                                replace_token="unk", data_path=common_p.data_path)
        zero_embeddings = mask_oov_embeds_custom(copy.deepcopy(list_full_embeddings), list_vocab, domain,
                                                 replace_token="zero", data_path=common_p.data_path)

    # get domain specific words
    domain_words = get_domain_words(domains, common_p.data_path, list_vocab)

    # convert to dataframes
    list_full_embeddings = pd.DataFrame(list_full_embeddings, index=list_tokens)
    unk_embeddings = pd.DataFrame(unk_embeddings, index=list_tokens)
    zero_embeddings = pd.DataFrame(zero_embeddings, index=list_tokens)

    # plot tsne
    plot_embeds(list_full_embeddings, unk_embeddings, zero_embeddings, domain_words)

    # get index of the tokens that are not in the domain
    indx = ((list_full_embeddings - zero_embeddings) != 0).any(axis=1)

    list_full_embeddings = list_full_embeddings[indx]
    unk_embeddings = unk_embeddings[indx]
    zero_embeddings = zero_embeddings[indx]

    # plot tsne
    #plot_embeds(list_full_embeddings, unk_embeddings, zero_embeddings, "domain", domain_words)

    exit()


if __name__ == "__main__":
    main()
