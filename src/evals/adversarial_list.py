import copy
import os
from typing import Dict, List, Optional
import torch.functional as F
import numpy as np
import rich.progress
import torch
from torch import nn, cosine_similarity
from torch.utils.data import DataLoader

import wandb
from src.commons import (LISTENER_CHK_DICT, SPEAKER_CHK, get_dataloaders,
                         get_domain_accuracy, hypo2utterance,
                         load_wandb_checkpoint, mask_attn, parse_args, speak2list_vocab)
from src.data.dataloaders import Vocab
from src.models import ListenerModel_hist, SpeakerModel_hist, get_model
from src.trainers.learning_to_stir import translate_utterance
from src.wandb_logging import ListenerLogger, WandbLogger


# FGSM attack code
def fgsm_attack(utt, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_utt = utt - epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_utt = torch.clamp(perturbed_utt, 0, 1)
    # Return the perturbed image
    return perturbed_utt


criterion=nn.CrossEntropyLoss()


def evaluate_trained_model(
    dataloader: DataLoader,
    list_model: torch.nn.Module,
    vocab: Vocab,
    domain: str,
    logger: WandbLogger,
    split: str,
    speak_model: Optional[torch.nn.Module] = None,
):
    accuracies = []
    ranks = []
    domains = []
    in_domain = domain == dataloader.dataset.domain

    original_hypo=[]
    modified_hypo=[]
    iterations=[]
    imgs=[]
    max_iters=10

    # define modality for wandb panels
    modality = split
    if in_domain:
        modality += "/in_domain"
    else:
        modality += "/out_domain"

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
            target_img_feats = data["target_img_feats"]
            prev_utterance = data["prev_utterance"]
            prev_utt_lengths = data["prev_length"]
            visual_context = data["concat_context"]

            # generate hypo with speaker
            utterance, _, _ = speak_model.generate_hypothesis(
                prev_utterance, prev_utt_lengths, visual_context, target_img_feats
            )
            translator(utterance)
            hypo  = [speak_vocab.decode(sent) for sent in utterance][0]

        else:
            # else take them from golden caption
            utterance = data["utterance"]
            hypo = data["orig_utterance"]


        # get datapoints
        origin_utt=hypo[0]
        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        lengths = [utterance.shape[1]]
        targets = data["target"]
        prev_hist = data["prev_histories"]

        max_length_tensor = utterance.shape[1]
        masks = mask_attn(lengths, max_length_tensor, list_model.device)

        # get listener output
        out = list_model(utterance, context_separate, context_concat, prev_hist, masks)


        preds = torch.argmax(out, dim=1)
        correct = torch.eq(preds, targets).float().item()
        loss=999
        itx=1
        iteration=0
        while not correct:

            prev_loss=loss
            loss=criterion(out, targets)

            # Zero all existing gradients
            list_model.zero_grad()

            # Calculate gradients of list_model in backward pass
            loss.backward(retain_graph=True)

            # Collect datagrad
            data_grad = list_model.embeddings.weight.grad[utterance]
            in_data=list_model.embeddings(utterance)

            # Call FGSM Attack
            epsiolon=0.2
            ratio=prev_loss/loss
            if ratio<=1.2:
                epsiolon+=itx/10
                itx+=1
            else:
                itx=1
            perturbed_emb = fgsm_attack(in_data,epsiolon , data_grad)
            perturbed_utts=[]

            # get perturbed utterance
            for idx in range(perturbed_emb.shape[1]):
                embed=perturbed_emb[:,idx].squeeze()
                sim=cosine_similarity(embed, list_model.embeddings.weight,dim=1)
                idx3=torch.argmax(sim).item()
                perturbed_utts.append(idx3)


            utterance=torch.as_tensor(perturbed_utts).unsqueeze(0)
            hypo = [speak_vocab.decode(sent) for sent in utterance][0]

            # get listener output
            out = list_model(utterance, context_separate, context_concat, prev_hist, masks)

            preds = torch.argmax(out, dim=1)
            correct = torch.eq(preds, targets).float().item()
            iteration+=1

            if iteration>max_iters:
                break

        if correct and iteration>0:
            modified_hypo.append(hypo)
            iterations.append(iteration)
            imgs.append(data["image_set"][0][data["target"][0]])
            original_hypo.append(origin_utt)

        accuracies.append(correct)
        domains.append(data["domain"][0])


    accuracy = np.mean(accuracies)

    metrics = {}
    metrics["accuracy"] = accuracy

    # log image\hypo and utterance
    imgs = [logger.img_id2path[str(img) ] for img in imgs]

    imgs = [wandb.Image(img) for img in imgs]

    columns=["image","original hypo","modified hypo","iterations"]

    table=wandb.Table(columns=columns,data=list(zip(imgs,original_hypo,modified_hypo,iterations)))
    metrics['table']=table

    if "out" in modality:
        domain_accuracy = get_domain_accuracy(accuracies, domains, logger.domains)
        metrics["domain_accuracy"] = domain_accuracy

    logger.on_eval_end(
        copy.deepcopy(metrics),
        list_domain=dataloader.dataset.domain,
        modality=modality,
    )



    return metrics


def generate_table_row(
    domain: str, modality: str, table_columns: List, metrics: Dict
) -> List:
    """
    Generate wandb table rows for the log_table function above
    Parameters
    ----------
    domain
    modality
    table_columns
    metrics

    Returns
    -------

    """

    data = [domain, modality]
    for key in table_columns:
        if key in ["modality", "list_domain"]:
            continue
        elif key in metrics.keys():
            data.append(metrics[key])
        elif (
            "domain_accuracy" in metrics.keys()
            and key in metrics["domain_accuracy"].keys()
        ):
            data.append(metrics["domain_accuracy"][key])
        elif key in metrics["aux"].keys():
            data.append(metrics["aux"][key])
        else:
            raise KeyError(f"No key '{key}' found in dict")
    return data


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    common_args = parse_args("list")

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)

    # load args
    speak_p = speak_check["args"]
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    # for reproducibility
    seed = common_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ####################################
    # SPEAKER
    ####################################
    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)

    img_dim = 2048
    model = get_model("speak", speak_p.model_type)
    speaker_model = model(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        speak_p.beam_size,
        speak_p.max_len,
        speak_p.top_k,
        speak_p.top_p,
        device,
        use_beam=speak_p.use_beam,
    )

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ####################################
    # LISTENER
    ####################################
    dom = common_args.train_domain
    url = LISTENER_CHK_DICT[dom]

    list_checkpoint, _ = load_wandb_checkpoint(url, device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.vocab_file = "vocab.csv"
    list_args.vectors_file = os.path.basename(list_args.vectors_file)
    list_args.device = device

    # for debug
    list_args.subset_size = common_args.subset_size
    list_args.debug = common_args.debug
    list_args.test_split = common_args.test_split

    # update paths
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    model = get_model("list", list_args.model_type)
    list_model = model(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device,
    )

    # load from checkpoint
    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)


    logger = ListenerLogger(
        vocab=list_vocab,
        opts=vars(list_args),
        group=list_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        project="adversarial_list",
        tags=common_args.tags,

    )

    speak2list_v = speak2list_vocab(speak_vocab, list_vocab)
    translator=translate_utterance(speak2list_v)

    list_model.eval()


    ########################
    #  EVAL OOD
    ########################
    _, test_loader, val_loader = get_dataloaders(list_args, list_vocab, "all")

    print(f"Eval on 'all' domain with golden caption")

    golden_metrics = evaluate_trained_model(
        dataloader=val_loader,
        list_model=list_model,
        vocab=list_vocab,
        domain=dom,
        logger=logger,
        split="eval",
    )
    print(golden_metrics)

    print(f"Eval on 'all' domain")

    # GENERATED
    gen_metrics = evaluate_trained_model(
        dataloader=val_loader,
        speak_model=speaker_model,
        list_model=list_model,
        vocab=list_vocab,
        domain=dom,
        logger=logger,
        split="eval",
    )
    print(gen_metrics)


    ########################
    #  TEST ODD
    ########################

    print(f"Test on 'all' domain with golden caption")
    golden_metrics = evaluate_trained_model(
        dataloader=test_loader,
        list_model=list_model,
        vocab=list_vocab,
        domain=dom,
        logger=logger,
        split="test",
    )
    print(golden_metrics)

    print(f"Test on 'all' domain")

    gen_metrics = evaluate_trained_model(
        dataloader=test_loader,
        speak_model=speaker_model,
        list_model=list_model,
        vocab=list_vocab,
        domain=dom,
        logger=logger,
        split="test",
    )
    print(gen_metrics)


    logger.wandb_close()

