import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_score import score

# beam search
# topk
# topp

# built via modifying https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py



def eval_beam_histatt(
    split_data_loader,
    model,
    args,
    best_score,
    beam_size,
    max_len,
    vocab,
    mask_attn,
    nlgeval_obj,
    isValidation,
    timestamp,
    isTest,
    logger,
):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references=[]
    hypotheses=[]
    device=args.device
    count = 0



    for i, data in enumerate(split_data_loader):
        # print(i)

        completed_sentences = []
        completed_scores = []

        beam_k = beam_size

        if breaking and count == 5:
            break

        count += 1
        ref = data["reference_chain"][0]  # batch size 1  # full set of references for a single instance

        hypo, model_params= model.generate_hypothesis(data,beam_k,max_len,device)
        references.append(ref)
        hypotheses.append(hypo)



    # Calculate scores
    metrics_dict = nlgeval_obj.compute_metrics(references, hypotheses)
    print(metrics_dict)

    (P, R, Fs), hashname = score(
        hypotheses,
        references,
        lang="en",
        return_hash=True,
        model_type="bert-base-uncased",
    )
    print(
        f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}"
    )

    ##########################################
    # Logging objects
    ##########################################

    # metrics
    logs = copy.deepcopy(metrics_dict)
    logs["precision"] = P.mean().numpy()
    logs["recal"] = R.mean().numpy()
    logs["Fscore"] = Fs.mean().numpy()


    model_out = dict(
        hypotheses=hypo,
    )

    logger.on_eval_end(logs, model_params, model_out, data)

    if args.metric == "cider":
        selected_metric_score = metrics_dict["CIDEr"]
        print(round(selected_metric_score, 5))

    elif args.metric == "bert":
        selected_metric_score = Fs.mean().item()
        print(round(selected_metric_score, 5))

    # from https://github.com/Maluuba/nlg-eval
    # where references is a list of lists of ground truth reference text strings and hypothesis is a list of
    # hypothesis text strings. Each inner list in references is one set of references for the hypothesis
    # (a list of single reference strings for each sentence in hypothesis in the same order).


    if isValidation:
        has_best_score = False

        if selected_metric_score > best_score:
            best_score = selected_metric_score
            has_best_score = True



        return best_score, selected_metric_score, metrics_dict, has_best_score


