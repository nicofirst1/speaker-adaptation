import concurrent
from concurrent.futures import ThreadPoolExecutor

import lovely_tensors as lt
import numpy as np
import torch
from rich.progress import Progress
from torch.utils.data import DataLoader

from src.commons import (
    get_dataloaders,
    LISTENER_CHK_DICT,
    SPEAKER_CHK,
    parse_args,
    SPEAKER_CHK_EC,
)
from src.commons.Translator import Translator
from src.commons.model_utils import get_mask_from_utts, load_wandb_checkpoint, set_seed
from src.data.dataloaders import Vocab
from src.models.listener.ListenerModel import ListenerModel
from src.models.speaker.SpeakerModel import SpeakerModel
from src.models.speaker.SpeakerModelEC import SpeakerModelEC
from src.wandb_logging import WandbLogger


def evaluate_subset(
    progress,
    task_id,
    data_subset,
    speaker_model,
    speaker_model_ec,
    list_model,
    translator,
):
    list_accuracies = []

    list_accuracies_ec = []

    # define modality for wandb panels

    for data in data_subset:
        # generate hypo with speaker
        target_img_feats = data["target_img_feats"]
        context_separate = data["separate_images"]
        context_concat = data["concat_context"]
        prev_utterance = data["prev_utterance"]
        prev_utt_lengths = data["prev_length"]
        targets = data["target"]

        # generate hypo with speaker
        utterance, _, _ = speaker_model.generate_hypothesis(
            prev_utterance, prev_utt_lengths, context_concat, target_img_feats
        )

        utterance_ec, _, _ = speaker_model_ec.generate_hypothesis(
            context_separate, target_img_feats
        )


        utterance = translator.s2l(utterance)
        utterance_ec = translator.s2l(utterance_ec)

        masks = get_mask_from_utts(utterance, translator.list_vocab, device)
        masks_ec = get_mask_from_utts(utterance_ec, translator.list_vocab, device)

        # get listener output
        list_out = list_model(utterance, context_separate, masks)
        list_out = list_out.squeeze(-1)
        list_preds = torch.argmax(list_out, dim=1)
        list_correct = torch.eq(list_preds, targets).float().item()
        list_accuracies.append(list_correct)

        list_out_ec = list_model(utterance_ec, context_separate, masks_ec)
        list_out_ec = list_out_ec.squeeze(-1)
        list_preds_ec = torch.argmax(list_out_ec, dim=1)
        list_correct_ec = torch.eq(list_preds_ec, targets).float().item()
        list_accuracies_ec.append(list_correct_ec)

        progress.update(task_id, advance=1)

    list_accuracy = np.mean(list_accuracies)
    list_accuracy_ec = np.mean(list_accuracies_ec)

    metrics = {}
    metrics["list_accuracy"] = list_accuracy
    metrics["list_accuracy_ec"] = list_accuracy_ec

    metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}

    return metrics


def evaluate(
    data_loader: DataLoader,
    speak_model: SpeakerModel,
    speaker_model_ec: SpeakerModelEC,
    list_model: ListenerModel,
    common_p,
):
    n_threads = 1 if common_p.debug else 6

    # Split the data into subsets
    data_list = list(data_loader)
    data_subsets = np.array_split(data_list, n_threads)

    # Create a shared Progress object
    progress = Progress()
    progress.start()

    # Create tasks for each subset
    tasks = [
        progress.add_task(f"[cyan]Evaluating (Thread {i + 1})", total=len(subset))
        for i, subset in enumerate(data_subsets)
    ]

    # Use ThreadPoolExecutor to process subsets concurrently
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_results = [
            executor.submit(
                evaluate_subset,
                progress,
                tasks[i],
                data_subsets[i],
                speak_model,
                speaker_model_ec,
                list_model,
                translator,
            )
            for i in range(n_threads)
        ]
        results = []
        for future in concurrent.futures.as_completed(future_results):
            results.append(future.result())

    progress.stop()

    ##############################
    # METRICS
    ##############################

    # normalize results
    list_accuracy = [x["list_accuracy"] for x in results]
    list_accuracy_ec = [x["list_accuracy_ec"] for x in results]

    list_accuracy = np.array(list_accuracy).mean()
    list_accuracy_ec = np.array(list_accuracy_ec).mean()

    metrics = dict(
        list_accuracy=list_accuracy,
        list_accuracy_ec=list_accuracy_ec,
    )

    # console.print(metrics)
    logger.on_eval_end(metrics, list_domain=data_loader.dataset.domain, modality="val")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lt.monkey_patch()

    common_args = parse_args("list")
    img_dim = 2048
    dom = common_args.train_domain

    # for reproducibility
    seed = common_args.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    list_vocab = Vocab(common_args.vocab_file, is_speaker=False)

    logger = WandbLogger(
        vocab=list_vocab,
        opts=vars(common_args),
        group=common_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        project="test_finetuned_speaker",
        tags=common_args.tags,
    )

    ####################################
    # ORIGINAL SPEAKER
    ####################################
    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK, device)

    # load args
    speak_p = speak_check["args"]
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    speak_p = speak_check["args"]

    speaker_model = SpeakerModel(
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
        device=device,
        use_beam=speak_p.use_beam,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    speaker_model = speaker_model.eval()

    ####################################
    # Finetuned SPEAKER
    ####################################

    speak_check, _ = load_wandb_checkpoint(SPEAKER_CHK_EC[dom], device)

    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()
    speak_p.__post_init__()

    speak_p = speak_check["args"]

    speaker_model_ec = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        speak_p.sampler_temp,
        speak_p.max_len,
        speak_p.top_k,
        speak_p.top_p,
        device=device,
    ).to(device)

    speaker_model_ec.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model_ec = speaker_model_ec.to(device)
    speaker_model_ec = speaker_model_ec.eval()

    ####################################
    # LISTENER
    ####################################
    url = LISTENER_CHK_DICT[dom]

    list_checkpoint, _ = load_wandb_checkpoint(url, device)
    list_args = list_checkpoint["args"]

    # update paths
    list_args.reset_paths()
    list_args.__post_init__()

    list_model = ListenerModel(
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

    ##########################
    # Logger
    ##########################

    translator = Translator(speak_vocab, list_vocab, device)

    ########################################
    # EVALUATE
    ########################################

    with torch.no_grad():
        list_model.eval()

        ########################
        #  OOD
        ########################
        _, _, val_loader = get_dataloaders(common_args, speak_vocab, "all")

        gen_metrics = evaluate(
            val_loader,
            speaker_model,
            speaker_model_ec,
            list_model,
            common_args,
        )
        print(gen_metrics)

        logger.wandb_close()
