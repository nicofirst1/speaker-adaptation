import datetime
from typing import Dict, Tuple

import lovely_tensors as lt
import numpy as np
import rich.progress
import torch
import wandb
from PIL import Image
from PIL import ImageDraw
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.commons import (
    SPEAKER_CHK,
    EarlyStopping,
    load_wandb_checkpoint,
    merge_dict,
    parse_args,
    save_model,
    set_seed,
    get_simulator_check,
)
from src.commons.Translator import Translator
from src.commons.model_utils import get_mask_from_utts
from src.data.dataloaders import Vocab
from src.data.dataloaders.FinetuneDataset import FinetuneDataset
from src.models import SpeakerModelEC
from src.models.simulator.SimulatorModel import SimulatorModel
from src.wandb_logging import WandbLogger

global common_p
global sim_vocab


def add_image(aux, sim_acc, logger, max_targets=2):
    idx = 0

    target_ids = np.random.choice(range(len(aux["target_id"])), max_targets)

    table_values = []
    for i in target_ids:
        t_id = aux["target_id"][i]
        utt = aux["utterance"][i]
        la = sim_acc[i]

        jdx = np.random.choice(range(len(t_id)))
        t_id = t_id[jdx]
        utt = utt[jdx]
        la = la[jdx]

        t_id = logger.img_id2path[t_id]

        # read image with PIL
        img = Image.open(t_id)
        color = "green" if la else "red"
        # add a green rectangle to the image if la is 1
        draw = ImageDraw.Draw(img)
        draw.rectangle(((0, 0), (img.width, img.height)), outline=color)

        # convert to wandb.Image
        img_orig = wandb.Image(img, caption=utt)

        # table_values.append([epoch, idx, utt, perb_utt, img, la])

        aux[f"img_{idx}"] = img_orig

        idx += 1


def normalize_aux(aux, logger, epoch, max_targets=2):
    batch_size = len(aux["target_id"][0])
    aux["loss"] = np.mean(aux["loss"])

    sim_acc = aux["sim_acc"]
    aux["sim_acc"] = np.mean([sum(x) for x in aux["sim_acc"]]) / batch_size

    # get max targets random ids in range of targets
    if epoch % 10 == 0:
        add_image(aux, sim_acc, logger, max_targets=max_targets)

    # aux["utt_table"] = wandb.Table(columns=table_columns, data=table_values)

    del aux["target_id"]
    del aux["utterance"]


def get_predictions(
    data: Dict,
    sim_model: SimulatorModel,
    speak_model: SpeakerModelEC,
    loss_f: nn.CrossEntropyLoss,
    translator: Translator,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """

    # get datapoints
    context_separate = data["image_set"]
    target = data["target_index"]
    target_id = data["target_id"]
    target_img_feat = data["target_img_feat"]

    hypos, _, embeds = speak_model.generate_hypothesis(
        context_separate, target_img_feat
    )

    utterance = hypos
    utterance = translator.s2l(utterance)
    dec_utt = translator.list_vocab.batch_decode(utterance)

    # get mask and translate utterance
    masks = get_mask_from_utts(utterance, translator.list_vocab, device=embeds.device)

    # get outputs
    sim_out = sim_model(
        separate_images=context_separate,
        utterance=utterance,
        masks=masks,
        speaker_embeds=embeds,
    )

    sim_out = sim_out.squeeze(-1)

    # Losses and preds
    sim_preds = torch.argmax(sim_out, dim=1)
    sim_acc = sim_preds.eq(target)
    loss = loss_f(sim_out, target).mean()

    aux = dict(
        loss=loss.detach().cpu().item(),
        utterance=dec_utt,
        target_id=target_id,
        sim_acc=sim_acc,
    )

    return loss, aux


def main():
    lt.monkey_patch()

    img_dim = 2048
    global common_p
    global sim_vocab

    common_p = parse_args("sim")
    domain = common_p.train_domain

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and common_p.device != "cpu"
        else torch.device("cpu")
    )

    # for reproducibility
    seed = common_p.seed
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ###################################
    ##  LOGGER
    ###################################

    # add debug label
    tags = []
    if common_p.debug or common_p.subset_size != -1:
        tags = ["debug"]

    speak_vocab = Vocab(parse_args("speak").vocab_file, is_speaker=True)

    logger = WandbLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="sim_finetune",
    )
    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK,
        device,
    )  # datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1]))
    # load args
    speak_p = speak_check["args"]
    speak_p.reset_paths()

    speak_vocab = Vocab(speak_p.vocab_file, is_speaker=True)
    common_speak_p = parse_args("speak")

    # init speak model and load state

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)
    speaker_model = speaker_model.eval()

    ##########################
    # SIMULATOR
    ##########################

    if common_p.force_resume_url == "":
        check = get_simulator_check(common_p.sim_domain, finetune=True)
    else:
        check = common_p.force_resume_url
    sim_check, _ = load_wandb_checkpoint(check, device)

    sim_vocab = Vocab(common_p.vocab_file, is_speaker=False)

    # load args
    sim_p = sim_check["args"]

    # warn the user if the hidden_dim and attention_dim are different
    if sim_p.hidden_dim != common_p.hidden_dim:
        print(
            "WARNING: hidden_dim is different in sim and common, {} vs {}".format(
                sim_p.hidden_dim, common_p.hidden_dim
            )
        )
    if sim_p.attention_dim != common_p.attention_dim:
        print(
            "WARNING: attention_dim is different in sim and common, {} vs {}".format(
                sim_p.attention_dim, common_p.attention_dim
            )
        )

    sim_model = SimulatorModel(
        len(sim_vocab),
        speak_p.hidden_dim,
        sim_p.hidden_dim,
        img_dim,
        sim_p.attention_dim,
        common_p.dropout_prob,
        common_p.sim_domain,
        device,
    ).to(device)

    sim_model.load_state_dict(sim_check["model_state_dict"])

    sim_model = sim_model.to(device)
    sim_model = sim_model.train()

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.AdamW(
        sim_model.parameters(), lr=common_p.learning_rate, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=10,
        factor=0.2,
        verbose=True,
        threshold=0.05,
        threshold_mode="abs",
    )
    loss_f = nn.CrossEntropyLoss(reduction="none")

    ###################################
    ## RESUME AND EARLYSTOPPING
    ###################################

    metric = common_p.metric

    if metric == "loss":
        es = EarlyStopping(common_p.patience, "min")
    elif metric == "accs":
        es = EarlyStopping(common_p.patience, "max")
    else:
        raise ValueError(f"metric of value '{metric}' not recognized")

    logger.watch_model([sim_model], log_freq=1000)

    ###################################
    ##  Get  dataloader
    ###################################

    print("Loading train data...")
    # need batchsize =1 for generating the new dataloaders
    data_domain = common_p.data_domain

    dataset = FinetuneDataset(
        domain=data_domain,
        num_images=common_p.episodes * common_p.batch_size,
        device=device,
        vectors_file=common_p.vectors_file,
        img2dom_file=common_p.img2dom_file,
    )
    dataloader_train = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    dataset = FinetuneDataset(
        domain=data_domain,
        num_images=common_p.episodes * common_p.batch_size,
        device=device,
        vectors_file=common_p.vectors_file,
        img2dom_file=common_p.img2dom_file,
    )
    dataloader_eval = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    translator = Translator(speak_vocab, sim_vocab, device)

    ###################################
    ##  START OF TRAINING LOOP
    ###################################

    t = datetime.datetime.now()
    timestamp = (
        str(t.date()) + "-" + str(t.hour) + "-" + str(t.minute) + "-" + str(t.second)
    )

    for epoch in range(common_p.epochs):
        print("Epoch : ", epoch)

        auxs = []

        sim_model.train()
        sim_model.freeze_utts_stream()

        ###################################
        ##  TRAIN LOOP
        ###################################

        for data in rich.progress.track(
            dataloader_train,
            total=common_p.episodes,
            description=f"Training epoch {epoch}",
        ):
            optimizer.zero_grad()

            # get datapoints
            loss, aux = get_predictions(
                data, sim_model, speaker_model, loss_f, translator
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            # nn.utils.clip_grad_value_(sim_model.parameters(), clip_value=1.0)
            optimizer.step()

        aux = merge_dict(auxs)
        aux["lr"] = optimizer.param_groups[0]["lr"]

        normalize_aux(aux, logger, epoch)

        logger.on_eval_end(aux, list_domain=data_domain, modality="train")

        print(f"Train loss {aux['loss']:.6f}, accuracy {aux['sim_acc'] * 100:.2f}% ")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            sim_model.eval()

            for data in rich.progress.track(
                dataloader_eval,
                total=len(dataloader_eval),
                description=f"evaluating...",
            ):
                loss, aux = get_predictions(
                    data, sim_model, speaker_model, loss_f, translator
                )

                auxs.append(aux)

            aux = merge_dict(auxs)

            normalize_aux(aux, logger, epoch)

            eval_accuracy, eval_loss = aux["sim_acc"], aux["loss"]

            scheduler.step(eval_accuracy)

            print(
                f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% "
            )
            logger.on_eval_end(aux, list_domain=data_domain, modality="eval")

        if common_p.sweep_file == "" and epoch > 0 and epoch % 7 == 0:
            save_model(
                model=sim_model,
                model_type="sim_ec",
                epoch=epoch,
                accuracy=eval_accuracy,
                optimizer=optimizer,
                args=common_p,
                timestamp=timestamp,
                logger=logger,
                loss=eval_loss,
            )

        # check for early stopping
        metric_val = eval_loss if common_p.metric == "loss" else eval_accuracy
        if es.should_stop(metric_val):
            break

        logger.on_train_end({}, epoch_id=epoch)
        print("\n\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, finishing run")
        wandb.finish()
