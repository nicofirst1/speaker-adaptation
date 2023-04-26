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
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.commons import (
    SPEAKER_CHK,
    EarlyStopping,
    load_wandb_checkpoint,
    mask_attn,
    merge_dict,
    parse_args,
    save_model,
    set_seed,
    get_simulator_check,
)
from src.commons.Baseline import MeanBaseline
from src.commons.Translator import Translator
from src.commons.model_utils import logprobs_from_logits
from src.data.dataloaders import Vocab
from src.data.dataloaders.EcDataset import EcDataset
from src.models import SpeakerModelEC
from src.models.simulator.SimulatorModel import SimulatorModel
from src.wandb_logging import ListenerLogger

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
    baseline: MeanBaseline,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """
    global common_p
    global sim_vocab

    # get datapoints
    context_separate = data["image_set"]
    target = data["target_index"]
    image_ids = data["image_ids"]
    target_id = data["target_id"]
    target_img_feat = data["target_img_feat"]

    hypos, _, embeds = speak_model.generate_hypothesis(
        context_separate, target_img_feat
    )

    utterance = hypos
    translator.s2l(utterance)
    dec_utt = sim_vocab.batch_decode(utterance)

    lengths = torch.tensor([len(x) for x in utterance])
    max_length_tensor = torch.max(lengths).item()
    # get mask and translate utterance
    masks = mask_attn(lengths, max_length_tensor, sim_model.device)

    # get outputs
    sim_out = sim_model(
        utterance,
        context_separate,
        masks,
        speaker_embeds=embeds,
    )
    sim_out = sim_out.squeeze(-1)

    # Losses and preds
    sim_preds = torch.argmax(sim_out, dim=1)
    sim_acc = sim_preds.eq(target)
    sim_loss = loss_f(sim_out, target).mean()


    loss = sim_loss

    baseline.update(sim_loss.detach())

    aux = dict(
        loss=loss.detach().cpu().item(),
        utterance=dec_utt,
        target_id=target_id,
        sim_acc=sim_acc,
    )

    return loss, aux


def evaluate(
    data_loader: DataLoader,
    speak_model: SpeakerModelEC,
    sim_model: SimulatorModel,
    translator,
    baseline: MeanBaseline,
    loss_f: torch.nn.Module,
    split: str,
) -> Dict:
    """
    Evaluate model on either in/out_domain dataloader
    :param data_loader:
    :param model:
    :param in_domain: when out_domain also estimate per domain accuracy
    :return:
    """

    auxs = []

    for data in rich.progress.track(
        data_loader,
        total=len(data_loader),
        description=f"evaluating '{split}' split...",
    ):
        loss, aux = get_predictions(
            data, sim_model, speak_model, loss_f, translator, baseline
        )

        auxs.append(aux)

    aux = merge_dict(auxs)

    return aux


def get_kwargs(split, common_p):
    kwargs = {
        "device": common_p.device,
        "episodes": common_p.episodes,
        "domain": common_p.train_domain,
        "utterances_file": f"{split}_{common_p.utterances_file}",
        "vectors_file": common_p.vectors_file,
        "chain_file": f"{split}_{common_p.chains_file}",
        "orig_ref_file": f"{split}_{common_p.orig_ref_file}",
        "split": split,
        "subset_size": common_p.subset_size,
        "image_size": common_p.image_size,
        "img2dom_file": common_p.img2dom_file,
        "data_dir": common_p.data_path,
        "batch_size": common_p.batch_size,
    }
    return kwargs


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

    logger = ListenerLogger(
        vocab=speak_vocab,
        opts=vars(common_p),
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="ec_sim_finetune",
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
    common_p.train_domain = domain
    common_p.device = device

    # override common_p with sim_p
    common_p.hidden_dim = sim_p.hidden_dim
    common_p.attention_dim = sim_p.attention_dim
    common_p.dropout_prob = sim_p.dropout_prob

    sim_model = SimulatorModel(
        len(sim_vocab),
        speak_p.hidden_dim,
        common_p.hidden_dim,
        img_dim,
        common_p.attention_dim,
        common_p.dropout_prob,
        common_p.sim_domain,
        common_p.device,
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

    logger.watch_model([sim_model, speaker_model], log_freq=1000)

    ###################################
    ##  Get  dataloader
    ###################################

    print("Loading train data...")
    # need batchsize =1 for generating the new dataloaders
    data_domain = common_p.data_domain

    kwargs = get_kwargs("train", common_p)
    dataset = EcDataset(**kwargs)
    dataloader_train = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )
    print("...Done.\nLoading eval data...")

    kwargs = get_kwargs("val", common_p)
    dataset = EcDataset(**kwargs)
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
        # dataloader_train.dataset.randomize_data()
        # dataloader_eval.dataset.randomize_data()

        # torch.enable_grad()
        ###################################
        ##  TRAIN LOOP
        ###################################
        baseline = MeanBaseline()

        for data in rich.progress.track(
            dataloader_train,
            total=len(dataloader_train),
            description=f"Training epoch {epoch}",
        ):
            optimizer.zero_grad()

            # get datapoints
            loss, aux = get_predictions(
                data, sim_model, speaker_model, loss_f, translator, baseline
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            # nn.utils.clip_grad_value_(speaker_model.parameters(), clip_value=1.0)
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

            print(f"\nEvaluation")
            aux = evaluate(
                dataloader_eval,
                speaker_model,
                sim_model,
                translator,
                baseline,
                loss_f,
                split="eval",
            )
            normalize_aux(aux, logger, epoch)

            eval_accuracy, eval_loss = aux["sim_acc"], aux["loss"]

            scheduler.step(eval_accuracy)

            print(
                f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% "
            )
            logger.on_eval_end(aux, list_domain=data_domain, modality="eval")

        if (
            common_p.sweep_file == ""
            and epoch > 0
            and epoch % (common_p.epochs // 20) == 0
        ):
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
