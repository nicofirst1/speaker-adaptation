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
    get_listener_check,
    load_wandb_checkpoint,
    mask_attn,
    mask_oov_embeds,
    merge_dict,
    parse_args,
    save_model,
    set_seed,
)
from src.commons.Baseline import MeanBaseline
from src.commons.Translator import Translator
from src.commons.model_utils import logprobs_from_logits, get_mask_from_utts, change2random
from src.data.dataloaders import Vocab
from src.data.dataloaders.FinetuneDataset import FinetuneDataset
from src.models import ListenerModel, SpeakerModelEC
from src.wandb_logging import WandbLogger

global common_p
global list_vocab


def add_image(aux, list_acc, logger, max_targets=2):
    idx = 0

    target_ids = np.random.choice(range(len(aux["target_id"])), max_targets)

    table_values = []
    for i in target_ids:
        t_id = aux["target_id"][i]
        utt = aux["utterance"][i]
        la = list_acc[i]

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
    aux["policy_loss"] = np.mean(aux["policy_loss"])
    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["entropy_loss"] = np.mean(aux["entropy_loss"])
    aux["weighted_policy_loss"] = np.mean(aux["weighted_policy_loss"])
    aux["weighted_entropy_loss"] = np.mean(aux["weighted_entropy_loss"])
    aux["weighted_list_loss"] = np.mean(aux["weighted_list_loss"])

    list_acc = aux["list_acc"]
    aux["list_acc"] = np.mean([sum(x) for x in aux["list_acc"]]) / batch_size
    aux["baseline"] = np.mean(aux["baseline"])

    if aux["enc_log_probs"][0].ndim==1:
        enc = torch.stack(aux["enc_log_probs"])
        enc=enc.mean(dim=0)
    else:
        enc = [x.mean(dim=0) for x in aux["enc_log_probs"] if x.size(0) > 0]
        enc = [x for x in enc if x.size(0) > 0]
        enc = torch.stack(enc)

    aux["enc_log_probs"] = enc

    dec = [x.mean(dim=0) for x in aux["dec_log_probs"] if x.size(0) > 0]
    dec = [x.mean(dim=0) if x.ndim > 1  else x for x in dec ]

    try:
        dec = torch.stack(dec).mean(dim=0).numpy()
    except RuntimeError:
        dec = np.zeros_like(aux["enc_log_probs"])

    aux["dec_log_probs"] = dec

    # remove nans from log probs
    aux["enc_log_probs"] = np.nan_to_num(aux["enc_log_probs"])
    aux["dec_log_probs"] = np.nan_to_num(aux["dec_log_probs"])

    # get max targets random ids in range of targets
    if epoch % 10 == 0:
        add_image(aux, list_acc, logger, max_targets=max_targets)

    # aux["utt_table"] = wandb.Table(columns=table_columns, data=table_values)

    del aux["target_id"]
    del aux["utterance"]


def get_predictions(
    data: Dict,
    list_model: ListenerModel,
    speak_model: SpeakerModelEC,
    loss_f: nn.CrossEntropyLoss,
    translator: Translator,
    baseline: MeanBaseline,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """
    global common_p
    global list_vocab

    # get datapoints
    context_separate = data["image_set"]
    target = data["target_index"]
    target_id = data["target_id"]
    target_img_feat = data["target_img_feat"]

    hypos, model_params, _ = speak_model.generate_hypothesis(
        context_separate, target_img_feat
    )

    enc_logits = model_params["encoder_logits"]
    dec_logits = model_params["decoder_logits"]

    utterance = hypos
    translator.s2l(utterance)
    dec_utt = list_vocab.batch_decode(utterance)

    masks = get_mask_from_utts(utterance, translator.list_vocab, device=enc_logits.device)


    # get outputs
    list_out = list_model(utterance, context_separate, masks)
    list_out = list_out.squeeze(-1)

    # Losses and preds
    list_preds = torch.argmax(list_out, dim=1)
    list_acc = list_preds.eq(target)
    list_loss = loss_f(list_out, target).mean()

    bs = baseline.predict(list_loss.detach())
    if common_p.logits_to_use == "enc":
        enc_log_probs = logprobs_from_logits(
            enc_logits, model_params["encoder_ids"].squeeze(dim=0)
        )

        policy_loss = (list_loss.detach() - bs) * enc_log_probs
        distr = Categorical(logits=enc_logits)
    elif common_p.logits_to_use == "dec":
        dec_log_probs = logprobs_from_logits(dec_logits, hypos)

        policy_loss = (list_loss.detach() - bs) * dec_log_probs
        distr = Categorical(logits=dec_log_probs)
    else:
        joint_log_probs = logprobs_from_logits(enc_logits[..., :-1] * dec_logits, hypos)
        policy_loss = (list_loss.detach() - bs) * joint_log_probs
        distr = Categorical(logits=joint_log_probs)

    policy_loss = policy_loss.mean()
    weighted_policy_loss = policy_loss * common_p.policy_loss_weight

    entropy = distr.entropy()
    entropy_loss = -entropy.mean()
    weighted_entropy_loss = entropy_loss * common_p.entropy_loss_weight

    weighted_list_loss = list_loss * common_p.list_loss_weight

    loss = weighted_policy_loss + weighted_entropy_loss + weighted_list_loss

    baseline.update(list_loss.detach())

    aux = dict(
        loss=loss.detach().cpu().item(),
        policy_loss=policy_loss.detach().cpu().item(),
        list_loss=list_loss.detach().cpu().item(),
        entropy_loss=entropy_loss.detach().cpu().item(),
        weighted_policy_loss=weighted_policy_loss.detach().cpu().item(),
        weighted_entropy_loss=weighted_entropy_loss.detach().cpu().item(),
        weighted_list_loss=weighted_list_loss.detach().cpu().item(),
        baseline=bs.detach().cpu().item(),
        utterance=dec_utt,
        target_id=target_id,
        list_acc=list_acc,
        enc_log_probs=torch.log_softmax(enc_logits, dim=-1).detach().cpu().squeeze(),
        dec_log_probs=torch.log_softmax(dec_logits, dim=-1).detach().cpu().squeeze(),
    )

    return loss, aux


def evaluate(
    data_loader: DataLoader,
    speak_model: SpeakerModelEC,
    list_model: ListenerModel,
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
            data, list_model, speak_model, loss_f, translator, baseline
        )

        auxs.append(aux)

    aux = merge_dict(auxs)

    return aux




def main():
    lt.monkey_patch()

    img_dim = 2048
    global common_p
    global list_vocab

    common_p = parse_args("speak")
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
        project="ec_pretrain",
    )

    ##########################
    # LISTENER
    ##########################

    list_check = get_listener_check(domain)

    list_checkpoint, _ = load_wandb_checkpoint(
        list_check,
        device,
    )
    # datadir=join("./artifacts", LISTENER_CHK_DICT[domain].split("/")[-1]))
    list_args = list_checkpoint["args"]

    # update list args
    list_args.device = device
    list_args.reset_paths()

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    list_vocab = Vocab(list_args.vocab_file, is_speaker=False)

    list_model = ListenerModel(
        len(list_vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
        list_args.train_domain,
        device=device,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)
    list_model.eval()

    with torch.no_grad():
        list_model.embeddings = mask_oov_embeds(
            list_model.embeddings,
            list_vocab,
            domain,
            replace_token=list_args.mask_oov_embed,
            data_path=common_p.data_path,
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
    common_p.embedding_dim = speak_p.embedding_dim

    # init speak model and load state

    speaker_model = SpeakerModelEC(
        speak_vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        common_p.dropout_prob,
        speak_p.attention_dim,
        common_speak_p.sampler_temp,
        speak_p.max_len,
        common_speak_p.top_k,
        common_speak_p.top_p,
        device=device,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"], strict=False)
    speaker_model = speaker_model.to(device)

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.AdamW(
        speaker_model.parameters(), lr=common_p.learning_rate, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=7,
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

    logger.watch_model([speaker_model], log_freq=100)

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

    translator = Translator(speak_vocab, list_vocab, device)

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

        speaker_model.train()

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
                data, list_model, speaker_model, loss_f, translator, baseline
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

        print(f"Train loss {aux['loss']:.6f}, accuracy {aux['list_acc'] * 100:.2f}% ")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            speaker_model.eval()

            print(f"\nEvaluation")
            aux = evaluate(
                dataloader_eval,
                speaker_model,
                list_model,
                translator,
                baseline,
                loss_f,
                split="eval",
            )
            normalize_aux(aux, logger, epoch)

            eval_accuracy, eval_loss = aux["list_acc"], aux["loss"]

            scheduler.step(eval_accuracy)

            print(
                f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% "
            )
            logger.on_eval_end(aux, list_domain=data_domain, modality="eval")

        if common_p.sweep_file =="" and epoch > 0 and epoch % 2 == 0:
            save_model(
                model=speaker_model,
                model_type="speaker_ec",
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
