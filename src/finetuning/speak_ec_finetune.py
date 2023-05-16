import copy
import datetime
import gc
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
from torch.utils.data import DataLoader

from src.commons import (
    SPEAKER_CHK,
    EarlyStopping,
    get_listener_check,
    load_wandb_checkpoint,
    mask_oov_embeds,
    merge_dict,
    parse_args,
    save_model,
    set_seed,
)
from src.commons.Baseline import MeanBaseline
from src.commons.Translator import Translator
from src.commons.model_utils import logprobs_from_logits, get_mask_from_utts
from src.data.dataloaders import Vocab
from src.data.dataloaders.FinetuneDataset import FinetuneDataset
from src.models import ListenerModel, SpeakerModelEC
from src.wandb_logging import WandbLogger

global common_p
global list_vocab


def add_image(aux, list_acc, logger, max_targets=2):
    set_seed(42)
    idx = 0

    target_ids = np.random.choice(range(len(aux["target_id"])), max_targets)

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
        draw.rectangle(((0, 0), (img.width, img.height)), outline=color, width=4)

        # convert to wandb.Image
        img_orig = wandb.Image(img, caption=utt)

        # table_values.append([epoch, idx, utt, perb_utt, img, la])

        aux[f"img_{idx}"] = img_orig

        idx += 1


def normalize_aux(aux, logger: WandbLogger, epoch, max_targets=2):
    batch_size = len(aux["target_id"][0])
    aux["loss"] = np.mean(aux["loss"])
    aux["policy_loss"] = np.mean(aux["policy_loss"])
    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["entropy_loss"] = np.mean(aux["entropy_loss"])
    aux["weighted_policy_loss"] = np.mean(aux["weighted_policy_loss"])
    aux["weighted_entropy_loss"] = np.mean(aux["weighted_entropy_loss"])
    aux["weighted_list_loss"] = np.mean(aux["weighted_list_loss"])

    # unfold the list of domains
    domains = [x for y in aux["domains"] for x in y]
    list_accs = [x for y in aux["list_acc"] for x in y]

    curr_domain = logger.opts["train_domain"]
    ood_acc = []
    # get the average accuracy per domain class
    for d in set(domains):
        acc = np.mean([x for x, y in zip(list_accs, domains) if y == d])
        aux[f"list_acc/{d}"] = acc
        if d not in curr_domain:
            ood_acc.append(acc)

    aux["ood_acc"] = np.mean(ood_acc)
    list_acc = aux["list_acc"]
    aux["list_acc"] = np.mean([sum(x) for x in aux["list_acc"]]) / batch_size
    aux["baseline"] = np.mean(aux["baseline"])

    if aux["enc_log_probs"][0].ndim == 1:
        enc = torch.stack(aux["enc_log_probs"])
        enc = enc.mean(dim=0)
    else:
        enc = [x.mean(dim=0) for x in aux["enc_log_probs"] if x.size(0) > 0]
        enc = [x for x in enc if x.size(0) > 0]
        enc = torch.stack(enc)
        enc = enc.mean(dim=0)

    aux["enc_log_probs"] = enc

    dec = [x.mean(dim=0) for x in aux["dec_log_probs"] if x.size(0) > 0]
    dec = [x.mean(dim=0) if x.ndim > 1 else x for x in dec]

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

    utterance = copy.deepcopy(hypos)
    utterance = translator.s2l(utterance)
    dec_utt = list_vocab.batch_decode(utterance)

    masks = get_mask_from_utts(
        utterance, translator.list_vocab, device=enc_logits.device
    )

    # get outputs
    list_out = list_model(utterance, context_separate, masks)
    list_out = list_out.squeeze(-1)

    # Losses and preds
    list_preds = torch.argmax(list_out, dim=1)
    list_acc = list_preds.eq(target)
    list_loss = loss_f(list_out, target)
    weighted_list_loss = list_loss.mean() * common_p.list_loss_weight

    # baseline
    bs = baseline.predict(list_loss.mean().detach())
    bs_diff = list_loss.detach() - bs

    # dec loss
    dec_log_probs = logprobs_from_logits(dec_logits, hypos)
    policy_loss = (bs_diff * dec_log_probs).mean()
    weighted_policy_loss = policy_loss * common_p.policy_loss_weight

    # entropy loss

    distr = Categorical(logits=dec_logits)
    entropy = distr.entropy()
    entropy_loss = -entropy.mean()
    weighted_entropy_loss = entropy_loss * common_p.entropy_loss_weight

    loss = weighted_policy_loss + weighted_entropy_loss + weighted_list_loss

    list_loss = list_loss.mean()
    if speak_model.training:
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
        domains=data["domain"],
    )

    return loss, aux


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
    torch.set_deterministic_debug_mode(True)

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
        project="speak_ec_finetune",
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

    # remove grad from list model
    for param in list_model.parameters():
        param.requires_grad = False

    ##########################
    # SPEAKER
    ##########################

    speak_check, _ = load_wandb_checkpoint(
        SPEAKER_CHK, device,
        #datadir=join("./artifacts", SPEAKER_CHK.split("/")[-1])
    )
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

    optimizer = optim.AdamW(speaker_model.parameters(), lr=common_p.learning_rate)

    loss_f = nn.CrossEntropyLoss(reduction="none")

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        verbose=True,

    )

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

    data_domain = common_p.data_domain
    common_p.data_domain = data_domain

    dataset = FinetuneDataset(
        domain=data_domain,
        num_images=common_p.episodes,
        device=device,
        vectors_file=common_p.vectors_file,
        img2dom_file=common_p.img2dom_file,
        seed=42,
    )
    dataloader_train = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    dataset = FinetuneDataset(
        domain=data_domain,
        num_images=common_p.episodes,
        device=device,
        vectors_file=common_p.vectors_file,
        img2dom_file=common_p.img2dom_file,
        seed=41,
    )
    dataloader_eval = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    translator = Translator(speak_vocab, list_vocab, device)
    baseline = MeanBaseline()

    ###################################
    ## Initial eval loop
    ###################################
    with torch.no_grad():
        speaker_model.eval()

        print(f"\nEvaluation")
        initial_aux = []

        for data in rich.progress.track(
            dataloader_eval,
            total=len(dataloader_eval),
            description=f"Initial Evaluation...",
        ):
            loss, aux = get_predictions(
                data, list_model, speaker_model, loss_f, translator, baseline
            )

            initial_aux.append(aux)

        initial_aux = merge_dict(initial_aux)

        normalize_aux(initial_aux, logger, 0)

        initial_aux["speak_embeds"] = (
            speaker_model.embedding.weight.mean().detach().cpu().numpy()
        )

        eval_accuracy, eval_loss = initial_aux["list_acc"], initial_aux["loss"]

        print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% ")
        logger.on_eval_end(
            initial_aux, list_domain=data_domain, modality="initial_eval"
        )
    aux['enc_log_probs']=torch.as_tensor(aux['enc_log_probs'])
    aux['dec_log_probs']=torch.as_tensor(aux['dec_log_probs'])

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
        dataloader_train.dataset.randomize_data(seed=epoch)
        scheduler.step()

        # torch.enable_grad()
        ###################################
        ##  TRAIN LOOP
        ###################################

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
            # nn.utils.clip_grad_value_(speaker_model.parameters(), clip_value=2.0)
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
            auxs = []

            for data in rich.progress.track(
                dataloader_eval,
                total=len(dataloader_eval),
                description=f"Evaluating...",
            ):
                loss, aux = get_predictions(
                    data, list_model, speaker_model, loss_f, translator, baseline
                )

                auxs.append(aux)

        aux = merge_dict(auxs)
        normalize_aux(aux, logger, epoch)

        eval_accuracy, eval_loss = aux["list_acc"], aux["loss"]

        print(f"Evaluation loss {eval_loss:.6f}, accuracy {eval_accuracy * 100:.3f}% ")
        logger.on_eval_end(aux, list_domain=data_domain, modality="eval")

        # Estimate diff with initial eval phase
        aux_diff = {}
        aux_diff["speak_embeds"] = (
            speaker_model.embedding.weight.mean().detach().cpu().numpy()
        )
        for k in aux.keys():
            try:
                aux_diff[k] = aux[k] - initial_aux[k]
            except:
                pass

        logger.on_eval_end(aux_diff, list_domain=data_domain, modality="diff_eval")

        ###################################
        ##  SAVE MODEL AND EARLY STOPPING
        ###################################

        if common_p.sweep_file == "" and epoch > 0 and epoch % 400 == 0:
            save_model(
                model=speaker_model,
                model_type=f"speaker_ec_{domain}",
                epoch=epoch,
                accuracy=eval_accuracy,
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

    save_model(
        model=speaker_model,
        model_type=f"speaker_ec_{domain}",
        epoch=epoch,
        accuracy=eval_accuracy,
        args=common_p,
        timestamp=timestamp,
        logger=logger,
        loss=eval_loss,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, finishing run")
        wandb.finish()
    finally:
        print("Clearing GPU memory")
        torch.cuda.empty_cache()
        gc.collect()
