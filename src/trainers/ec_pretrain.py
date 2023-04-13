import datetime
from typing import Dict, Tuple

import lovely_tensors as lt
import numpy as np
import rich.progress
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.commons import (SPEAKER_CHK, EarlyStopping,
                         get_listener_check, load_wandb_checkpoint,
                         mask_attn, mask_oov_embeds,
                         merge_dict, parse_args, save_model, set_seed)
from src.commons.Baseline import MeanBaseline
from src.commons.Translator import Translator
from src.commons.attack_utils import AttackModule
from src.commons.model_utils import logprobs_from_logits
from src.data.dataloaders import Vocab
from src.data.dataloaders.EcDataset import EcDataset
from src.models import ListenerModel, SpeakerModelEC
from src.wandb_logging import ListenerLogger

global common_p
global list_vocab


def add_image(aux, list_acc, logger, max_targets=2):
    idx = 0

    target_ids = np.random.choice(range(len(aux["target_id"])), max_targets)

    table_columns = ["epch", "img_id", "utterance", "perturbed_utterance", "img", "was correct?"]
    table_values = []
    for i in target_ids:
        t_id = aux["target_id"][i]
        utt = aux["utterance"][i]
        perb_utt = aux["perturbed_utts"][i]
        la = list_acc[i]

        jdx = np.random.choice(range(len(t_id)))
        t_id = t_id[jdx]
        utt = utt[jdx]
        perb_utt = perb_utt[jdx]
        la = la[jdx]

        t_id = logger.img_id2path[t_id]

        img_orig = wandb.Image(t_id, caption=utt)

        # table_values.append([epoch, idx, utt, perb_utt, img, la])

        aux[f"img_{idx}"] = img_orig

        if perb_utt != "-":
            img_pert = wandb.Image(t_id, caption=perb_utt)
            aux[f"perturbed_img_{idx}"] = img_pert

        idx += 1


def normalize_aux(aux, logger, epoch, max_targets=2):
    batch_size = len(aux["target_id"][0])
    aux["loss"] = np.mean(aux["loss"])
    aux["policy_loss"] = np.mean(aux["policy_loss"])
    aux["list_loss"] = np.mean(aux["list_loss"])
    aux["entropy_loss"] = np.mean(aux["entropy_loss"])
    aux["adversarial_loss"] = np.mean(aux["adversarial_loss"])
    aux["weighted_policy_loss"] = np.mean(aux["weighted_policy_loss"])
    aux["weighted_entropy_loss"] = np.mean(aux["weighted_entropy_loss"])
    aux["weighted_list_loss"] = np.mean(aux["weighted_list_loss"])
    aux["weighted_adversarial_loss"] = np.mean(aux["weighted_adversarial_loss"])

    aux["perplexity"] = np.mean(aux["perplexity"])

    list_acc = aux["list_acc"]
    aux["list_acc"] = np.mean([sum(x) for x in aux["list_acc"]]) / batch_size
    aux["baseline"] = np.mean(aux["baseline"])
    aux['enc_log_probs'] = torch.stack(aux['enc_log_probs']).mean(dim=0).numpy()

    dec = [x.mean(dim=0) for x in aux['dec_log_probs'] if x.size(0) > 0]
    dec = [x for x in dec if x.size(0) > 0]
    try:
        dec = torch.stack(dec).mean(dim=0).numpy()
    except RuntimeError:
        dec = np.zeros_like(aux['enc_log_probs'])

    aux['dec_log_probs'] = dec

    # remove nans from log probs
    aux['enc_log_probs'] = np.nan_to_num(aux['enc_log_probs'])
    aux['dec_log_probs'] = np.nan_to_num(aux['dec_log_probs'])

    # get max targets random ids in range of targets
    if epoch % 10 == 0:
        add_image(aux, list_acc, logger, max_targets=max_targets)

    # aux["utt_table"] = wandb.Table(columns=table_columns, data=table_values)

    del aux["target_id"]
    del aux["utterance"]
    del aux["perturbed_utts"]


# FGSM attack code


def get_predictions(
        data: Dict,
        list_model: ListenerModel,
        speak_model: SpeakerModelEC,
        loss_f: nn.CrossEntropyLoss,
        translator: Translator,
        baseline: MeanBaseline,
        attack: AttackModule = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract data, get list/sim out, estimate losses and create log dict

    """
    global common_p
    global list_vocab

    # get datapoints
    context_separate = data["image_set"]
    target = data["target_index"]
    image_ids = data["image_ids"]
    target_id = data["target_id"]
    target_img_feat = data["target_img_feat"]

    hypos, model_params, _ = speak_model.generate_hypothesis(context_separate, target_img_feat)

    enc_logits = model_params['encoder_logits']
    dec_logits = model_params['decoder_logits']

    utterance = hypos
    translator.s2l(utterance)
    dec_utt = list_vocab.batch_decode(utterance)

    lengths = torch.tensor([len(x) for x in utterance])
    max_length_tensor = torch.max(lengths).item()
    # get mask and translate utterance
    masks = mask_attn(lengths, max_length_tensor, list_model.device)

    # get outputs
    list_out = list_model(utterance, context_separate, masks)
    list_out = list_out.squeeze(-1)

    # Losses and preds
    list_preds = torch.argmax(list_out, dim=1)
    list_acc = list_preds.eq(target)
    list_loss = loss_f(list_out, target).mean()

    if common_p.adversarial_loss_weight > 0 and speak_model.training and not list_acc.sum():

        perturbed_batch = attack(utterance, (context_separate, masks), target)

        perturbed_hypo = list_vocab.batch_decode(perturbed_batch)

        dl = dec_logits.permute(1, 2, 0)

        if (perturbed_batch != utterance).any():
            translator.l2s(perturbed_batch)
            adversarial_loss = loss_f(dl, perturbed_batch).mean()
            perplexity = torch.exp(adversarial_loss)
        else:
            adversarial_loss = torch.tensor(0.0)
            perturbed_hypo = ["-" for _ in range(len(utterance))]
            perplexity = torch.tensor(0.0)
    else:
        adversarial_loss = torch.tensor(0.0)
        perturbed_hypo = ["-" for _ in range(len(utterance))]
        perplexity = torch.tensor(0.0)

    enc_log_probs = logprobs_from_logits(enc_logits, model_params['encoder_ids'].squeeze(dim=0))
    dec_log_probs = logprobs_from_logits(dec_logits, hypos)

    bs = baseline.predict(list_loss.detach())
    if common_p.logits_to_use == "enc":
        policy_loss = (list_loss.detach() - bs) * enc_log_probs
        distr = Categorical(logits=enc_logits)
    elif common_p.logits_to_use == "dec":

        policy_loss = (list_loss.detach() - bs) * dec_log_probs
        distr = Categorical(logits=dec_log_probs)
    else:
        joint_log_probs = enc_log_probs + dec_log_probs
        policy_loss = (list_loss.detach() - bs) * joint_log_probs
        distr = Categorical(logits=joint_log_probs)

    policy_loss = policy_loss.mean()
    weighted_policy_loss = policy_loss * common_p.policy_loss_weight

    entropy = distr.entropy()
    entropy_loss = - entropy.mean()
    weighted_entropy_loss = entropy_loss * common_p.entropy_loss_weight

    weighted_list_loss = list_loss * common_p.list_loss_weight
    weighted_adversarial_loss = adversarial_loss * common_p.adversarial_loss_weight

    loss = weighted_policy_loss + weighted_entropy_loss + weighted_adversarial_loss + weighted_list_loss

    baseline.update(list_loss.detach())

    aux = dict(
        loss=loss.detach().cpu().item(),
        policy_loss=policy_loss.detach().cpu().item(),
        list_loss=list_loss.detach().cpu().item(),
        entropy_loss=entropy_loss.detach().cpu().item(),
        adversarial_loss=adversarial_loss.detach().cpu().item(),
        weighted_policy_loss=weighted_policy_loss.detach().cpu().item(),
        weighted_entropy_loss=weighted_entropy_loss.detach().cpu().item(),
        weighted_adversarial_loss=weighted_adversarial_loss.detach().cpu().item(),
        weighted_list_loss=weighted_list_loss.detach().cpu().item(),

        baseline=bs.detach().cpu().item(),
        perplexity=perplexity.detach().cpu().item(),

        utterance=dec_utt,
        perturbed_utts=perturbed_hypo,
        target_id=target_id,
        list_acc=list_acc,

        enc_log_probs=torch.log_softmax(enc_logits, dim=-1).detach().cpu().squeeze(),
        dec_log_probs=torch.log_softmax(dec_logits.permute(1,0,2), dim=-1).detach().cpu().squeeze(),

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
    episodes = common_p.episodes

    for data in rich.progress.track(
            data_loader,
            total=episodes,
            description=f"evaluating '{split}' split...",
    ):
        loss, aux = get_predictions(
            data, list_model, speak_model, loss_f, translator, baseline
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
    global list_vocab

    common_p = parse_args("list")
    domain = common_p.train_domain

    device = torch.device("cuda") if torch.cuda.is_available() and common_p.device!="cpu" else torch.device("cpu")

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
        project="ec_pretrain",
    )

    ##########################
    # LISTENER
    ##########################

    list_check = get_listener_check(domain, common_p.golden_data_perc)

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
            replace_token=common_p.mask_oov_embed,
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

    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    optimizer = optim.AdamW(
        speaker_model.parameters(), lr=common_p.learning_rate, weight_decay=0.0001
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=common_p.epochs // 10, factor=0.5, verbose=True, threshold=0.5
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
    attack = AttackModule(list_model, eps=common_p.attack_eps, steps=common_p.attack_steps, top_k=common_p.attack_top_k,
                          std_mult=common_p.attack_std_mult)

    ###################################
    ##  Get  dataloader
    ###################################
    # need batchsize =1 for generating the new dataloaders
    data_domain = common_p.data_domain

    kwargs = get_kwargs("train", common_p)
    dataset = EcDataset(**kwargs)
    dataloader_train = DataLoader(
        dataset,
        batch_size=common_p.batch_size,
        collate_fn=dataset.get_collate_fn(),
    )

    kwargs = get_kwargs("val", common_p)
    dataset = EcDataset(**kwargs)
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
        # dataloader_train.dataset.randomize_data()
        # dataloader_eval.dataset.randomize_data()

        # torch.enable_grad()
        ###################################
        ##  TRAIN LOOP
        ###################################
        baseline = MeanBaseline()

        for data in rich.progress.track(
                dataloader_train,
                total=common_p.episodes,
                description=f"Training epoch {epoch}",
        ):
            optimizer.zero_grad()

            # get datapoints
            loss, aux = get_predictions(
                data,
                list_model,
                speaker_model,
                loss_f,
                translator,
                baseline,
                attack
            )

            auxs.append(aux)

            # optimizer
            loss.backward()
            # nn.utils.clip_grad_value_(speaker_model.parameters(), clip_value=1.0)
            optimizer.step()

        aux = merge_dict(auxs)
        aux['lr'] = optimizer.param_groups[0]['lr']

        normalize_aux(aux, logger, epoch)

        aux.update(attack.get_stats())

        logger.on_eval_end(
            aux, list_domain=data_domain, modality="train"
        )

        print(
            f"Train loss {aux['loss']:.6f}, accuracy {aux['list_acc'] * 100:.2f}% "
        )

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
            logger.on_eval_end(
                aux, list_domain=data_domain, modality="eval"
            )

        if epoch > 0 and epoch % 50 == 0:
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
