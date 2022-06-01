import os

import numpy as np
import rich.progress
import torch
from torch import optim, nn
from transformers import BertTokenizer, BertModel

from data.dataloaders.Vocab import Vocab
from data.dataloaders.utils import get_dataloaders
from evals.utils import hypo2utterance
from models.listener.model_listener import ListenerModel
from models.simualator.model_simulator import SimulatorModel
from models.speaker.model_speaker_hist_att import SpeakerModelHistAtt
from trainers.parsers import parse_args
from trainers.utils import mask_attn
from wandb_logging import ListenerLogger, load_wandb_checkpoint

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    common_args = parse_args("speak")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # ALREADY do_lower_case=True

    # Load pre-trained model (weights)
    model_bert = BertModel.from_pretrained("bert-base-uncased")

    # Set the model in evaluation mode to deactivate the DropOut modules
    # This is IMPORTANT to have reproducible results during evaluation!
    model_bert.eval()
    model_bert.to(device)

    speaker_url = "adaptive-speaker/speaker/SpeakerModelHistAtt:v6"

    speak_check, _ = load_wandb_checkpoint(speaker_url, device)

    # load args
    speak_p = speak_check["args"]
    speak_p.vocab_file = "vocab.csv"
    speak_p.__post_init__()

    print(speak_p)

    # for reproducibility
    seed = speak_p.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(speak_p.vocab_file)
    vocab.index2word[len(vocab)] = "<nohs>"  # special token placeholder for no prev utt
    vocab.word2index["<nohs>"] = len(vocab)  # len(vocab) updated (depends on w2i)

    img_dim = 2048

    # init speak model and load state
    speaker_model = SpeakerModelHistAtt(
        vocab,
        speak_p.embedding_dim,
        speak_p.hidden_dim,
        img_dim,
        speak_p.dropout_prob,
        speak_p.attention_dim,
    ).to(device)

    speaker_model.load_state_dict(speak_check["model_state_dict"])
    speaker_model = speaker_model.to(device)

    # listener dict
    listener_dict = dict(
        all="adaptive-speaker/listener/ListenerModel_all:v20",
        appliances="adaptive-speaker/listener/ListenerModel_appliances:v20",
        food="adaptive-speaker/listener/ListenerModel_food:v20",
        indoor="adaptive-speaker/listener/ListenerModel_indoor:v20",
        outdoor="adaptive-speaker/listener/ListenerModel_outdoor:v20",
        vehicles="adaptive-speaker/listener/ListenerModel_vehicles:v20",
    )

    domain = "food"

    list_checkpoint, _ = load_wandb_checkpoint(listener_dict[domain], device)
    list_args = list_checkpoint["args"]

    # update list args
    list_args.batch_size = 1  # hypotesis generation does not support batch
    list_args.vocab_file = "vocab.csv"
    list_args.vectors_file = os.path.basename(list_args.vectors_file)
    list_args.device = device

    # for debug
    # list_args.subset_size = 10

    # update paths
    # list_args.__parse_args()
    list_args.__post_init__()
    vocab = Vocab(list_args.vocab_file)

    list_model = ListenerModel(
        len(vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
    ).to(device)

    list_model.load_state_dict(list_checkpoint["model_state_dict"])
    list_model = list_model.to(device)

    sim_model = SimulatorModel(
        len(vocab),
        list_args.embed_dim,
        list_args.hidden_dim,
        img_dim,
        list_args.attention_dim,
        list_args.dropout_prob,
    ).to(device)


    ###################################
    ##  LOSS AND OPTIMIZER
    ###################################

    learning_rate = list_args.learning_rate
    optimizer = optim.Adam(sim_model.parameters(), lr=learning_rate)

    reduction_method = list_args.reduction
    criterion = nn.CrossEntropyLoss(reduction=reduction_method)


    # add debug label
    tags = []
    if list_args.debug or list_args.subset_size != -1:
        tags = ["debug"]

    logger = ListenerLogger(
        vocab=vocab,
        opts=vars(list_args),
        group=list_args.train_domain,
        train_logging_step=1,
        val_logging_step=1,
        tags=tags,
        project="speaker-list-dom"
    )

    beam_k = 5
    max_len = 30

    for epoch in range(list_args['epochs']):

        print("Epoch : ", epoch)

        if epoch > 0:
            # load datasets again to shuffle the image sets to avoid biases
            training_loader, _, val_loader, _ = get_dataloaders(list_args, vocab, domain)

        losses = []

        list_model.train()
        torch.enable_grad()

        count = 0

        ###################################
        ##  TRAIN LOOP
        ###################################

        for i, data in rich.track(
                enumerate(training_loader),
                total=len(training_loader),
                description="Training",
        ):

            # generate hypo with speaker
            hypo, _ = speaker_model.generate_hypothesis(data, beam_k, max_len, device)

            utterance = hypo2utterance(hypo, tokenizer, vocab)

            # get datapoints
            context_separate = data["separate_images"]
            context_concat = data["concat_context"]
            lengths = [utterance.shape[1]]
            targets = data["target"]
            prev_hist = data["prev_histories"]

            max_length_tensor = utterance.shape[1]
            masks = mask_attn(lengths, max_length_tensor, device)

            sim_out, influence = sim_model(
                utterance, context_separate, context_concat, prev_hist, masks, device
            )
            hypo, _ = speaker_model.generate_hypothesis(data, beam_k, max_len, device, sim_embedding=influence)

            utterance = hypo2utterance(hypo, tokenizer, vocab)

            # get listener output
            list_out = list_model(
                utterance, context_separate, context_concat, prev_hist, masks, device
            )

            targets = targets.to(device)

            list_loss = criterion(list_out, targets)
            sim_loss = criterion(list_out, sim_out)
            loss=list_loss+sim_loss

            preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
            logger.on_batch_end(
                loss, data, aux={"preds": preds}, batch_id=i, modality="train"
            )

            losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            preds = torch.argmax(list_out, dim=1)

            correct = torch.eq(preds, targets).sum()
            accuracies.append(float(correct))

            scores_ranked, images_ranked = torch.sort(list_out.squeeze(), descending=True)

            if list_out.shape[0] > 1:
                for s in range(list_out.shape[0]):
                    # WARNING - assumes batch size > 1
                    rank_target = images_ranked[s].tolist().index(targets[s].item())
                    ranks.append(rank_target + 1)  # no 0

            else:
                rank_target = images_ranked.tolist().index(targets.item())
                ranks.append(rank_target + 1)  # no 0

            # targets = torch.tensor([[torch.argmax(tg)] for tg in targets]).to(device)
            # TARGETS SUITABLE FOR CROSS-ENTROPY LOSS
            targets = targets.to(device)
            loss = criterion(list_out, targets)

            preds = torch.argmax(list_out.squeeze(dim=-1), dim=1)
            logger.on_batch_end(
                loss, data, aux={"preds": preds}, batch_id=i, modality="train"
            )



        losses = np.mean(losses)
        print("Train loss sum", round(losses, 5))  # sum all the batches for this epoch
        logger.log_datapoint(data, preds, modality="train")
        logger.log_viz_embeddings(data, modality="train")

        ###################################
        ##  EVAL LOOP
        ###################################

        with torch.no_grad():
            model.eval()

            isValidation = True
            print(f'\nVal Eval on domain "{domain}"')
            current_accuracy, current_loss, current_MRR = evaluate(
                val_loader, model, in_domain=True
            )

            print(f"\nVal Eval on all domains")
            evaluate(val_loader_speaker, model, in_domain=False)

            if metric == "loss":

                if best_loss <= current_loss:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_loss = current_loss
                    best_epoch = epoch

                    save_model(
                        model,
                        model_type,
                        best_epoch,
                        current_accuracy,
                        current_loss,
                        current_MRR,
                        optimizer,
                        args,
                        "loss",
                        timestamp,
                        args.seed,
                        t,
                    )

                print("patience", patience_counter, "\n")

                print("\nBest", best_epoch, round(best_loss, 5), metric)  # validset
                print()

            elif metric == "accs":

                if best_accuracy >= current_accuracy:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_accuracy = current_accuracy
                    best_epoch = epoch

                    save_model(
                        model=model,
                        model_type=model_type,
                        epoch=best_epoch,
                        accuracy=current_accuracy,
                        optimizer=optimizer,
                        args=args,
                        timestamp=timestamp,
                        logger=logger,
                        loss=current_loss,
                        mrr=current_MRR,
                    )

                print("patience", patience_counter)

                print(
                    "\nBest", best_epoch, round(best_accuracy, 5), metric, "\n"
                )  # validset

            elif metric == "mrr":

                if best_mrr >= current_MRR:

                    patience_counter += 1

                    if patience_counter == patience:
                        duration = datetime.datetime.now() - t

                        print("training ending duration", duration)

                        break
                else:

                    patience_counter = 0

                    best_mrr = current_MRR
                    best_epoch = epoch

                    save_model(
                        model,
                        model_type,
                        best_epoch,
                        current_accuracy,
                        current_loss,
                        current_MRR,
                        optimizer,
                        args,
                        "mrr",
                        timestamp,
                        args.seed,
                        t,
                    )

                print("patience", patience_counter)

                print(
                    "\nBest", best_epoch, round(best_mrr, 5), metric, "\n"
                )  # validset

            prev_accuracy = current_accuracy
            prev_loss = current_loss
            prev_mrr = current_MRR

        logger.on_train_end({"loss": losses}, epoch_id=epoch)
