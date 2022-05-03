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

def eval_beam_base(split_data_loader, model, args, best_score, print_gen, device,
                   beam_size, max_len, vocab, nlgeval_obj, isValidation, timestamp, isTest):
    """
        Evaluation

        :param beam_size: beam size at which to generate captions for evaluation
        :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
        """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = []
    hypotheses = []

    count = 0

    empty_count = 0

    breaking = args.breaking

    sos_token = torch.tensor(vocab['<sos>']).to(device)
    eos_token = torch.tensor(vocab['<eos>']).to(device)

    if isValidation:
        split = 'val'
    elif isTest:
        split = 'test'
    else:
        split = 'train'

    file_name = args.model_type + '_' + args.metric + '_' + split + '_' + timestamp  # overwrites previous versions!

    for i, data in enumerate(split_data_loader):
        # print(i)

        completed_sentences = []
        completed_scores = []

        beam_k = beam_size

        if breaking and count == 5:
            break

        count += 1

        # dataset details
        # only the parts I will use for this type of model

        # utterance = data['utterance']  # to be decoded, we don't use this here in beam search!
        # target_utterance = utterance[:,1:]
        # I am using the one below as references for the calculation of metric scores
        orig_text_reference = data['orig_utterance']  # original reference without unk, eos, sos, pad
        reference_chain = data['reference_chain'][0]  # batch size 1  # full set of references for a single instance
        # obtained from the whole chain

        visual_context = data['concat_context']
        target_img_feats = data['target_img_feats']

        # this model uses only the visual input to start off the decoder to generate the next utterance
        # no attention, no history

        visual_context_hid = model.relu(model.lin_viscontext(visual_context))
        target_img_hid = model.relu(model.linear_separate(target_img_feats))

        decoder_hid = model.relu(model.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1)))

        decoder_hid = decoder_hid.expand(beam_k, -1)
        # multiple copies of the decoder
        h1, c1 = decoder_hid, decoder_hid

        # ***** beam search *****

        gen_len = 0

        decoder_input = sos_token.expand(beam_k, 1)  # beam_k sos copies

        gen_sentences_k = decoder_input  # all start off with sos now

        top_scores = torch.zeros(beam_k, 1).to(device)  # top-k generation scores

        while True:

            # EOS?

            if gen_len > max_len:
                break  # very long sentence generated

            # generate

            # sos segment eos
            # base model with visual input

            decoder_embeds = model.embedding(decoder_input).squeeze(1)

            h1, c1 = model.lstm_decoder(model.lin_mm(torch.cat((decoder_embeds, decoder_hid), dim=1)), hx=(h1, c1))

            word_pred = F.log_softmax(model.lin2voc(h1), dim=1)

            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                # all same

                # static std::tuple<Tensor, Tensor> at::topk(const Tensor &self, int64_t k,
                # int64_t dim = -1, bool largest = true, bool sorted = true)

                top_scores, top_words = word_pred[0].topk(beam_k, 0, True, True)

            else:
                # unrolled
                top_scores, top_words = word_pred.view(-1).topk(beam_k, 0, True, True)

            # vocab - 1 to exclude <NOHS>
            sentence_index = top_words / (len(vocab) - 1)  # which sentence it will be added to
            word_index = top_words % (len(vocab) - 1)  # predicted word

            gen_len += 1

            # add the newly generated word to the sentences
            gen_sentences_k = torch.cat((gen_sentences_k[sentence_index], word_index.unsqueeze(1)), dim=1)

            # there could be incomplete sentences
            incomplete_sents_inds = [inc for inc in range(len(gen_sentences_k)) if
                                     eos_token not in gen_sentences_k[inc]]

            complete_sents_inds = list(set(range(len(word_index))) - set(incomplete_sents_inds))

            # save the completed sentences
            if len(complete_sents_inds) > 0:
                completed_sentences.extend(gen_sentences_k[complete_sents_inds].tolist())
                completed_scores.extend(top_scores[complete_sents_inds])

                beam_k -= len(complete_sents_inds)  # fewer, because we closed at least 1 beam

            if beam_k == 0:
                break

            # continue generation for the incomplete sentences
            gen_sentences_k = gen_sentences_k[incomplete_sents_inds]

            # use the ongoing hidden states of the incomplete sentences
            h1, c1 = h1[sentence_index[incomplete_sents_inds]], c1[sentence_index[incomplete_sents_inds]],

            top_scores = top_scores[incomplete_sents_inds].unsqueeze(1)
            decoder_input = word_index[incomplete_sents_inds]
            decoder_hid = decoder_hid[incomplete_sents_inds]

        if len(completed_scores) == 0:
            empty_count += 1
            # print('emptyseq', empty_count)

            # all incomplete here

            completed_sentences.extend((gen_sentences_k[incomplete_sents_inds].tolist()))
            completed_scores.extend(top_scores[incomplete_sents_inds])

        sorted_scores, sorted_indices = torch.sort(torch.tensor(completed_scores), descending=True)

        best_seq = completed_sentences[sorted_indices[0]]

        hypothesis = [vocab.index2word[w] for w in best_seq if w not in
                      [vocab.word2index['<sos>'], vocab.word2index['<eos>'], vocab.word2index['<pad>']]]
        # remove sos and pads # I want to check eos
        hypothesis_string = ' '.join(hypothesis)
        hypotheses.append(hypothesis_string)

        if not os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
            # Reference
            references.append(reference_chain)

        if print_gen:
            # Reference
            print('REF:', orig_text_reference)  # single one
            print('HYP:', hypothesis_string)

    if os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
        with open('speaker_outputs/refs_' + file_name + '.json', 'r') as f:
            references = json.load(f)
    else:
        with open('speaker_outputs/refs_' + file_name + '.json', 'w') as f:
            json.dump(references, f)
    #
    # if os.path.isfile('speaker_outputs/refs_BERT_' + file_name + '.json'):
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'r') as f:
    #         references_BERT = json.load(f)
    # else:
    #     references_BERT = [r[0] for r in references]
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'w') as f:
    #         json.dump(references_BERT, f)

    # Calculate scores
    metrics_dict = nlgeval_obj.compute_metrics(references, hypotheses)
    print(metrics_dict)

    (P, R, Fs), hashname = score(hypotheses, references, lang='en', return_hash=True, model_type="bert-base-uncased")
    print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')

    if args.metric == 'cider':
        selected_metric_score = metrics_dict['CIDEr']
        print(round(selected_metric_score, 5))

    elif args.metric == 'bert':
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

            with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
                json.dump(hypotheses, f)

        return best_score, selected_metric_score, metrics_dict, has_best_score

    if isTest:
        with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
            json.dump(hypotheses, f)


def eval_beam_histatt(split_data_loader, model, args, best_score, print_gen, device,
                      beam_size, max_len, vocab, mask_attn, nlgeval_obj, isValidation, timestamp, isTest):
    """
        Evaluation

        :param beam_size: beam size at which to generate captions for evaluation
        :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
        """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = []
    hypotheses = []

    count = 0

    empty_count = 0

    breaking = args.breaking

    sos_token = torch.tensor(vocab['<sos>']).to(device)
    eos_token = torch.tensor(vocab['<eos>']).to(device)

    if isValidation:
        split = 'val'
    elif isTest:
        split = 'test'
    else:
        split = 'train'

    file_name = args.model_type + '_' + args.metric + '_' + split + '_' + timestamp  # overwrites previous versions!

    for i, data in enumerate(split_data_loader):
        # print(i)

        completed_sentences = []
        completed_scores = []

        beam_k = beam_size

        if breaking and count == 5:
            break

        count += 1

        # dataset details
        # only the parts I will use for this type of model

        utterance = data['utterance']  # to be decoded, we don't use this here in beam search!
        # target_utterance = utterance[:,1:]
        # I am using the one below as references for the calculation of metric scores
        orig_text_reference = data['orig_utterance']  # original reference without unk, eos, sos, pad
        reference_chain = data['reference_chain'][0]  # batch size 1  # full set of references for a single instance
        # obtained from the whole chain

        prev_utterance = data['prev_utterance']
        prev_utt_lengths = data['prev_length']

        visual_context = data['concat_context']
        target_img_feats = data['target_img_feats']

        max_length_tensor = prev_utterance.shape[1]

        masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

        visual_context_hid = model.relu(model.lin_viscontext(visual_context))
        target_img_hid = model.relu(model.linear_separate(target_img_feats))

        concat_visual_input = model.relu(model.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1)))

        embeds_words = model.embedding(prev_utterance)  # b, l, d

        # pack sequence

        sorted_prev_utt_lens, sorted_idx = torch.sort(prev_utt_lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        concat_visual_input = concat_visual_input[sorted_idx]

        # RuntimeError: Cannot pack empty tensors.
        packed_input = nn.utils.rnn.pack_padded_sequence(embeds_words, sorted_prev_utt_lens, batch_first=True)

        # start lstm with average visual context:
        # conditioned on the visual context

        # he, ce = self.init_hidden(batch_size, device)
        concat_visual_input = torch.stack((concat_visual_input, concat_visual_input), dim=0)

        packed_outputs, hidden = model.lstm_encoder(packed_input, hx=(concat_visual_input, concat_visual_input))

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # already concat forward backward

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        # ONLY THE HIDDEN AND OUTPUT ARE REVERSED
        # next_utterance is aligned (pre_utterance info is not)
        batch_out_hidden = hidden[0][:, reversed_idx]  # .squeeze(0)

        # start decoder with these

        # teacher forcing

        decoder_hid = model.linear_dec(torch.cat((batch_out_hidden[0], batch_out_hidden[1]), dim=1))

        history_att = model.lin2att_hist(outputs)

        decoder_hid = decoder_hid.expand(beam_k, -1)

        # multiple copies of the decoder
        h1, c1 = decoder_hid, decoder_hid

        # ***** beam search *****

        gen_len = 0

        decoder_input = sos_token.expand(beam_k, 1)  # beam_k sos copies

        gen_sentences_k = decoder_input  # all start off with sos now

        top_scores = torch.zeros(beam_k, 1).to(device)  # top-k generation scores

        while True:

            # EOS?

            if gen_len > max_len:
                break  # very long sentence generated

            # generate

            # sos segment eos
            # base model with visual input

            decoder_embeds = model.embedding(decoder_input).squeeze(1)

            h1, c1 = model.lstm_decoder(decoder_embeds, hx=(h1, c1))

            h1_att = model.lin2att_hid(h1)

            attention_out = model.attention(model.tanh(history_att + h1_att.unsqueeze(1)))

            attention_out = attention_out.masked_fill_(masks, float('-inf'))

            att_weights = model.softmax(attention_out)

            att_context_vector = (history_att * att_weights).sum(dim=1)

            word_pred = F.log_softmax(model.lin2voc(torch.cat((h1, att_context_vector), dim=1)), dim=1)

            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                # all same

                # static std::tuple<Tensor, Tensor> at::topk(const Tensor &self, int64_t k,
                # int64_t dim = -1, bool largest = true, bool sorted = true)

                top_scores, top_words = word_pred[0].topk(beam_k, 0, True, True)

            else:
                # unrolled
                top_scores, top_words = word_pred.view(-1).topk(beam_k, 0, True, True)

            # vocab - 1 to exclude <NOHS>
            sentence_index = top_words // (len(vocab) - 1)  # which sentence it will be added to
            word_index = top_words % (len(vocab) - 1)  # predicted word

            gen_len += 1

            # add the newly generated word to the sentences
            gen_sentences_k = torch.cat((gen_sentences_k[sentence_index], word_index.unsqueeze(1)), dim=1)

            # there could be incomplete sentences
            incomplete_sents_inds = [inc for inc in range(len(gen_sentences_k)) if
                                     eos_token not in gen_sentences_k[inc]]

            complete_sents_inds = list(set(range(len(word_index))) - set(incomplete_sents_inds))

            # save the completed sentences
            if len(complete_sents_inds) > 0:
                completed_sentences.extend(gen_sentences_k[complete_sents_inds].tolist())
                completed_scores.extend(top_scores[complete_sents_inds])

                beam_k -= len(complete_sents_inds)  # fewer, because we closed at least 1 beam

            if beam_k == 0:
                break

            # continue generation for the incomplete sentences
            gen_sentences_k = gen_sentences_k[incomplete_sents_inds]

            # use the ongoing hidden states of the incomplete sentences
            h1, c1 = h1[sentence_index[incomplete_sents_inds]], c1[sentence_index[incomplete_sents_inds]],

            top_scores = top_scores[incomplete_sents_inds].unsqueeze(1)
            decoder_input = word_index[incomplete_sents_inds]
            decoder_hid = decoder_hid[incomplete_sents_inds]

        if len(completed_scores) == 0:
            empty_count += 1
            # print('emptyseq', empty_count)

            # all incomplete here

            completed_sentences.extend((gen_sentences_k[incomplete_sents_inds].tolist()))
            completed_scores.extend(top_scores[incomplete_sents_inds])

        sorted_scores, sorted_indices = torch.sort(torch.tensor(completed_scores), descending=True)

        best_seq = completed_sentences[sorted_indices[0]]

        hypothesis = [vocab.index2word[w] for w in best_seq if w not in
                      [vocab.word2index['<sos>'], vocab.word2index['<eos>'], vocab.word2index['<pad>']]]
        # remove sos and pads # I want to check eos
        hypothesis_string = ' '.join(hypothesis)
        hypotheses.append(hypothesis_string)

        if not os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
            # Reference
            references.append(reference_chain)

        if print_gen:
            # Reference
            print('REF:', orig_text_reference)  # single one
            print('HYP:', hypothesis_string)

    if os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
        with open('speaker_outputs/refs_' + file_name + '.json', 'r') as f:
            references = json.load(f)
    else:
        with open('speaker_outputs/refs_' + file_name + '.json', 'w') as f:
            json.dump(references, f)
    #
    # if os.path.isfile('speaker_outputs/refs_BERT_' + file_name + '.json'):
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'r') as f:
    #         references_BERT = json.load(f)
    # else:
    #     references_BERT = [r[0] for r in references]
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'w') as f:
    #         json.dump(references_BERT, f)

    # Calculate scores
    metrics_dict = nlgeval_obj.compute_metrics([references], hypotheses)
    print(metrics_dict)

    (P, R, Fs), hashname = score(hypotheses, references, lang='en', return_hash=True, model_type="bert-base-uncased")
    print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')

    if args.metric == 'cider':
        selected_metric_score = metrics_dict['CIDEr']
        print(round(selected_metric_score, 5))

    elif args.metric == 'bert':
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

            with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
                json.dump(hypotheses, f)

        return best_score, selected_metric_score, metrics_dict, has_best_score

    if isTest:
        with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
            json.dump(hypotheses, f)
