from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.commons import mask_attn, standardize, to_concat_context


class SpeakerModelEC(nn.Module):
    def __init__(
            self,
            vocab,
            embedding_dim,
            hidden_dim,
            img_dim,
            dropout_prob,
            attention_dim,
            beam_k,
            max_len,
            top_k,
            top_p,
            device,
            use_beam=False,
            use_prev_utterances=False,

    ):
        super().__init__()
        self.vocab = vocab

        # remove nohos from prediction
        vocab_len = len(vocab) - 1

        self.vocab_size = vocab_len
        self.beam_k = beam_k
        self.max_len = max_len
        self.top_k = top_k
        self.top_p = top_p
        self.use_beam = use_beam
        self.use_prev_utterances = use_prev_utterances

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.dropout_prob = dropout_prob
        self.device = device

        # attention over encoder steps
        self.attention_dim = attention_dim

        # embeddings learned from scratch (adding +1 because the embedding for nohs is also learned)
        self.embedding = nn.Embedding(
            self.vocab_size + 1,
            self.embedding_dim,
            padding_idx=0,
            scale_grad_by_freq=True,
        )

        # Bidirectional LSTM encoder for the previous utterance
        self.lstm_encoder = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # BIDIRECTIONAL

        # LSTM decoder for generating the next utterance
        self.lstm_decoder = nn.LSTMCell(self.embedding_dim, self.hidden_dim, bias=True)

        self.linear_hid = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.linear_dec = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.lin_viscontext = nn.Linear(self.img_dim * 6, self.hidden_dim)

        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        # attention related hidden layers
        self.lin2att_hid = nn.Linear(self.hidden_dim, self.attention_dim)

        self.lin2att_hist = nn.Linear(
            self.hidden_dim * 2, self.attention_dim
        )  # 2 because of BiLSTM

        # project to vocabulary size
        self.lin2voc = nn.Linear(self.attention_dim + self.hidden_dim, self.vocab_size)
        self.hid2voc = nn.Linear(self.hidden_dim * 2, self.vocab_size)

        self.lin_mm = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.attention = nn.Linear(self.attention_dim, 1)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        for ll in [
            self.linear_hid,
            self.linear_dec,
            self.linear_separate,
            self.lin2voc,
            self.lin_viscontext,
            self.lin_mm,
            self.lin2att_hist,
            self.lin2att_hid,
            self.attention,
        ]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def generate_hypothesis(
            self,
            visual_context: torch.Tensor,
            target_img_feats: torch.Tensor,
    ) -> Tuple[str, Dict, torch.Tensor]:
        """
        Generate an hypothesis (natural language sentence) based on the current output
        Does not support batch, so needs batch_size=1

        Returns
        -------
        hypos : str = string for hypotesis
        model_params: dict = dictionary with various model parameters used for logging
        decoder_hid : torch.Tensor = the decoder hidden output, used for simulator train
        """

        decoder_hid, history_att, model_params = self.partial_forward(
              visual_context, target_img_feats
        )

        model_params["history_att"] = history_att



        hypos = self.nucleus_sampling(
            decoder_hid, history_att, top_p=self.top_p, top_k=self.top_k
        )


        return hypos, model_params, decoder_hid

    def partial_forward(
            self,
            visual_context: torch.Tensor,
            target_img_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:



        batch_size = visual_context.shape[0]

        # standardize input
        visual_context=to_concat_context(visual_context)
        visual_context = standardize(visual_context)
        target_img_feats = standardize(target_img_feats)

        visual_context_hid = self.relu(self.lin_viscontext(visual_context))
        target_img_hid = self.relu(self.linear_separate(target_img_feats))

        concat_visual_input = self.relu(
            self.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1))
        )

        # here we need to get rid of all prev_utterances, do so by filling an empty torch tensor
        pad_val = self.vocab.word2index["<pad>"]
        nohs_val = self.vocab.word2index["<nohs>"]

        empty_utt = torch.full((batch_size,self.vocab_size), pad_val)
        empty_utt[:, 0] = nohs_val
        prev_utterance = empty_utt.to(self.device)
        prev_utt_lengths = torch.tensor([1]).to(self.device)

        embeds_words = self.embedding(prev_utterance)  # b, l, d

        # pack sequence
        sorted_prev_utt_lens, sorted_idx = torch.sort(prev_utt_lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        concat_visual_input = concat_visual_input[sorted_idx]

        # RuntimeError: Cannot pack empty tensors.
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeds_words.cpu(), sorted_prev_utt_lens.cpu(), batch_first=True
        )
        packed_input = packed_input.to(self.device)

        # start lstm with average visual context:
        # conditioned on the visual context

        # he, ce = self.init_hidden(batch_size, self.device)
        concat_visual_input = torch.stack(
            (concat_visual_input, concat_visual_input), dim=0
        )

        packed_outputs, hidden = self.lstm_encoder(
            packed_input, hx=(concat_visual_input, concat_visual_input)
        )

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # already concat forward backward

        logit=self.hid2voc(outputs)

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        # ONLY THE HIDDEN AND OUTPUT ARE REVERSED
        # next_utterance is aligned (pre_utterance info is not)
        batch_out_hidden = hidden[0][:, reversed_idx]  # .squeeze(0)

        # start decoder with these

        # teacher forcing

        decoder_hid = self.linear_dec(
            torch.cat((batch_out_hidden[0], batch_out_hidden[1]), dim=1)
        )
        decoder_hid = self.tanh(decoder_hid)

        if decoder_hid.ndim == 1:
            decoder_hid = decoder_hid.unsqueeze(0)

        history_att = self.lin2att_hist(outputs)

        model_params = dict(
            embeds_words=embeds_words,
            target_img_hid=target_img_hid,
            visual_context_hid=visual_context_hid,
            logits=logit,
        )

        return decoder_hid, history_att, model_params

    def nucleus_sampling(
            self,
            decoder_hid,
            history_att,
            top_k=0,
            top_p=0.0,
            filter_value=-float("Inf"),
    ):
        """Filter a distribution using top-k and/or nucleus (top-p) filtering
        Args:
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """

        completed_sentences = []
        batch_size = decoder_hid.shape[0]

        # standardize input
        history_att = standardize(history_att)

        sos_token = torch.tensor(self.vocab["<sos>"]).to(self.device)
        eos_token = torch.tensor(self.vocab["<eos>"]).to(self.device)
        sos_token = sos_token.repeat((batch_size, 1))
        eos_token = eos_token.repeat((batch_size, 1))

        # multiple copies of the decoder
        h1, c1 = decoder_hid, decoder_hid

        # ***** beam search *****

        gen_len = 0

        decoder_input = sos_token  # beam_k sos copies

        while True:

            # EOS?

            if gen_len > self.max_len:
                break  # very long sentence generated

            # generate

            # sos segment eos
            # base self with visual input

            decoder_embeds = self.embedding(decoder_input)
            decoder_embeds = decoder_embeds.squeeze(1)
            h1, c1 = self.lstm_decoder(decoder_embeds, hx=(h1, c1))

            h1_att = self.lin2att_hid(h1)
            h1_att = h1_att.unsqueeze(1)
            attention_out = self.relu(history_att + h1_att)
            attention_out = standardize(attention_out)
            attention_out = self.attention(attention_out)

            masks = torch.zeros_like(attention_out).bool()
            attention_out = attention_out.masked_fill_(masks, float("-inf"))

            att_weights = self.softmax(attention_out)

            att_context_vector = (history_att * att_weights).sum(dim=1)

            word_pred = F.log_softmax(
                self.lin2voc(torch.cat((h1, att_context_vector), dim=1)),
                dim=1,
            )

            word_pred = word_pred.squeeze()


            top_k = min(top_k, word_pred.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = (
                        word_pred < torch.topk(word_pred, top_k)[0][..., -1, None]
                )
                word_pred[indices_to_remove] = filter_value

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(word_pred, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                if sorted_indices.dim() == 1:
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        0, sorted_indices, sorted_indices_to_remove
                    )
                else:
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                word_pred = word_pred.masked_fill(indices_to_remove, filter_value)

            probabilities = F.softmax(word_pred, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            # next_token = next_token.squeeze()
            decoder_input = next_token

            word_index = next_token % (len(self.vocab) - 1)  # predicted word
            completed_sentences.append(word_index)

            if (word_index == eos_token).all():
                break

            gen_len += 1

        # truncate after eos
        completed_sentences = torch.concat(completed_sentences, dim=-1)
        completed_sentences = [sent for sent in completed_sentences]

        completed_sentences = torch.stack(completed_sentences)

        if completed_sentences.ndim == 1:
            completed_sentences = completed_sentences.unsqueeze(0)

        for i in range(len(completed_sentences)):
            x = completed_sentences[i]

            # get index of eos
            idx = (x == self.vocab["<eos>"]).nonzero(as_tuple=True)[0]
            # if found
            if len(idx) > 0:
                idx = idx[0]
                # pad from index to max length with pad token
                completed_sentences[i][idx:] = torch.tensor([self.vocab["<pad>"]]) * (
                        self.max_len - len(x)
                )

        last_idx = torch.nonzero(completed_sentences)[:, -1]
        if len(last_idx) > 0:
            last_idx = torch.max(last_idx) + 1
            completed_sentences = completed_sentences[:, :last_idx]


        return completed_sentences

    def beam_serach(
            self,
            decoder_hid: torch.Tensor,
            history_att: torch.Tensor,
            masks: torch.Tensor,
            model_params: Optional[Dict] = {},
    ) -> str:
        completed_sentences = []
        completed_scores = []
        empty_count = 0
        beam_k = self.beam_k

        sos_token = torch.tensor(self.vocab["<sos>"]).to(self.device)
        eos_token = torch.tensor(self.vocab["<eos>"]).to(self.device)

        decoder_hid = decoder_hid.expand(beam_k, -1)

        # multiple copies of the decoder
        h1, c1 = decoder_hid, decoder_hid

        # ***** beam search *****

        gen_len = 0

        decoder_input = sos_token.expand(beam_k, 1)  # beam_k sos copies

        gen_sentences_k = decoder_input  # all start off with sos now

        top_scores = torch.zeros(beam_k, 1).to(self.device)  # top-k generation scores

        while True:

            # EOS?

            if gen_len > self.max_len:
                break  # very long sentence generated

            # generate

            # sos segment eos
            # base self with visual input

            decoder_embeds = self.embedding(decoder_input).squeeze(1)

            h1, c1 = self.lstm_decoder(decoder_embeds, hx=(h1, c1))

            h1_att = self.lin2att_hid(h1)

            attention_out = self.attention(self.tanh(history_att + h1_att.unsqueeze(1)))

            attention_out = attention_out.masked_fill_(masks, float("-inf"))

            att_weights = self.softmax(attention_out)

            att_context_vector = (history_att * att_weights).sum(dim=1)

            word_pred = F.log_softmax(
                self.lin2voc(torch.cat((h1, att_context_vector), dim=1)), dim=1
            )

            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                # all same

                # static std::tuple<Tensor, Tensor> at::topk(const Tensor &self, int64_t k,
                # int64_t dim = -1, bool largest = true, bool sorted = true)

                top_scores, top_words = word_pred[0].topk(beam_k, 0, True, True)

            else:
                # unrolled
                top_scores, top_words = word_pred.view(-1).topk(beam_k, 0, True, True)

            # self.vocab - 1 to exclude <NOHS>
            sentence_index = top_words // (
                    len(self.vocab) - 1
            )  # which sentence it will be added to
            word_index = top_words % (len(self.vocab) - 1)  # predicted word

            gen_len += 1

            # add the newly generated word to the sentences
            gen_sentences_k = torch.cat(
                (gen_sentences_k[sentence_index], word_index.unsqueeze(1)), dim=1
            )

            # there could be incomplete sentences
            incomplete_sents_inds = [
                inc
                for inc in range(len(gen_sentences_k))
                if eos_token not in gen_sentences_k[inc]
            ]

            complete_sents_inds = list(
                set(range(len(word_index))) - set(incomplete_sents_inds)
            )

            # save the completed sentences
            if len(complete_sents_inds) > 0:
                completed_sentences.extend(
                    gen_sentences_k[complete_sents_inds].tolist()
                )
                completed_scores.extend(top_scores[complete_sents_inds])

                beam_k -= len(
                    complete_sents_inds
                )  # fewer, because we closed at least 1 beam

            if beam_k == 0:
                break

            # continue generation for the incomplete sentences
            gen_sentences_k = gen_sentences_k[incomplete_sents_inds]

            # use the ongoing hidden states of the incomplete sentences
            h1, c1 = (
                h1[sentence_index[incomplete_sents_inds]],
                c1[sentence_index[incomplete_sents_inds]],
            )

            top_scores = top_scores[incomplete_sents_inds].unsqueeze(1)
            decoder_input = word_index[incomplete_sents_inds]
            decoder_hid = decoder_hid[incomplete_sents_inds]

        if len(completed_scores) == 0:
            empty_count += 1
            # print('emptyseq', empty_count)

            # all incomplete here

            completed_sentences.extend(
                (gen_sentences_k[incomplete_sents_inds].tolist())
            )
            completed_scores.extend(top_scores[incomplete_sents_inds])

        sorted_scores, sorted_indices = torch.sort(
            torch.tensor(completed_scores), descending=True
        )

        best_seq = completed_sentences[sorted_indices[0]]

        hypothesis = [
            self.vocab.index2word[w]
            for w in best_seq
            if w
               not in [
                   self.vocab.word2index["<sos>"],
                   self.vocab.word2index["<eos>"],
                   self.vocab.word2index["<pad>"],
               ]
        ]
        # remove sos and pads # I want to check eos
        hypothesis_string = " ".join(hypothesis)

        model_params.update(
            dict(
                att_context_vector=att_context_vector,
                decoder_embeds=decoder_embeds,
            )
        )

        return hypothesis_string

    def forward(
            self,
            utterance,
            prev_utterance,
            prev_utt_lengths,
            visual_context,
            target_img_feats,
            masks,
    ):

        """
        @param utterance: ground-truth subsequent utterance converted into indices using the reduced vocabulary,
        which will be fed into the decoder during teacher forcing
        @param prev_utterance: if exists, the previous utterance for the image, if not <nohs>
        @param prev_utt_lengths: length of the previous utterance
        @param visual_context: concatenation of 6 images in the context
        @param target_img_feats: features of the image for which we will generate a new utterance
        @param masks: masks for pad tokens
        """

        batch_size = utterance.shape[0]  # effective batch size
        decode_length = utterance.shape[1] - 1  # teacher forcing (except eos)

        decoder_hid, history_att, model_params = self.partial_forward(
            prev_utterance, prev_utt_lengths, visual_context, target_img_feats
        )

        # word prediction scores
        predictions = torch.zeros(batch_size, decode_length, self.vocab_size).to(
            self.device
        )

        # start decoder with the hidden states of the encoder
        h1, c1 = decoder_hid, decoder_hid

        # teacher forcing during training, decoder input: ground-truth subsequent utterance
        target_utterance_embeds = self.embedding(utterance)

        for l in range(decode_length):
            # decoder takes target word embeddings
            h1, c1 = self.lstm_decoder(target_utterance_embeds[:, l], hx=(h1, c1))

            # use h1 in attention calculations over the history
            h1_att = self.lin2att_hid(h1)

            # attention calculation (previous utterance and current decoder state interacts)
            attention_out = self.attention(self.tanh(history_att + h1_att.unsqueeze(1)))

            # pad tokens in the previous utterance to mask them out
            attention_out = attention_out.masked_fill_(masks, float("-inf"))

            # final attention weights
            att_weights = self.softmax(attention_out)

            # encoder context representation
            att_context_vector = (history_att * att_weights).sum(dim=1)

            # projection to vocabulary size to predict the word to be generated
            # decoder's current hidden state and encoder context vector
            word_pred = self.lin2voc(torch.cat((h1, att_context_vector), dim=1))

            predictions[:, l] = word_pred

        return predictions
