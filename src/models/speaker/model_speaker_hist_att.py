import torch
import torch.nn as nn
import torch.nn.functional as F

from src.commons import mask_attn


class SpeakerModel(nn.Module):
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
            device,
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab) - 1
        self.beam_k = beam_k
        self.max_len = max_len

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

        self.lin_mm = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.attention = nn.Linear(self.attention_dim, 1)

        self.relu = nn.ReLU()

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

    def forward(
            self,
            utterance,
            lengths,
            prev_utterance,
            prev_utt_lengths,
            visual_context_sep,
            visual_context,
            target_img_feats,
            targets,
            prev_hist,
            prev_hist_len,
            normalize,
            masks,

    ):

        """
        @param utterance: ground-truth subsequent utterance converted into indices using the reduced vocabulary,
        which will be fed into the decoder during teacher forcing
        @param lengths: utterance lengths
        @param prev_utterance: if exists, the previous utterance for the image, if not <nohs>
        @param prev_utt_lengths: length of the previous utterance
        @param visual_context_sep: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param target_img_feats: features of the image for which we will generate a new utterance
        @param targets, prev_hist, prev_hist_len, normalize: not used in this self
        @param masks: masks for pad tokens
        @param device: device to which the tensors are moved
        """

        batch_size = utterance.shape[0]  # effective batch size
        decode_length = utterance.shape[1] - 1  # teacher forcing (except eos)

        # visual context and target image features are processed
        visual_context_hid = self.relu(
            self.lin_viscontext(self.dropout(visual_context))
        )
        target_img_hid = self.relu(self.linear_separate(self.dropout(target_img_feats)))

        # concatenated visual input (context; target)
        concat_visual_input = self.relu(
            self.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1))
        )

        # previous utterance is embedded
        embeds_words = self.dropout(self.embedding(prev_utterance))  # b, l, d

        # pack sequence
        sorted_prev_utt_lens, sorted_idx = torch.sort(prev_utt_lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        concat_visual_input = concat_visual_input[sorted_idx]

        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeds_words.cpu(), sorted_prev_utt_lens.cpu(), batch_first=True
        )
        packed_input = packed_input.to(self.device)

        # start LSTM encoder conditioned on the visual input
        concat_visual_input = torch.stack(
            (concat_visual_input, concat_visual_input), dim=0
        )

        # feed the previous utterance into the LSTM encoder
        packed_outputs, hidden = self.lstm_encoder(
            packed_input, hx=(concat_visual_input, concat_visual_input)
        )

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # already concat forward backward (timestep t, same position)

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        batch_out_hidden = hidden[0][:, reversed_idx]

        # teacher forcing

        # word prediction scores
        predictions = torch.zeros(batch_size, decode_length, self.vocab_size).to(self.device)

        # forward backward concatenation of encoder's last hidden states
        decoder_hid = self.linear_dec(
            torch.cat((batch_out_hidden[0], batch_out_hidden[1]), dim=1)
        )

        history_att = self.lin2att_hist(outputs)

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

    def generate_hypothesis(self, prev_utterance, prev_utt_lengths, visual_context, target_img_feats):

        # dataset details
        # only the parts I will use for this type of self

        completed_sentences = []
        completed_scores = []
        empty_count = 0
        beam_k = self.beam_k

        sos_token = torch.tensor(self.vocab["<sos>"]).to(self.device)
        eos_token = torch.tensor(self.vocab["<eos>"]).to(self.device)

        # obtained from the whole chain

        max_length_tensor = prev_utterance.shape[1]

        masks = mask_attn(prev_utt_lengths, max_length_tensor, self.device)

        visual_context_hid = self.relu(self.lin_viscontext(visual_context))
        target_img_hid = self.relu(self.linear_separate(target_img_feats))

        concat_visual_input = self.relu(
            self.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1))
        )

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

        history_att = self.lin2att_hist(outputs)

        # todo: for adaptive speak h1 should have grad, also in eval
        h1_sim = decoder_hid

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

        model_params = dict(
            att_context_vector=att_context_vector,
            decoder_embeds=decoder_embeds,
            embeds_words=embeds_words,
            target_img_hid=target_img_hid,
            visual_context_hid=visual_context_hid,
        )

        return hypothesis_string, model_params, h1_sim

    def simulator_forward(self, **kwargs):

        predictions, target_utterance_embeds = self.forward(**kwargs, simulator=True)

        return predictions, target_utterance_embeds
