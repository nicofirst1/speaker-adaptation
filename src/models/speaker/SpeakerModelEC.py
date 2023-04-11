from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial
from src.commons import standardize, to_concat_context, set_seed

class TemperatureSampler:
    """
    ## Sampler with Temperature
    """
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        """
        Sample from logits
        """

        # Create a categorical distribution with temperature adjusted logits
        dist = Categorical(logits=logits / self.temperature)


        # Sample
        return dist.sample()

class SpeakerModelEC(nn.Module):
    def __init__(
            self,
            vocab,
            embedding_dim,
            hidden_dim,
            img_dim,
            dropout_prob,
            attention_dim,
            sampler_temp,
            max_len,
            top_k,
            top_p,
            device,

    ):
        super().__init__()
        self.vocab = vocab

        # remove nohos from prediction
        vocab_len = len(vocab) - 1

        self.vocab_size = vocab_len
        self.max_len = max_len
        self.top_k = top_k
        self.top_p = top_p

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.dropout_prob = dropout_prob
        self.device = device

        self.sampler = TemperatureSampler(temperature=sampler_temp)

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
        self.enc_hid2voc = nn.Linear(self.hidden_dim * 2, self.vocab_size )
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
            self.enc_hid2voc,
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

        hypos, dec_logits = self.nucleus_sampling(
            decoder_hid, history_att, top_p=self.top_p
        )

        model_params["decoder_logits"] = dec_logits

        return hypos, model_params, decoder_hid

    def partial_forward(
            self,
            visual_context: torch.Tensor,
            target_img_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        batch_size = visual_context.shape[0]

        # standardize input
        visual_context = to_concat_context(visual_context)
        visual_context = standardize(visual_context)
        target_img_feats = standardize(target_img_feats)

        visual_context_hid = self.relu(self.lin_viscontext(visual_context))
        target_img_hid = self.relu(self.linear_separate(target_img_feats))

        concat_visual_input = torch.cat((visual_context_hid, target_img_hid), dim=1)

        concat_visual_input = self.relu(self.linear_hid(concat_visual_input))

        # here we need to get rid of all prev_utterances, do so by filling an empty torch tensor
        pad_val = self.vocab.word2index["<pad>"]
        nohs_val = self.vocab.word2index["<nohs>"]

        empty_utt = torch.full((batch_size, 1), pad_val)
        empty_utt[:, 0] = nohs_val
        prev_utterance = empty_utt.to(self.device)
        prev_utt_lengths = torch.as_tensor([1]*batch_size).to(self.device)

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
        outputs = outputs.squeeze(1)

        logit = self.enc_hid2voc(outputs)

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
            encoder_logits=logit,
        )

        return decoder_hid, history_att, model_params

    def nucleus_sampling(
            self,
            decoder_hid,
            history_att,
            top_p=0.0,
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


        gen_len = 0

        decoder_input = sos_token  # beam_k sos copies
        dec_logits = torch.zeros((self.max_len, batch_size, self.lin2voc.out_features)).to(self.device)
        eos_mask = torch.zeros((self.max_len +1, batch_size)).to(self.device)


        while True:


            if gen_len >= self.max_len:
                break  # very long sentence generated


            decoder_embeds = self.embedding(decoder_input)
            decoder_embeds = decoder_embeds.squeeze(1)
            h1, c1 = self.lstm_decoder(decoder_embeds, hx=(h1, c1))

            h1_att = torch.cat((h1, history_att), dim=-1)
            dec_logit = self.lin2voc(h1_att)
            dec_logits[gen_len] = dec_logit

            word_pred = F.softmax(dec_logit, dim=1, )

            word_pred = word_pred.squeeze()


            if top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(word_pred,dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                nucleus = cumulative_probs < top_p


                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                sorted_log_probs = torch.log(sorted_probs)
                sorted_log_probs[~nucleus] = float('-inf')

                sampled_sorted_indexes = self.sampler(sorted_log_probs)
                next_token = sorted_indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
                next_token.squeeze(-1)



            decoder_input = next_token

            word_index = next_token % (len(self.vocab) - 1)  # predicted word
            completed_sentences.append(word_index)

            eos_idxs = word_index == eos_token
            # get index of eos
            idx = eos_idxs.nonzero(as_tuple=True)[0]
            if len(idx) > 0:

                # normalize logit
                for i in idx:
                    eos_mask[gen_len + 1:, i] = 1

            if eos_mask[gen_len + 1].all():
                break

            gen_len += 1

        # truncate after eos
        completed_sentences = torch.concat(completed_sentences, dim=-1)
        completed_sentences = [sent for sent in completed_sentences]

        completed_sentences = torch.stack(completed_sentences)

        if completed_sentences.ndim == 1:
            completed_sentences = completed_sentences.unsqueeze(0)

        # truncate
        eos_mask=eos_mask[:self.max_len,:]
        eos_mask = eos_mask[:gen_len + 1, :]
        dec_logits = dec_logits[:gen_len + 1, :, :]

        # for each column of the mask get the index of the first 1
        eos_idxs = torch.argmax(eos_mask, dim=0)
        # change values zero with max_len
        eos_idxs[eos_idxs == 0] = self.max_len

        # convert to mask
        eos_mask = eos_mask.bool()

        # apply mask to complete_sentences
        completed_sentences = completed_sentences.masked_fill(eos_mask.T, self.vocab["<pad>"])
        dec_logits[eos_mask,:] =0

        # sum and normalize logits
        dec_logits= F.normalize(dec_logits, p=2, dim=0)
        #dec_logits = dec_logits.sum(0)

        # for idx in range(len(eos_idxs)):
        #     dec_logits[idx,:] = dec_logits[idx,:] / eos_idxs[idx]
        #

        # dec_logits=torch.matmul(dec_logits, self.embedding.weight)
        # dec_logits = F.normalize(dec_logits, p=2, dim=-1)

        return completed_sentences, dec_logits
