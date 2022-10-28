from typing import List

import torch
import torch.nn.functional as F
from src.models.listener.ListenerModel_no_hist import ListenerModel_no_hist
from torch import nn

from typing import List

import torch
import torch.nn.functional as F
from src.models.listener.ListenerModel_no_hist import ListenerModel_no_hist
from torch import nn


class InterpreterModel_tom(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        img_dim,
        att_dim,
        dropout_prob,
        domain,
        device,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.attention_dim = att_dim
        self.device = device
        self.domain = domain


        self.dropout = nn.Dropout(dropout_prob)

        # embeddings learned from scratch
        self.embeddings = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0,
            scale_grad_by_freq=True,
        )

        # project images to hidden dimensions
        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        # from embedding dimensions to hidden dimensions
        #self.lin_emb2hid = nn.Linear(self.embedding_dim, self.hidden_dim)

        self.lin_emb2hid=self.init_sequential(self.embedding_dim, 1, use_leaky=True)


        # Concatenation of 6 images in the context projected to hidden
        #self.lin_context = nn.Linear(self.img_dim * 6, self.hidden_dim)
        self.lin_context=self.init_sequential(self.img_dim *6 , 3, use_leaky=False)


        # Multimodal (text representation; visual context)
        #self.lin_mm = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_mm=self.init_sequential(self.hidden_dim * 2, 1, use_leaky=False)

        # attention linear layers
        #self.att_linear_1 = nn.Linear(self.hidden_dim, self.attention_dim)
        self.att_linear_1 = self.init_sequential(self.hidden_dim , 3, use_leaky=True, end_dim=self.attention_dim)
        self.att_linear_2 = nn.Linear(self.attention_dim, 1)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)



    def init_sequential(self, input_dim, num_layers, use_leaky=False, end_dim=-1):
        """
        Initializes the sequential layers of the model
        """

        nn_lin = nn.LeakyReLU() if use_leaky else nn.ReLU()

        layers = [nn.Linear(input_dim, self.hidden_dim), nn_lin, self.dropout]

        for l in range(num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn_lin)
            layers.append(self.dropout)

        if end_dim != -1:
            layers.append(nn.Linear(self.hidden_dim, end_dim))
            layers.append(nn_lin)
            layers.append(self.dropout)

        return nn.Sequential(*layers)

    def utterance_forward(self, speaker_utterances,projected_context, masks):

        representations = self.embeddings(speaker_utterances)
        # utterance representations are processed
        input_reps = self.lin_emb2hid(representations)
        input_reps = F.normalize(input_reps, p=2, dim=1)


        repeated_context = projected_context.unsqueeze(1).repeat(
            1, input_reps.shape[1], 1
        )
        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, repeated_context), dim=-1))
        )

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_1(mm_reps)
        outputs_att = F.normalize(outputs_att, p=2, dim=1)
        outputs_att = self.att_linear_2(outputs_att)

        # mask pads so that no attention is paid to them (with -inf)
        masks = masks.bool()
        outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = (mm_reps * att_weights).sum(dim=1)

        return attended_hids

    def embeds_forward(self,speaker_embeds,projected_context):

        # utterance representations are processed
        input_reps = self.lin_emb2hid(speaker_embeds)
        input_reps = F.normalize(input_reps, p=2, dim=1)


        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, projected_context), dim=-1))
        )

        return mm_reps



    def forward(
        self,
        speaker_out: torch.Tensor,
        separate_images: torch.Tensor,
        visual_context: torch.Tensor,
        prev_hist: List,
        masks: torch.Tensor,
    ):
        """
        @param speaker_embeds: utterances coming from the speaker embeddings
        @param separate_images: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param prev_hist: contains histories for 6 images separately (if exists for a given image)
        @param masks: attention mask for pad tokens
        """
        speaker_embeds, speaker_utterances = speaker_out
        speaker_utterances = speaker_utterances.to(self.device)
        separate_images = separate_images.to(self.device)
        visual_context = visual_context.to(self.device)

        # [32,512]
        # visual context is processed
        projected_context = self.lin_context(visual_context)
        projected_context = F.normalize(projected_context, p=2, dim=1)

        utt_out = self.utterance_forward(speaker_utterances,projected_context,masks)
        embeds_out = self.embeds_forward(speaker_embeds,projected_context)

        #representations = speaker_embeds
        # [32,512]

        batch_size = speaker_utterances.shape[0]  # effective batch size



        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance

        embeds=utt_out*embeds_out
        dot = torch.bmm(separate_images, embeds.view(batch_size, self.hidden_dim, 1))
        # [batch, 6, 1]

        return dot
