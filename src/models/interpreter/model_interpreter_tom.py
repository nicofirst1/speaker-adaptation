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


class InterpreterModel_tom(ListenerModel_no_hist):
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
        args,
    ):
        super(InterpreterModel_tom, self).__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            img_dim,
            att_dim,
            dropout_prob,
            domain,
            device,
        )
        self.relu = nn.LeakyReLU()

        # self.att_linear_2 = nn.Linear(self.attention_dim, self.hidden_dim)
        self.init_weights()  # initialize layers

    def utterance_forward(self, speaker_utterances,projected_context, masks):

        representations = self.embeddings(speaker_utterances)
        # utterance representations are processed
        representations = self.dropout(representations)
        input_reps = self.lrelu(self.lin_emb2hid(representations))
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
        outputs_att = self.relu(outputs_att)
        outputs_att = F.normalize(outputs_att, p=2, dim=1)
        outputs_att = self.att_linear_2(outputs_att)
        # outputs_att = self.relu(outputs_att)

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
        representations = self.dropout(speaker_embeds)
        input_reps = self.lrelu(self.lin_emb2hid(representations))
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
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_context(visual_context))
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

        embeds=torch.bmm(utt_out,embeds_out)
        dot = torch.bmm(separate_images, embeds.view(batch_size, self.hidden_dim, 1))
        # [batch, 6, 1]

        return dot
