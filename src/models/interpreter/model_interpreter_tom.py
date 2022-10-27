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
    ):
        super(InterpreterModel_no_hist, self).__init__(
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

        representations = self.embeddings(speaker_utterances)

        #representations = speaker_embeds
        # [32,512]

        batch_size = representations.shape[0]  # effective batch size

        # utterance representations are processed
        representations = self.dropout(representations)
        input_reps = self.lrelu(self.lin_emb2hid(representations))
        input_reps = F.normalize(input_reps, p=2, dim=1)

        # [32,512]
        # visual context is processed
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_context(visual_context))
        projected_context = F.normalize(projected_context, p=2, dim=1)

        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, projected_context), dim=-1))
        )

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        dot = torch.bmm(separate_images, mm_reps.view(batch_size, self.hidden_dim, 1))
        # [batch, 6, 1]

        return dot
