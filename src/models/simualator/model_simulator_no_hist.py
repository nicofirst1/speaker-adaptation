from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from src.models.listener.ListenerModel_hist import ListenerModel_hist
from src.models.listener.ListenerModel_no_hist import ListenerModel_no_hist


class SimulatorModel_no_hist(ListenerModel_no_hist):
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
        super(SimulatorModel_no_hist, self).__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            img_dim,
            att_dim,
            dropout_prob,
            domain,
            device,
        )

        self.att_linear_2 = nn.Linear(self.attention_dim, self.hidden_dim)
        self.init_weights()  # initialize layers

    def forward(
        self,
        speaker_embeds: torch.Tensor,
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

        separate_images = separate_images.to(self.device)
        visual_context = visual_context.to(self.device)

        representations = speaker_embeds
        # [32,512]

        batch_size = representations.shape[0]  # effective batch size

        # utterance representations are processed
        representations = self.dropout(representations)
        input_reps = self.relu(self.lin_emb2hid(representations))
        # [32,512]

        # visual context is processed
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_context(visual_context))


        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, projected_context), dim=-1))
        )

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_2(self.tanh(self.att_linear_1(mm_reps)))

        # mask pads so that no attention is paid to them (with -inf)
        # outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        #att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = mm_reps * outputs_att

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        dot = torch.bmm(
            separate_images, attended_hids.view(batch_size, self.hidden_dim, 1)
        )
        #[batch, 6, 1]

        return dot
