from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from src.models.listener.ListenerModel_hist import ListenerModel_hist
from src.models.listener.ListenerModel_no_hist import ListenerModel_no_hist


class SimulatorModel_multi(ListenerModel_no_hist):
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
        super(SimulatorModel_multi, self).__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            img_dim,
            att_dim,
            dropout_prob,
            domain,
            device,
        )

        self.att_linear_11 = nn.Linear(self.hidden_dim, self.attention_dim)
        self.att_linear_21 = nn.Linear(self.attention_dim, 1)

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
        input_reps = input_reps.unsqueeze(dim=1)

        # visual context is processed
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_context(visual_context))

        repeated_context = projected_context.unsqueeze(1).repeat(
            1, input_reps.shape[1], 1
        )
        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, repeated_context), dim=2))
        )

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_list = self.att_linear_2(self.tanh(self.att_linear_1(mm_reps)))
        outputs_targ = self.att_linear_21(self.tanh(self.att_linear_11(mm_reps)))

        # mask pads so that no attention is paid to them (with -inf)
        # outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        att_weights_list = self.softmax(outputs_list)
        att_weights_targ = self.softmax(outputs_targ)

        # encoder context representation
        attended_hids_list = (mm_reps * att_weights_list).sum(dim=1)
        attended_hids_targ = (mm_reps * att_weights_targ).sum(dim=1)

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        dot_list = torch.bmm(
            separate_images, attended_hids_list.view(batch_size, self.hidden_dim, 1)
        )
        dot_targ = torch.bmm(
            separate_images, attended_hids_targ.view(batch_size, self.hidden_dim, 1)
        )
        #[batch, 6, 1]

        return dot_list,dot_targ
