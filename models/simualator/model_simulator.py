from typing import List

import torch
from torch import nn

from models.listener.model_listener import ListenerModel


class SimulatorModel(ListenerModel):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob
    ):
        super(ListenerModel).__init__(
            vocab_size, embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob
        )
        self.speaker_influence = nn.Linear(6, self.hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        separate_images: torch.Tensor,
        visual_context: torch.Tensor,
        prev_hist: List,
        masks: torch.Tensor,
        device,
    ):
        """
        @param text: utterances to be converted into embeddings
        @param separate_images: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param prev_hist: contains histories for 6 images separately (if exists for a given image)
        @param masks: attention mask for pad tokens
        @param device: device to which the tensors are moved
        """

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance, dim [batch_size,6,1]
        dot = super(SimulatorModel, self).forward(
            text, separate_images, visual_context, prev_hist, masks, device
        )

        return dot
