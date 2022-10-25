from typing import List

import torch
import torch.nn.functional as F
from src.models.listener.ListenerModel_no_hist import ListenerModel_no_hist
from torch import nn


class InterpreterModel_dev(nn.Module):
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
            num_layers,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.attention_dim = att_dim
        self.device = device
        self.domain = domain

        # embeddings learned from scratch
        self.embeddings = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0,
            scale_grad_by_freq=True,
        )

        self.dropout = nn.Dropout(dropout_prob)


        # from embedding dimensions to hidden dimensions
        self.lin_emb2hid=self.init_sequential(self.embedding_dim, num_layers['num_layers_emb2hid'], use_leaky=True)

        # Concatenation of 6 images in the context projected to hidden
        self.lin_context=self.init_sequential(self.img_dim * 6, num_layers['num_layers_contx'], use_leaky=False)

        # Multimodal (text representation; visual context)
        self.lin_mm=self.init_sequential(self.hidden_dim * 2, num_layers['num_layers_mm'], use_leaky=False)

        # project images to hidden dimensions
        layers = [nn.Linear(self.img_dim, self.hidden_dim), nn.LeakyReLU(), self.dropout]

        for l in range(num_layers['num_layers_sep']):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(self.dropout)

        self.linear_separate= nn.Sequential(*layers)

    def init_sequential(self, input_dim, num_layers, use_leaky=False):
        """
        Initializes the sequential layers of the model
        """

        nn_lin=nn.LeakyReLU() if use_leaky else nn.ReLU()

        layers=[nn.Linear(input_dim, self.hidden_dim), nn_lin, self.dropout, nn.BatchNorm1d(self.hidden_dim)]

        for l in range(num_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn_lin)
            layers.append(self.dropout)
            layers.append(nn.BatchNorm1d(self.hidden_dim))

        return nn.Sequential(*layers)




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

        input_reps=self.lin_emb2hid(representations)
        #input_reps = F.normalize(input_reps, p=2, dim=1)

        # [32,512]
        # visual context is processed

        projected_context = self.lin_context(visual_context)
        #projected_context = F.normalize(projected_context, p=2, dim=1)

        # multimodal utterance representations
        mm_reps =self.lin_mm(torch.cat((input_reps, projected_context), dim=-1))


        # image features per image in context are processed
        separate_images = self.linear_separate(separate_images)
        #separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        dot = torch.bmm(separate_images, mm_reps.view(batch_size, self.hidden_dim, 1))
        # [batch, 6, 1]

        return dot
