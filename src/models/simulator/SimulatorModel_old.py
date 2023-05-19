import torch
from src.commons import standardize, to_concat_context
from torch import nn


def linear(input_dim, output_dim):
    """
    Initializes a linear layer with xavier initialization
    """
    linear = nn.Linear(input_dim, output_dim)
    torch.nn.init.xavier_uniform_(linear.weight)
    return linear


class SimulatorModel_old(nn.Module):
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

        # initialize embeddings
        torch.nn.init.xavier_uniform_(self.embeddings.weight)

        # project images to hidden dimensions
        self.linear_separate = linear(self.img_dim, self.hidden_dim)

        # from embedding dimensions to hidden dimensions
        # self.lin_emb2hid = linear(self.embedding_dim, self.hidden_dim)

        self.lin_emb2hid = self.init_sequential(embedding_dim, 1, use_leaky=True)

        # Concatenation of 6 images in the context projected to hidden
        # self.lin_context = linear(self.img_dim * 6, self.hidden_dim)
        self.lin_context = self.init_sequential(self.img_dim * 6, 2, use_leaky=True)

        # Multimodal (text representation; visual context)
        # self.lin_mm = linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_mm = self.init_sequential(self.hidden_dim, 1, use_leaky=False)

        # attention linear layers
        self.att_linear_1 = linear(self.hidden_dim, self.attention_dim)
        self.att_linear_2 = linear(self.attention_dim, 1)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

    def init_sequential(self, input_dim, num_layers, use_leaky=False, end_dim=-1):
        """
        Initializes the sequential layers of the model
        """

        nn_lin = nn.LeakyReLU() if use_leaky else nn.ReLU()

        layers = [self.dropout, linear(input_dim, self.hidden_dim), nn_lin]

        for l in range(num_layers - 1):
            layers.append(self.dropout)
            layers.append(linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn_lin)

        if end_dim != -1:
            layers.append(self.dropout)
            layers.append(linear(self.hidden_dim, end_dim))
            layers.append(nn_lin)

        return nn.Sequential(*layers)

    def utterance_forward(self, speaker_utterances, projected_context, masks):
        """
        Forward pass for the utterance representations
        @param speaker_utterances: the speaker generated utterances
        @param projected_context: the visual context
        @param masks: the masks for the utterances
        @return:

        """

        representations = self.embeddings(speaker_utterances)

        # utterance representations are processed
        input_reps = self.lin_emb2hid(representations)
        input_reps = standardize(input_reps)

        repeated_context = projected_context.unsqueeze(1)

        # multimodal utterance representations
        mm_reps = input_reps * repeated_context
        mm_reps = self.lin_mm(mm_reps)

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_1(mm_reps)
        outputs_att = self.lrelu(outputs_att)
        outputs_att = standardize(outputs_att)
        outputs_att = self.att_linear_2(outputs_att)

        # mask pads so that no attention is paid to them (with -inf)
        masks = masks.bool()
        outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = (mm_reps * att_weights).sum(dim=1)

        return attended_hids

    def embeds_forward(self, speaker_embeds, projected_context):
        """
        Forward pass for the embedding representations
        @param speaker_embeds: speaker embeddings
        @param projected_context: visual context
        @return:
        """

        # utterance representations are processed
        input_reps = self.lin_emb2hid(speaker_embeds)
        # input_reps = F.normalize(input_reps, p=2, dim=1)
        input_reps = standardize(input_reps)

        # multimodal utterance representations
        mm_reps = input_reps * projected_context

        mm_reps = self.lin_mm(mm_reps)
        mm_reps = standardize(mm_reps)

        return mm_reps

    def forward(
        self,
        speaker_embeds: torch.Tensor,
        utterance: torch.Tensor,
        separate_images: torch.Tensor,
        masks: torch.Tensor,
    ):
        """
        @param speaker_embeds: utterances coming from the speaker embeddings
        @param separate_images: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param masks: attention mask for pad tokens
        """
        utterance = utterance.to(self.device)
        separate_images = separate_images.to(self.device)
        visual_context = to_concat_context(separate_images)

        visual_context = standardize(visual_context)
        separate_images = standardize(separate_images)
        speaker_embeds = standardize(speaker_embeds)

        # visual context is processed
        projected_context = self.lin_context(visual_context)
        projected_context = standardize(projected_context)

        # utterance representations are processed
        utt_out = self.utterance_forward(utterance, projected_context, masks)
        embeds_out = self.embeds_forward(speaker_embeds, projected_context)

        #################
        # visual context
        #################

        batch_size = utterance.shape[0]

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)
        separate_images = self.relu(separate_images)
        separate_images = standardize(separate_images)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance

        embeds = utt_out * embeds_out
        dot = torch.bmm(separate_images, embeds.view(batch_size, self.hidden_dim, 1))
        # [batch, 6, 1]

        return dot
