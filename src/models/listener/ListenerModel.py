import torch
import torch.nn as nn
from src.commons import standardize, to_concat_context


class ListenerModel(nn.Module):
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
        self.lin_emb2hid = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.lin_emb2HIST = nn.Linear(
            self.embedding_dim, self.hidden_dim
        )  # here we also have history

        # Concatenation of 6 images in the context projected to hidden
        self.lin_context = nn.Linear(self.img_dim * 6, self.hidden_dim)

        # Multimodal (text representation; visual context)
        self.lin_mm = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # attention linear layers
        self.att_linear_1 = nn.Linear(self.hidden_dim, self.attention_dim)
        self.att_linear_2 = nn.Linear(self.attention_dim, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize layers

    def use_batchnorm(self, use_bn):

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                if not use_bn:
                    m.eval()
                else:
                    m.train()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """

        self.embeddings.weight.data.uniform_(-0.1, 0.1)

        for ll in [
            self.linear_separate,
            self.lin_emb2hid,
            self.lin_emb2HIST,
            self.lin_context,
            self.lin_mm,
            self.att_linear_1,
            self.att_linear_2,
        ]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(
        self,
        text: torch.Tensor,
        separate_images: torch.Tensor,
        masks: torch.Tensor,
    ):

        """
        @param text: utterances to be converted into embeddings
        @param separate_images: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param masks: attention mask for pad tokens
        """

        text = text.to(self.device)
        separate_images = separate_images.to(self.device)
        visual_context = to_concat_context(separate_images)
        masks = masks.to(self.device)

        representations = self.embeddings(text)
        # [32,33,786]

        batch_size = representations.shape[0]  # effective batch size

        # utterance representations are processed
        representations = self.dropout(representations)
        input_reps = self.lrelu(self.lin_emb2hid(representations))
        input_reps = standardize(input_reps)

        # [32,33,512]

        # visual context is processed
        visual_context = self.dropout(visual_context)
        projected_context = self.relu(self.lin_context(visual_context))
        projected_context = standardize(projected_context)
        repeated_context = projected_context.unsqueeze(1).repeat(
            1, input_reps.shape[1], 1
        )

        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(torch.cat((input_reps, repeated_context), dim=2))
        )

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_1(mm_reps)
        outputs_att = self.relu(outputs_att)
        outputs_att = standardize(outputs_att)
        outputs_att = self.att_linear_2(outputs_att)
        # outputs_att = self.relu(outputs_att)

        # mask pads so that no attention is paid to them (with -inf)
        masks = masks.bool()
        outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = (mm_reps * att_weights).sum(dim=1)

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        separate_images = self.relu(separate_images)
        separate_images = standardize(separate_images)

        # dot product between the candid ate images and
        # the final multimodal representation of the input utterance
        dot = torch.bmm(
            separate_images, attended_hids.view(batch_size, self.hidden_dim, 1)
        )

        return dot
