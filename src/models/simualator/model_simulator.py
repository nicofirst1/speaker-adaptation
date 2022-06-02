import torch
import torch.nn.functional as F
from torch import nn

from src.models.listener.model_listener import ListenerModel


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
            speaker_embeds: torch.Tensor
    ):
        """
        @param speaker_embeds:embeddings coming from speaker

        """
        batch_size = speaker_embeds.shape[0]  # effective batch size

        # multimodal utterance representations
        mm_reps = self.relu(
            self.lin_mm(speaker_embeds)
        )

        # attention over the multimodal utterance representations (tokens and visual context interact)
        outputs_att = self.att_linear_2(self.tanh(self.att_linear_1(mm_reps)))

        # mask pads so that no attention is paid to them (with -inf)
        outputs_att = outputs_att.masked_fill_(masks, float("-inf"))

        # final attention weights
        att_weights = self.softmax(outputs_att)

        # encoder context representation
        attended_hids = (mm_reps * att_weights).sum(dim=1)

        # image features per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)

        # this is where we add history to candidates

        for b in range(batch_size):

            batch_prev_hist = prev_hist[b]

            for s in range(len(batch_prev_hist)):

                if len(batch_prev_hist[s]) > 0:
                    # print(batch_prev_hist[s])
                    prev = torch.Tensor(batch_prev_hist[s]).long().to(self.device)
                    hist_rep = self.embeddings(prev).to(self.device)
                    # if there is history for a candidate image
                    # hist_rep = torch.stack(batch_prev_hist[s]).to(self.device)

                    # take the average history vector
                    hist_avg = self.dropout(hist_rep.sum(dim=0) / hist_rep.shape[0])

                    # process the history representation and add to image representations
                    separate_images[b][s] += self.relu(self.lin_emb2HIST(hist_avg))

        # some candidates are now multimodal with the addition of history

        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # dot product between the candidate images and
        # the final multimodal representation of the input utterance
        dot = torch.bmm(
            separate_images, attended_hids.view(batch_size, self.hidden_dim, 1)
        )

        return dot
