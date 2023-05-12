import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from classifiers.seeded_layers import SeededEmbedding, SeededLinear, SeededLSTM

class BiLSTMModel(nn.Module):
    def __init__(self, model_rng, embedding_data:torch.Tensor, num_classes, dropout=None):
        '''
        BiLSTM model for top-level intent classification
        '''
        super(BiLSTMModel, self).__init__()
        self.dropout = dropout
        self.model_rng = model_rng
        num_emb, emb_dim = embedding_data.size()
        self.pad_idx = num_emb - 1
        self.emb_dim = emb_dim
        self.word_embedding = SeededEmbedding.from_pretrained(model_rng,
                                                              embedding_data, freeze=True,
                                                              padding_idx=self.pad_idx)

        self.lstm = SeededLSTM(model_rng,
                               input_size=emb_dim, hidden_size=emb_dim,
                               batch_first=True, bidirectional=True)
        self.output = SeededLinear(model_rng, emb_dim * 2, num_classes)


    def _encode(self, x):
        device = next(self.parameters()).device
        # Count non-zero embeddings
        with torch.no_grad():
            num_of_pad = (x == self.pad_idx).int().sum(dim=-1)
            lens = torch.ones(len(x)) * x.size(-1)
            num_of_pad = num_of_pad.to(device)
            lens = lens.to(device)
            lens -= num_of_pad
        x = self.word_embedding(x)
        embeddings_pack = pack_padded_sequence(x, lens,
                                               batch_first=True, enforce_sorted=False)
        lstm_hidden_pack, (hidden, _) = self.lstm(embeddings_pack)

        # lstm_hidden, unpacked_lens = pad_packed_sequence(lstm_hidden_pack, batch_first=True)
        # avg_hiddens = []
        # for hidden, l in zip(lstm_hidden, lens):
        #     avg_hidden = hidden[:l].mean(dim=0)
        #     avg_hiddens.append(avg_hidden)
        # avg_hiddens = torch.stack(avg_hiddens, dim=0)
        avg_hiddens = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return avg_hiddens

    def forward(self, x):
        avg_hiddens = self._encode(x)
        if self.dropout is not None:
            avg_hiddens = F.dropout(avg_hiddens, self.dropout, training=self.training)
        logits = self.output(avg_hiddens)
        return logits
