import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, model_rng, emb_dim, num_classes, dropout=None):
        '''
        BiLSTM model for top-level intent classification
        tokenize_func takes a sentence string and returns a list of tokens for each word in the sentence,
        as well as the list of tokenized words. unknown word is given the token value == num-tokens
        '''
        super(BiLSTMModel, self).__init__()
        self.dropout = dropout
        self.model_rng = model_rng # TODO
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim,
                            batch_first=True, bidirectional=True)
        self.output = nn.Linear(emb_dim * 2, num_classes)


    def forward(self, x):
        device = next(self.parameters()).device
        # Count non-zero embeddings
        with torch.no_grad():
            lens = (x.sum(dim=-1) != 0).sum(dim=-1).to(device)
        embeddings_pack = pack_padded_sequence(x, torch.tensor(lens),
                                               batch_first=True, enforce_sorted=False)
        lstm_hidden_pack, (hidden, _) = self.lstm(embeddings_pack)

        # lstm_hidden, unpacked_lens = pad_packed_sequence(lstm_hidden_pack, batch_first=True)
        # avg_hiddens = []
        # for hidden, l in zip(lstm_hidden, lens):
        #     avg_hidden = hidden[:l].mean(dim=0)
        #     avg_hiddens.append(avg_hidden)
        # avg_hiddens = torch.stack(avg_hiddens, dim=0)
        avg_hiddens = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        if self.dropout is not None:
            avg_hiddens = F.dropout(avg_hiddens, self.dropout, training=self.training)
        logits = self.output(avg_hiddens)
        return logits
