# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
import torch
from torch.autograd import Variable
import torch.nn as nn


class SojaNet(nn.Module):

    def __init__(self, categorical, cat_size, numerical, device="cpu"):
        """

        :param categorical: define which features are categories
        """
        super(SojaNet, self).__init__()
        self.device = device
        self.categorical = categorical
        self.cat_size = cat_size
        self.extra_feat = numerical

        self.emb_size = 3

        embeddings = [nn.Embedding(cat_size[c], self.emb_size) for c in range(len(cat_size))]
        self.embeddings = nn.Sequential(*embeddings)

        self.embedding_size = self.emb_size * len(cat_size)

        # Etapa convolutional.
        self.conv_filters = [self.embedding_size + self.extra_feat]
        self.conv_filters = [32, 32]
        self.kernel = [3, 3]
        self.pooling = [2, 2]
        conv = []
      
        for k, n_filter in enumerate(self.conv_filters):
            if k == 0:
                n_filter_in = self.embedding_size+self.extra_feat
                n_filter_out = n_filter
            else:
                n_filter_in = n_filter_out
                n_filter_out = n_filter
                
            conv.append(nn.Conv1d(n_filter_in, n_filter_out, kernel_size=self.kernel[k]))
            conv.append(nn.ELU())
            conv.append(nn.BatchNorm1d(n_filter))
            if self.pooling[k] > 0:
                conv.append(nn.MaxPool1d(self.pooling[k]))
                
            n_filter_out = n_filter
        self.conv = nn.Sequential(*conv)
        
      
        # Etapa GRU.
        self.gru_directions = 1
        self.hidden_size = 10
        self.gru_layers = 2
        self.dropout = .5
        self.gru = torch.nn.GRU(input_size=self.conv_filters[-1], hidden_size=self.hidden_size,
                                batch_first=True, num_layers=self.gru_layers, dropout=self.dropout)

        # Etapa lineal.
        self.linear = nn.Linear(self.hidden_size, 1)

        lr = 1e-3
        self.optimizer = torch.optim.Adam([{"params": self.embeddings.parameters(), "lr": lr},
                                           {"params": self.conv.parameters(), "lr": lr},
                                           {"params": self.gru.parameters(), "lr": lr},
                                           {"params": self.linear.parameters(), "lr": lr}])

        self.h = torch.zeros((self.gru_directions*self.gru_layers, 32, self.hidden_size)).to(self.device)
        
    def forward(self, x):
        """
        x: Batch x Seq x Feat
        lenghts: Batch x L 
        """
        batch = x.shape[0]
        
        # Embedding
        x1 = torch.zeros((x.shape[0], x.shape[1], self.embedding_size+self.extra_feat)).to(self.device)
        c = 0
        for k, emb in enumerate(self.embeddings):
            x1[:, :, c: c+self.emb_size] = emb(x[:, :, self.categorical[k]].long())
            c += self.emb_size

        # Features numéricas
        for f in range(x.shape[2]):
            if f not in self.categorical:
                x1[:, :, c] = x[:, :, f].float()
                c += 1

        # Etapa convolucional.
        x1 = self.conv(x1.transpose(1, 2)).transpose(1, 2)

        # Etapa GRU
        h = Variable(self.h)
        y, h = self.gru(x1, h)
       
        out = self.linear(y[:, -10:, :]).squeeze()
        return out
