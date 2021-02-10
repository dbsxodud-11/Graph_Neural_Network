import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(GCN, self).__init__()
        self.fc = nn.Linear(n_features, n_hidden)
        self.activ = nn.PReLU() 
        self.init_weight()

    def init_weight(self) :
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, x, adj):
        x = self.fc(x)
        x = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        x = self.activ(x)
        return x

class ReadOut(nn.Module) :
    def __init__(self) :
        super(ReadOut, self).__init__()
    
    def forward(self, h) :
        return torch.mean(h, dim=1)

class Discriminator(nn.Module) :
    def __init__(self, n_h) :
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.init_weight()

    def init_weight(self) :
        torch.nn.init.xavier_normal_(self.f_k.weight.data)

    def forward(self, s, h_1, h_2) :
        expanded_s = torch.unsqueeze(s, 1).expand_as(h_1)
        score_1 = torch.squeeze(self.f_k(h_1, expanded_s), 2)
        score_2 = torch.squeeze(self.f_k(h_2, expanded_s), 2)
        logits = torch.cat([score_1, score_2], dim=1)
        return logits

class DGI(nn.Module) :
    def __init__(self, n_features, n_hidden) :
        super(DGI, self).__init__()
        self.gcn = GCN(n_features, n_hidden)
        self.readout = ReadOut()
        self.sigmoid = nn.Sigmoid()
        self.discriminator = Discriminator(n_hidden)

    def forward(self, x_1, x_2, adj) :  
        h_1 = self.gcn(x_1, adj)
        h_2 = self.gcn(x_2, adj)

        s = self.readout(h_1)
        s = self.sigmoid(s)
        score = self.discriminator(s, h_1, h_2)
        return score
    
    def embed(self, h, adj) :
        h = self.gcn(h, adj)
        s = self.readout(h)
        return h.detach()

class LogisticRegression(nn.Module) :
    def __init__(self, n_hidden, n_class) :
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(n_hidden, n_class)
        self.init_weight()

    def init_weight(self) :
        torch.nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, h) :
        return self.fc(h)