import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

import math

class GraphConvolution(Module) :
    def __init__(self, input_dim, output_dim) :
        super(GraphConvolution, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self) :
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj) :
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        return output + self.bias

class GCN(nn.Module) :
    def __init__(self, n_features, n_hidden, n_class, dropout) :
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout
    
    def forward(self, x, adj) :
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)