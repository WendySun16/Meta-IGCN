import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class Meta_GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Meta_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = nn.ParameterList()

        weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
        if bias is not None:
            torch.nn.init.constant_(bias, 0.0)
        self.v.append(weight)
        self.v.append(bias)

    def forward(self, input, adj, v=None):
        if v is None:
            v = self.v
        support = torch.mm(input, v[0])
        output = torch.spmm(adj, support)
        if v[1] is not None:
            return output + v[1]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Meta_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Meta_GCN, self).__init__()

        self.gc1 = Meta_GraphConvolution(nfeat, nhid)
        self.gc2 = Meta_GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, vars=None):  # vars: [w1, b1, w2, b2]
        if vars is not None:
            v = [vars[i:i+2] for i in range(0, len(vars), 2)]  # v: [[w1, b1], [w2, b2]]
            x = F.relu(self.gc1(x, adj, v[0]))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj, v[1])
        else:
            x = F.relu(self.gc1(x, adj, None))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj, None)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, npre):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, npre)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, npre, nhid, nout):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(npre, nhid)
        self.linear2 = nn.Linear(nhid, nout)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, nfeat, nhid, npre, nout):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(nfeat, nhid, npre)
        self.decoder = Decoder(npre, nhid, nout)

    
    def forward(self, x):
        low_feat = self.encoder(x)
        high_feat = self.decoder(low_feat)
        return low_feat, high_feat


class SE(nn.Module):
    def __init__(self, nfeat, ratio):
        super(SE, self).__init__()

        self.linear1 = nn.Linear(nfeat, nfeat // ratio)
        self.linear2 = nn.Linear(nfeat // ratio, nfeat)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x