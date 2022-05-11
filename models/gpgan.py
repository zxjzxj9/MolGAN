#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class MolGenerator(nn.Module):
    def __init__(self, nlatent, nhiddens, dropout, adj_shape, feat_shape):
        super().__init__()

        self.nlatent = nlatent
        nlayers = [nlatent, *nhiddens]
        mods = []
        for nin, nout in zip(nlayers[:-1], nlayers[1:]):
            mods.append(nn.Linear(nin, nout))
            mods.append(nn.Tanh())
            mods.append(nn.Dropout(dropout))

        self.base = nn.Sequential(*mods)
        self.adj = nn.Linear(nlayers[-1], np.prod(adj_shape))
        self.adj_shape = adj_shape
        self.feat = nn.Linear(nlayers[-1], np.prod(feat_shape))
        self.feat_shape = feat_shape

    def forward(self, x):
        h = self.base(x)
        adj = self.adj(h).view(-1, *self.adj_shape)
        adj = 0.5*(adj + adj.transpose(-1, -2))
        adj = adj.softmax(1)
        feat = self.feat(h).view(-1, *self.feat_shape)
        feat = feat.softmax(2)
        return adj, feat


class RGConv(nn.Module):
    def __init__(self, nbond, nin, nout, act=F.relu, bias=False):
        super().__init__()
        self.act = act

        self.weight = nn.Parameter(torch.zeros(nbond, nin, nout))
        self.bias = None
        if bias:
            nn.Parmeter(torch.zeros(nbond, 1, nout))
        self._init_param()

    def _init_param(self):
        nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain(self.act.__name__))

    def forward(self, adj, feat):
        x = torch.einsum("bijk,bkl->bijl", adj, feat)
        x = torch.einsum("bijk,ikl->bijl", x, self.weight)
        if self.bias is not None:
            x += self.bias
        x = self.act(x.sum(1))
        return x


class MolDiscriminator(nn.Module):
    def __init__(self, ngcn, nfc, dropout, adj_shape, feat_shape):
        super().__init__()

        nbond = adj_shape[0]
        nfeat = feat_shape[-1]

        nlayers = [nbond, *ngcn]
        self.gcn_mods = nn.ModuleList()
        for nin, nout in zip(nlayers[:-1], nlayers[1:]):
            self.gcn_mods.append(RGConv(nbond, nin, nout))

        nlayers = [nout, *nfc]
        mods = []
        for nin, nout in zip(nlayers[:-1], nlayers[1:]):
            mods.append(nn.Linear(nin, nout))
            mods.append(nn.ReLU())
            mods.append(nn.Dropout(dropout))

        mods.append(nn.Linear(nout, 1))
        mods.append(nn.Flatten())
        self.fc = nn.Sequential(*mods)

    def forward(self, adj, feat):
        x = feat
        for mod in self.gcn_mods:
            x = mod(adj, x)
        x = x.mean(1)
        x = self.fc(x)
        return x