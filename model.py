#! /usr/bin/env python

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F

class MolGen(nn.Module):
    def __init__(self, natom, num_atom_typ, num_bond_typ, nhidden, nfeats):
        super().__init__()
        # self.atom_embedding = nn.Embedding(num_atom_typ, nhidden)
        self.mod = nn.Module([])
        self.nhidden = nhidden
        prev = nhidden
        for nfeat in nfeats:
            self.mod.append(nn.Linear(prev, nfeat))
            self.mod.append(nn.ReLU())
            prev = nfeat
        self.fc_atom = nn.Linear(prev, num_atom_typ)
        self.fc_mol = nn.Linear(prev, natom*num_bond_typ)

    def forward(self):
        pass

class MolDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

if __name__ == "__main__":
    pass