#! /usr/bin/env python

import dgl
import dgl.function as fn
import torch
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
        self.fc_bond = nn.Linear(prev, natom*natom*num_bond_typ)
        self.natom = natom
        self.num_atom_typ = num_atom_typ
        self.num_bond_typ = num_bond_typ

    def forward(self, bs=32, tau=1.0):

        x = torch.randn(bs, self.nhidden)
        for mod in self.mod:
            x = mod(x)
        atom = self.fc_atom(x)
        bond = self.fc_bond(x)\
            .view(-1, self.natom, self.natom, self.num_bond_typ)

        if self.training:
            atom = F.gumbel_softmax(atom, tau=tau, hard=True)
            bond = F.gumbel_softmax(bond, tau=tau, hard=True)

            # Build a molecular graph

        else:
            pass

class MolDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

if __name__ == "__main__":
    pass