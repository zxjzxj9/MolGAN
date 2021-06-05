#! /usr/bin/env python

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F

class MolGen(nn.Module):
    def __init__(self, num_atom_typ, nhidden):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_typ, nhidden)

    def forward(self):
        pass

class MolDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

if __name__ == "__main__":
    pass