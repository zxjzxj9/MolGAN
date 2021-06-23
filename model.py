#! /usr/bin/env python

import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch import RelGraphConv
import torch.nn as nn
import torch.nn.functional as F

class MolGen(nn.Module):
    def __init__(self, natom, num_atom_typ, num_bond_typ, nhidden, nfeats):
        super().__init__()
        # self.atom_embedding = nn.Embedding(num_atom_typ, nhidden)
        self.mod = nn.ModuleList([])
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
            graph = dgl.DGLGraph()
            graph.add_nodes(bs*self.natom, {'x': atom})

            start, end = torch.meshgrid(torch.arange(self.natom), torch.arange(self.natom))
            start = torch.cat(bs*[start]).flatten()
            end = torch.cat(bs*[end]).flatten()
            graph.add_edges(start, end, {'h': bond.view(-1, bond.shape[-1])})

            # indeed we are generating a batch of graphs
            return graph
        else:
            pass

class MolDis(nn.Module):
    def __init__(self, natom, num_atom_typ, num_bond_typ):
        super().__init__()
        self.natom = natom
        self.num_atom_typ = num_atom_typ
        self.num_bond_type = num_bond_typ
        self.layer1 = RelGraphConv(num_atom_typ, 32, num_bond_typ)
        self.layer2 = RelGraphConv(num_atom_typ, 64, num_bond_typ)

    def forward(self, g, bs=32):
        x = self.layer1(g, g['x'], g['h'])
        x = F.relu(x)
        x = self.layer1(g, x, g['h'])
        x = F.relu(x)
        # How to aggregate it?
        xs = x.split(bs, dim=0)
        vs = torch.stack([x.mean(dim=0) for x in xs], dim=0)
        return vs

if __name__ == "__main__":
    gen = MolGen(20, 8, 5, 32, [128, 256, 512])
    print(gen())