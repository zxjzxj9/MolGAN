#! /usr/bin/env python

# import dgl
# import dgl.function as fn
import torch
# from dgl.nn.pytorch import RelGraphConv
import torch.nn as nn
import torch.nn.functional as F

# should refer to the following link for details
# https://github.com/nicola-decao/MolGAN/blob/master/utils/layers.py
from torch.utils.data import DataLoader

from datareader import QM9CSVDataset


class RGCN(nn.Module):
    def __init__(self, ninput, noutput, edge_type):
        super().__init__()
        self.fc = nn.Linear(ninput, noutput)
        self.edge_type = edge_type # Not include non-bond type
        self.edge_mod = nn.ModuleList([nn.Linear(ninput, noutput) for _ in range(edge_type)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, node, edge, hidden=None):
        # Remove non-bonding edge: bsxnaxnaxtype
        edge = edge[..., 1:]
        # print(edge.shape)
        # print(node.shape, hidden.shape if hidden is not None else None)
        if hidden is not None: node = torch.cat([node, hidden], -1)
        # bsxnaxhiddenxtype
        x = torch.stack([mod(node) for mod in self.edge_mod], -1)
        node = self.fc(node)
        # print(x.shape)
        x = torch.einsum("bijk,bjlk->bil", edge, x)
        # print(x.shape, node.shape)
        x = F.relu(x + node)
        x = self.dropout(x)
        return x

class GraphAggr(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.fc1 = nn.Linear(ninput, noutput)
        self.fc2 = nn.Linear(ninput, noutput)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node):
        # node: bs x na x feat
        x = self.fc1(node).sigmoid()
        y = self.fc2(node).relu()
        x = (x*y).sum(1).relu()
        x = self.dropout(x)
        return x

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
        self.fc_atom = nn.Linear(prev, natom*num_atom_typ)
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
            atom = atom.view(-1, self.natom, self.num_atom_typ)
            bond = bond.view(-1, self.natom, self.natom, self.num_bond_typ)
            return atom, bond
        else:
            atom = atom.view(-1, self.natom, self.num_atom_typ).softmax(-1)
            bond = bond.view(-1, self.natom, self.natom, self.num_bond_typ).softmax(-1)
            return atom, bond

class MolDis(nn.Module):
    def __init__(self, natom, num_atom_typ, num_bond_typ):
        super().__init__()
        self.natom = natom
        self.num_atom_typ = num_atom_typ
        self.num_bond_type = num_bond_typ
        self.layer1 = RGCN(num_atom_typ-1, 32, num_bond_typ-1)
        self.agg1 = GraphAggr(32, 32)
        self.layer2 = RGCN(32 + num_atom_typ-1, 64, num_bond_typ-1)
        self.agg2 = GraphAggr(64, 64)
        self.agg3 = GraphAggr(64, 1)

    def forward(self, node, edge):
        node = node[:, :, 1:] # remove none atoms
        x = self.layer1(node, edge)
        # x = self.agg1(x)
        x = self.layer2(node, edge, x)
        # x = self.agg2(x)
        x = self.agg3(x).squeeze()
        return x

if __name__ == "__main__":
    gen = MolGen(20, 8, 5, 32, [128, 256, 512])
    dis = MolDis(20, 8, 5)
    atom, bond = gen()
    print(atom.shape, bond.shape)
    ret = dis(atom, bond)
    print(ret.shape)
    print(ret.sigmoid())

    ds = QM9CSVDataset("./datafolder/qm9.csv")
    dl = DataLoader(ds, 20, shuffle=True, num_workers=0, pin_memory=True)
    for atom_d, bond_d in dl:
        print(atom_d.shape)
        print(bond_d.shape)
        atom_g, bond_g = gen()
        print(atom_g.shape)
        print(bond_g.shape)
        ret_g = dis(atom_g, bond_g)
        ret_d = dis(atom_d, bond_d)
        print(ret_g)
        print(ret_d)
        break