#! /usr/bin/env python

import dgl
import dgl.function as fn
import torch
from dgl.nn.pytorch import RelGraphConv
import torch.nn as nn
import torch.nn.functional as F

# should refer to the following link for details
# https://github.com/nicola-decao/MolGAN/blob/master/utils/layers.py

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
        if hidden is not None: node = torch.cat([node, hidden], -1)
        # bsxnaxhiddenxtype
        x = torch.stack([mod(edge) for mod in self.edge_mod], -1)
        x = torch.einsum("bijk,bjlk->bil", edge, x)
        x = F.relu(x + node)
        x = self.dropout(x)
        return x

class GraphAggr(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.fc1 = nn.Linear(ninput, noutput)
        self.fc2 = nn.Linear(ninput, noutput)
        self.dropout = nn.Dropout(0.1)

    def forwar(self, node):
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
            print(atom.shape)

            # Build a molecular graph
            graph = dgl.DGLGraph()
            graph.add_nodes(bs*self.natom, {'x': atom.view(-1, self.num_atom_typ)})

            start, end = torch.meshgrid(torch.arange(self.natom), torch.arange(self.natom))
            start = torch.cat(bs*[start]).flatten()
            end = torch.cat(bs*[end]).flatten()
            print(start.shape, end.shape)

            _, edges = bond.view(-1, self.num_bond_typ).max(-1)
            graph.add_edges(start, end, {'h': edges})

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
        self.layer1 = RelGraphConv(num_atom_typ, 32, num_bond_typ, self_loop=False)
        self.layer2 = RelGraphConv(num_atom_typ, 64, num_bond_typ, self_loop=False)

    def forward(self, g, bs=32):
        x = self.layer1(g, g.ndata['x'], g.edata['h'])
        x = F.relu(x)
        x = self.layer1(g, x, g.edata['h'])
        x = F.relu(x)
        # How to aggregate it?
        xs = x.split(bs, dim=0)
        vs = torch.stack([x.mean(dim=0) for x in xs], dim=0)
        return vs

if __name__ == "__main__":
    gen = MolGen(20, 8, 5, 32, [128, 256, 512])
    dis = MolDis(20, 8, 5)
    mol = gen()
    print(mol.ndata['x'].shape)
    print(mol.edata['h'].shape)
    ret = dis(mol)
    # print(ret)