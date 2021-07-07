#! /user/bin/env python

import tarfile
from io import StringIO

import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import rdkit

from xyz2mol import read_xyz_file, read_qm9_xyz, xyz2AC, xyz2mol

bond_map = {
    rdkit.Chem.BondType.ZERO: 0,
    rdkit.Chem.BondType.SINGLE: 1,
    rdkit.Chem.BondType.DOUBLE: 2,
    rdkit.Chem.BondType.TRIPLE: 3,
    rdkit.Chem.BondType.AROMATIC: 4
}

nbond = len(bond_map)

atom_map = {
    "UNK": 0,
    "C": 1,
    "H": 2,
    "O": 3,
    "F": 4,
    "N": 5,
    "P": 6,
    "S": 7
}

natom = len(atom_map)

def mol_to_graph(mol: rdkit.Chem.Mol, max_atom = 16):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    nodes = [atom_map[atom] if atom in atom_map else 0 for atom in atoms]
    nodes = nodes + [0]*(max_atom - len(nodes))
    nodes = F.one_hot(nodes, len(atom_map))
    nodes = nodes[:max_atom]
    atom = nodes
    feat = torch.zeros(max_atom, max_atom, dtype=torch.int32)
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat[end, start] = bond_map[bond.GetBondType()]
        feat[start, end] = bond_map[bond.GetBondType()]
    feat = F.one_hot(feat, len(bond_map))
    bond = feat
    return atom, bond

def graph_to_mol(atom, bond):
    pass

class QM9BZ2Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.fp = tarfile.open(self.filename, "r:bz2")
        # filter out .xyz file
        self.file_info = [fn for fn in self.fp if fn.name.endswith(".xyz")]

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        info = self.file_info[idx]
        fin = self.fp.extractfile(info)
        xyz = fin.read().decode("ascii")
        # print(xyz)
        atoms, charge, xyz_coordinates = read_qm9_xyz(StringIO(xyz))
        # conn_mat, mol = xyz2AC(atoms, xyz_coordinates, charge)
        mol: rdkit.Chem.Mol = xyz2mol(atoms, xyz_coordinates)[0]

        graph = dgl.DGLGraph()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        nodes = [atom_map[atom] if atom in atom_map else 0 for atom in atoms]
        # print(nodes)
        # graph.add_nodes(mol.GetNumAtoms(), )
        graph.add_nodes(len(nodes), {'x': torch.tensor(nodes)})
        bond: rdkit.Chem.Bond
        start = []
        end = []
        feat = []
        for bond in mol.GetBonds():
            start.append(bond.GetBeginAtomIdx())
            end.append(bond.GetEndAtomIdx())
            if bond.GetBondType() in bond_map:
                feat.append(bond_map[bond.GetBondType()])
            else:
                feat.append(0)
        start = torch.tensor(start)
        end = torch.tensor(end)
        feat = F.one_hot(torch.tensor(feat), len(bond_map))
        graph.add_edges(start, end, {'h': feat})
        return graph

if __name__ == "__main__":
    qmd = QM9BZ2Dataset("./datafolder/dsgdb9nsd.xyz.tar.bz2")
    mol: rdkit.Chem.Mol = qmd[10]
    print(rdkit.Chem.MolToSmiles(mol))