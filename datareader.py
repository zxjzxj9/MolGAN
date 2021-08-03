#! /user/bin/env python

import tarfile
from io import StringIO

# import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# import numpy as np
import rdkit

from xyz2mol import read_xyz_file, read_qm9_xyz, xyz2AC, xyz2mol

bond_map = {
    rdkit.Chem.BondType.ZERO: 0,
    rdkit.Chem.BondType.SINGLE: 1,
    rdkit.Chem.BondType.DOUBLE: 2,
    rdkit.Chem.BondType.TRIPLE: 3,
    rdkit.Chem.BondType.AROMATIC: 4
}
bondtyp_map = {v:k for k, v in bond_map.items()}

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

serial_map = {v: k for k, v in atom_map.items()}
natom = len(atom_map)

def mol_to_graph(mol: rdkit.Chem.Mol, max_atom = 16):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    nodes = [atom_map[atom] if atom in atom_map else 0 for atom in atoms]
    nodes = torch.tensor(nodes + [0]*(max_atom - len(nodes)))
    # print(nodes)
    nodes = F.one_hot(nodes, len(atom_map))
    nodes = nodes[:max_atom]
    atom = nodes
    feat = torch.zeros(max_atom, max_atom, dtype=torch.int64)
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat[end, start] = bond_map[bond.GetBondType()]
        feat[start, end] = bond_map[bond.GetBondType()]
    feat = F.one_hot(feat, len(bond_map))
    bond = feat
    return atom, bond

def graph_to_mol(atom, bond):
    # atom: naxfeat
    # bond: naxnaxfeat
    mol = rdkit.Chem.RWMol()
    atom = atom.argmax(-1)
    bond = bond.argmax(-1)
    valid_atoms = {} # graph -> mol
    cnt = 0
    for idx, val in enumerate(list(atom)):
        if val > 0:
            valid_atoms[idx] = cnt
            cnt += 1
            mol.AddAtom(rdkit.Chem.Atom(serial_map[val.item()]))
    vk = valid_atoms.keys()
    for i in range(len(vk)):
        for j in range(0, i):
            if bond[i][j] > 0:
                mol.AddBond(valid_atoms[i], valid_atoms[j],
                            bondtyp_map[bond[i][j].item()])
    return mol

class QM9BZ2Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.fp = tarfile.open(self.filename, "r")
        # filter out .xyz file
        self.file_info = [fn for fn in self.fp if fn.name.endswith(".xyz")]

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        info = self.file_info[idx]
        print(info)
        fin = self.fp.extractfile(info)
        xyz = fin.read().decode("ascii")
        fin.close()
        # print(xyz)
        atoms, charge, xyz_coordinates = read_qm9_xyz(StringIO(xyz))
        # conn_mat, mol = xyz2AC(atoms, xyz_coordinates, charge)
        mol: rdkit.Chem.Mol = xyz2mol(atoms, xyz_coordinates)[0]
        # print(rdkit.Chem.MolToSmiles(mol))
        return mol_to_graph(mol)

if __name__ == "__main__":
    qmd = QM9BZ2Dataset("./datafolder/dsgdb9nsd.xyz.tar")
    # mol: rdkit.Chem.Mol = qmd[10]
    # print(rdkit.Chem.MolToSmiles(mol))
    atom, bond = qmd[12]
    # print(atom, bond)
    mol = graph_to_mol(atom, bond)
    print(rdkit.Chem.MolToSmiles(mol))