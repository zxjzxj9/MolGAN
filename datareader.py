#! /user/bin/env python

import tarfile
from io import StringIO

from torch.utils.data import Dataset
import numpy as np
import rdkit

from xyz2mol import read_xyz_file, read_qm9_xyz, xyz2AC


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
        conn_mat, mol = xyz2AC(atoms, xyz_coordinates, charge)
        return mol

if __name__ == "__main__":
    qmd = QM9BZ2Dataset("./datafolder/dsgdb9nsd.xyz.tar.bz2")
    mol: rdkit.Chem.Mol = qmd[12]
    print(rdkit.Chem.MolToSmiles(mol))