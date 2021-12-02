#! /usr/bin/env python

import rdkit
from rdkit.Chem import Draw

def mol_to_image(mol: rdkit.Chem.Mol):
    img = Draw.MolToImage(mol)
    return img