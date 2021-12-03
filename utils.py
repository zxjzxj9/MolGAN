#! /usr/bin/env python

import rdkit
from rdkit.Chem import Draw
from torch.utils.tensorboard import SummaryWriter
from typing import Sequence
import numpy as np

def mol_to_image(writer: SummaryWriter, mols: Sequence[rdkit.Chem.Mol], tag, niter):
    imgs = [Draw.MolToImage(mol) for mol in mols]
    img_tensor = np.stack([np.array(imgs)], axis=0)
    writer.add_image(img_tensor=img_tensor, tag=tag, global_step=niter)