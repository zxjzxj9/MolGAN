#! /usr/bin/env python

import argparse

import rdkit.Chem
import toml
import torch

from datareader import graph_to_mol
from model import MolGen, MolDis

args = argparse.ArgumentParser("MolGAN inference script")
args.add_argument("-c", "--conf", type=str, default="hparam.toml", help="Config file location")
args.add_argument("-m", "--model", type=str, default="model.pt", help="Checkpoint model file")
args.add_argument("-b", "--bs", type=int, default=16, help="Inference batch size")
args.add_argument("-t", "--tau", type=float, default=0.001, help="Inference temperature")
opt = args.parse_args()

if __name__ == "__main__":
    with open("hparam.toml", "r") as fin:
        conf = toml.load(opt.conf)
    use_cuda = conf["use_cuda"]

    model = {
        "gen": MolGen(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"],
                      conf["num_hidden"], conf["gen_layer"]),
        "dis": MolDis(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"])
    }

    sd = torch.load(opt.model)
    model["gen"].load_state_dict(sd["gen"])
    model["dis"].load_state_dict(sd["dis"])

    for v in model.values():
        v.eval()

    atom_g, bond_g = model["gen"](opt.bs, opt.tau)

    for idx in range(opt.bs):
        atom_t = atom_g[idx, ...].cpu()
        bond_t = bond_g[idx, ...].cpu()
        mol_t = graph_to_mol(atom_t, bond_t)
        print(rdkit.Chem.MolToSmiles(mol_t))

