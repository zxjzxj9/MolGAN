#! /usr/bin/env python

import argparse
import toml

from model import MolGen, MolDis

args = argparse.ArgumentParser("MolGAN inference script")
args.add_argument("-c", "--conf", type=str, default="hparam.toml", help="Config file location")
opt = args.parse_args()

if __name__ == "__main__":
    with open("hparam.toml", "r") as fin:
        conf = toml.load(args.conf)

    use_cuda = conf["use_cuda"]

    model = {
        "gen": MolGen(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"],
                      conf["num_hidden"], conf["gen_layer"]),
        "dis": MolDis(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"])
    }