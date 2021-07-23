#! /usr/bin/env python

from model import MolGen, MolDis
import toml
import torch
from datareader import QM9BZ2Dataset
from torch.utils.data import DataLoader

def train(data, model, opt, niter, bs=32, tau=1.0):
    for atom_d, bond_d in data:
        opt["gen"].zero_grad()
        atom_g, bond_g = model["gen"](bs, tau)
        opt["dis"].zero_grad()

    return niter + 1

# For GAN, we have no test, just generate the molecule
def test(data, model):
    pass

if __name__ == "__main__":
    with open("hparam.toml", "r") as fin:
        conf = toml.load(fin)

    model = {
        "gen": MolGen(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"],
                      conf["num_hidden"], conf["gen_layer"]),
        "dis": MolDis(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"])
    }

    optimizer = {
        "gen": torch.optim.Adam(params=model["gen"].parameters(), lr=conf["learning_rate"]),
        "dis": torch.optim.Adam(params=model["dis"].parameters(), lr=conf["learning_rate"])
    }

    ds = QM9BZ2Dataset(conf["data_path"])
    dl = DataLoader(ds, conf["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

    niter = 0
    for epoch in range(conf["nepoch"]):
        print("In epoch %d", epoch+1)
        niter = train(dl, model, optimizer, niter)