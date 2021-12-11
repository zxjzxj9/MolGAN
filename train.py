#! /usr/bin/env python

from model import MolGen, MolDis
import toml
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datareader import QM9CSVDataset, graph_to_mol
from torch.utils.data import DataLoader
import numpy as np
import rdkit

writer = SummaryWriter("./log")

def train(data, model, opt, niter, bs=32, tau=1.0):
    for v in model.values():
        v.train()

    for atom_d, bond_d in data:
        # Do temperature decay
        if niter % 1000 == 1: tau *= 1.1
        niter += 1
        # First optimize G
        opt["gen"].zero_grad()
        atom_g, bond_g = model["gen"](bs, tau)
        logit_g = model["dis"](atom_g, bond_g)
        loss = -torch.mean(F.logsigmoid(logit_g))
        writer.add_scalar("Gen Loss", loss.item())
        loss.backward()
        opt["gen"].step()
        gen_loss = loss.item()
        # print("gen:", loss.item())

        # Then optimize D
        opt["dis"].zero_grad()
        atom_g, bond_g = model["gen"](bs, tau)
        logit_g = model["dis"](atom_g, bond_g)
        logit_d = model["dis"](atom_d.float(), bond_d.float())
        # print(logit_g, logit_d)
        loss = -torch.mean(F.logsigmoid(logit_d) + (1 - logit_g.sigmoid()).log())
        dis_loss = loss.item()
        writer.add_scalar("Dis Loss", loss.item())
        loss.backward()
        # print("dis:", loss.item())
        print("In iteration {:6d}, Gen Loss {:12.6f}, Dis Loss {:12.6f}".format(niter, gen_loss, dis_loss), end='\r')
        opt["dis"].step()
    return niter

# For GAN, we have no test, just generate the molecule
def test(model, niter, bs=32):
    gen = model["gen"]
    gen.eval()
    with torch.no_grad():
        atom_g, bond_g = model["gen"](bs, tau)

        # write mols to TF Board
        imgs = []
        for idx in range(conf["batch_size"]):
            atom_t = atom_g[idx, ...]
            bond_t = bond_g[idx, ...]
            mol_t = graph_to_mol(atom_t, bond_t)
            imgs.append(rdkit.Chem.Draw.MolToImage(mol_t))
        img_tensor = np.stack([np.array(imgs)], axis=0)
        writer.add_image(img_tensor=img_tensor, tag=f"mol_{idx:%04d}", global_step=niter)

    return atom_g, bond_g

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

    ds = QM9CSVDataset(conf["data_path"])
    dl = DataLoader(ds, conf["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    niter = 0
    tau = 1.0
    for epoch in range(conf["nepoch"]):
        print("In epoch {:4d}".format(epoch+1))
        print("Training Stage...")
        niter = train(dl, model, optimizer, niter, conf["batch_size"], tau)
        tau *= 0.9
        print("Saving model checkpoints...")
        torch.save(f"model_{epoch:04d}.pt", {k: v.state_dict() for k, v in model.items()})
        print("Done.")
        print("Testing Stage...")
        test(model, epoch)