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
use_cuda = False

def train(data, model, opt, niter, bs=32, tau=1.0):
    for v in model.values():
        v.train()

    for atom_d, bond_d in data:
        # Add cuda data conversion code
        if use_cuda:
            atom_d = atom_d.cuda()
            bond_d = bond_d.cuda()

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
        for idx in range(bs):
            atom_t = atom_g[idx, ...].cpu()
            bond_t = bond_g[idx, ...].cpu()
            mol_t = graph_to_mol(atom_t, bond_t)
            imgs.append(np.array(rdkit.Chem.Draw.MolToImage(mol_t)))
            writer.add_text(text_string=rdkit.Chem.MolFromSmiles(mol_t), tag=f"molecule_{idx:04d}", global_step=niter)
        img_tensor = np.stack(imgs, axis=0)
        writer.add_images(img_tensor=img_tensor, tag="molecule", global_step=niter, dataformats="NHWC")

    return atom_g, bond_g


if __name__ == "__main__":
    with open("hparam.toml", "r") as fin:
        conf = toml.load(fin)

    use_cuda = conf["use_cuda"]

    model = {
        "gen": MolGen(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"],
                      conf["num_hidden"], conf["gen_layer"]),
        "dis": MolDis(conf["num_atom"], conf["num_atom_type"], conf["num_bond_type"])
    }

    if use_cuda:
        for v in model.values(): v.cuda()

    if conf["ckpt_path"] != "":
        print(f"Load ckpt file from: {conf['ckpt_path']}...")
        ckpt = torch.load(conf['ckpt_path'])
        model["gen"].load_state_dict(ckpt["gen"])
        model["dis"].load_state_dict(ckpt["dis"])

    optimizer = {
        "gen": torch.optim.Adam(params=model["gen"].parameters(), lr=conf["learning_rate"]),
        "dis": torch.optim.Adam(params=model["dis"].parameters(), lr=conf["learning_rate"])
    }

    ds = QM9CSVDataset(conf["data_path"])
    dl = DataLoader(ds, conf["batch_size"], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    niter = 0
    tau = 1.0
    for epoch in range(conf["nepoch"]):
        print("In epoch {:4d}".format(epoch + 1))
        print("Training Stage...")
        niter = train(dl, model, optimizer, niter, conf["batch_size"], tau)
        tau *= 0.9
        print("")
        print("Saving model checkpoints...")
        torch.save({k: v.state_dict() for k, v in model.items()}, f"model_{epoch:04d}.pt")
        print("Done.")
        print("Testing Stage...")
        print(model['gen'])
        test(model, epoch, bs=32)
