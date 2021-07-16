#! /usr/bin/env python

from model import MolGen, MolDis
import toml

def train(model, niter):
    pass

def test(model):
    pass

if __name__ == "__main__":
    with open("hparam.toml", "r") as fin:
        conf = toml.load(fin)

    model = {
        "gen": None,
        "dis": None
    }