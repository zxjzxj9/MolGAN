#! /usr/bin/env python

import argparse

args = argparse.ArgumentParser("MolGAN inference script")
args.add_argument("-c", "--config", type=str, default="hparam.toml", help="Config file location")
opt = args.parse_args()

if __name__ == "__main__":
    pass