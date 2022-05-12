#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActNorm2d(nn.Module):

    def __int__(self):
        super().__init__()

    def forward(self, x):
        return x