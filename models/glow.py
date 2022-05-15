#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

def squeeze2d(x, factor):
    n, c, h, w = x.shape
    x = x.view(n, c, h//factor, factor, w//factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(n, c*factor*factor, h//factor, w//factor)
    return x

def unsqueeze(x, factor):
    n, c, h, w = x.shape
    x = x.view(n, c//factor**2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(n, c//factor**2, h*factor, w*factor)
    return x

class ActNorm2d(nn.Module):

    def __int__(self, num_features, scale=1.0):
        super().__init__()

        # nchw
        size = (1, 1, num_features, 1)
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def init_param(self, x):
        with torch.no_grad():
            bias = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((x + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias)
            self.logs.data.copy_(logs)
        self.inited = True

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, logdet=None, reverse=False):
        if reverse:
            x = x * self.logs.exp()
        else:
            x = x * self.logs.neg().exp()

        if logdet is not None:
            n, c, h, w = x.shape
            dlogdet = self.logs.sum() * h * w
            if reverse:
                dlogdet *= -1
            logdet += dlogdet
        return x, logdet

    def forward(self, x, logdet=None, reverse=False):

        if not self.inited:
            self.init_param(x)

        if reverse:
            x, logdet = self._scale(x, logdet, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, logdet = self._scale(x, logdet, reverse)

        return x, logdet

class FlowStep(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, x, logdet=None, reverse=False):
        pass