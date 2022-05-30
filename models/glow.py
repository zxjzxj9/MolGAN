#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

log2pi = math.log(2 * math.pi)

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]

class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        output = self.conv(x)
        return output * torch.exp(self.logs * self.logscale_factor)

def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(hidden_channels, hidden_channels, 1),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def squeeze2d(x, factor):
    n, c, h, w = x.shape
    x = x.view(n, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(n, c * factor * factor, h // factor, w // factor)
    return x


def unsqueeze2d(x, factor):
    n, c, h, w = x.shape
    x = x.view(n, c // factor ** 2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(n, c // factor ** 2, h * factor, w * factor)
    return x


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + log2pi)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    z = torch.normal(mean, torch.exp(logs) * temperature)
    return z


class ActNorm2d(nn.Module):

    def __init__(self, num_features, scale=1.0):
        super().__init__()

        # nchw
        size = (1, num_features, 1, 1)
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
            # print(bias.shape, vars.shape)
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


class Squeeze(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, nchan, lu=False):
        super().__init__()

        w_shape = [nchan, nchan]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not lu:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, l, u = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(u)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            u = torch.triu(u, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)
            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(l)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(u)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.lu = lu

    def get_weight(self, x, reverse=False):
        n, c, h, w = x.shape

        if not self.lu:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)
                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, x, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels // 2, num_channels, 3, padding=1)


    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class FlowStep(nn.Module):
    def __init__(self, c_in, c_hid, act_s, flow_perm, flow_coup, lu):
        super().__init__()
        self.actnorm = ActNorm2d(c_in, act_s)

        if flow_perm == "inv_conv":
            self.perm = InvertibleConv1x1(c_in, lu)
        elif flow_perm == "shuffle":
            # self.perm = Permute2d(c_in, shuffle=True)
            pass
        else:
            # self.perm = Permute2d(c_in, shuffle=True)
            pass

        if flow_coup == "additive":
            self.block = get_block(c_in // 2, c_in // 2, c_hid)
        elif flow_coup == "affine":
            self.block = get_block(c_in // 2, c_in, c_hid)

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, logdet)
        else:
            return self.reverse_flow(x, logdet)

    def normal_flow(self, x, logdet):
        # 1. actnorm
        z, logdet = self.actnorm(x, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.perm(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, x, logdet):
        # 1.coupling
        z1, z2 = split_feature(x, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


if __name__ == "__main__":
    print("Validating actnorm layer")
    a = torch.randn(3, 8, 32, 32)
    print(a.shape)
    x = squeeze2d(a, 2)
    print(x.shape)
    y = unsqueeze2d(x, 2)
    print(y.shape)
    print((a - y).norm())

    print("Validating flow layer")
    a = torch.randn(3, 8, 32, 32)
    fmod = FlowStep(32, 64, 1.0, "inv_conv", "affine", False)
    x, det1 = fmod(a, reverse=False)
    y, det2 = fmod(x, reverse=True)
    print((x-y).nrom())
    print(det1 + det2)
