""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce



class GL(nn.Module):
    def __init__(self, in_dim, win):
        super(GL, self).__init__()
        assert isinstance(win, int) and in_dim % win == 0
        self.dsp = nn.Parameter(torch.empty([in_dim]))
        self.bias = nn.Parameter(torch.empty([in_dim // win]))
        self.win = win

    def forward(self, x):
        x = self.dsp * x
        x = reduce(x, f'i j (k l) -> i j k', 'sum', l=self.win)
        x = x + self.bias
        return x


class DFC(nn.Module):
    def __init__(self, dim, theta=8, dropout=0.,
                 multi=True):
        super().__init__()
        self.mask = MaskF(dim, dim, theta, shift=True)

        self.d = nn.Dropout(dropout)
        self.multi = multi

    def forward(self, x):
        if self.multi:
            return self.mask(x) * x
        else:
            return self.mask(x)


class MaskF(nn.Module):
    def __init__(self, in_dim, out_dim, theta=4, shortcut=False, shift=False):
        super().__init__()
        if isinstance(theta, int) and theta > 0 and in_dim % theta == 0:
            self.theta = theta

        self.mask = nn.Linear(self.theta, out_dim)
        self.dsp = DownSampleF(in_dim, in_dim // self.theta)

        self.shortcut = shortcut
        self.shift = shift

    def forward(self, x):
        if self.shortcut:
            x = self.mask(self.dsp(x)) + torch.roll(x, shifts=-1, dims=-1)
        else:
            x = self.mask(self.dsp(x))

        return x


class MlpD(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., ratio=6):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        theta = out_features // ratio

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = DFC(hidden_features, theta, bias=False, act=None,
                           shortcut=True, multi=True, share=False, norm=False)

        self.drop = nn.Dropout(drop)
        self.dsp = DownSampleF(hidden_features, hidden_features // out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dsp(x)
        x = self.drop(x)
        return x


class GatedMlp_RF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        if gate_layer is not None:
            # assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features, act=act_layer)
            hidden_features = hidden_features // 3  # FIXME base reduction on gate property?
            self.gate_norm = nn.LayerNorm(hidden_features)
        else:
            self.gate = nn.Identity()
            self.gate_norm = nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.act(x)
        x = self.gate(x)
        x = self.drop(x)

        x = self.gate_norm(x)
        x = self.fc2(x)


        return x


