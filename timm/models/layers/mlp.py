""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import nn as nn

from .helpers import to_2tuple


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.,
            split=0., pos_scale=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        self.part_features = int(out_features * split)
        self.rest_features = hidden_features-self.part_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features-self.part_features,
                             out_features-self.part_features,
                             kernel_size=1, bias=bias[1])
        self.pos_scale = pos_scale
        if pos_scale > 0:
            self.pos_bias = nn.Parameter(torch.linspace(0, pos_scale, hidden_features),
                                         requires_grad=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x)
        if self.pos_scale > 0:
            # x = x + self.pos_bias
            x = x.add(self.pos_bias.reshape(1, -1, 1, 1))

        # x = self.drop1(x)
        if self.part_features > 0:
            x1, x2 = torch.split(x, [self.rest_features, self.part_features], dim=1)
            x = torch.cat([self.fc2(x1), x2], dim=1)
        else:
            x = self.fc2(x)

        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
                 split=0., pos_scale=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.part_features = int(out_features * split)
        self.rest_features = hidden_features-self.part_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features-self.part_features,
                             out_features-self.part_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.pos_scale = pos_scale
        self.first = True
        self.train_scale = 1.
        if pos_scale > 0:
            self.pos_bias = nn.Parameter(torch.linspace(0, pos_scale, hidden_features), requires_grad=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # if self.training or self.pos_scale <= 0:
        if True:
            if self.pos_scale > 0:
                x = x + self.train_scale * self.pos_bias

            x = self.drop1(x)
            if self.part_features > 0:
                x1, x2 = torch.split(x, [self.rest_features, self.part_features], dim=-1)
                x = torch.cat([self.fc2(x1), x2], dim=-1)
            else:
                x = self.fc2(x)
        # for inference after training
        else:
            if self.part_features > 0:
                if self.first:
                    self.fc2.bias = torch.nn.Parameter(self.fc2.bias + self.fc2.weight @
                                                   self.pos_bias[:self.rest_features])
                    self.pos_bias = torch.nn.Parameter(self.pos_bias[self.rest_features:])
                    self.first = False
                x1, x2 = torch.split(x, [self.rest_features, self.part_features], dim=-1)
                x = torch.cat([self.fc2(x1), x2.add(self.pos_bias)], dim=-1)
            else:
                if self.first:
                    self.fc2.bias = torch.nn.Parameter(self.fc2.bias + self.fc2.weight @ self.pos_bias)
                    self.first = False
                x = self.fc2(x)

        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
            gate_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



