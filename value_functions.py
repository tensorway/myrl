import gym
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ValueFunctionMLP(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, x):
        h = torch.tensor(x, dtype=torch.float)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return h


class DQN(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
    
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.odim = net_arch[-1]
    def forward(self, h):
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return h
    def act(self, obs, epsilon=0.1, debug=False):
        obs = torch.tensor(obs).float()
        qs = self.forward(obs)
        if random.uniform(0, 1) > epsilon:
            ii = torch.argmax(qs, dim=-1).unsqueeze(-1)
        else:
            ii = torch.randint(0, self.odim-1, (qs.shape[0], 1))     
        dummy = torch.tensor([[1]]).float()   
        return ii.numpy(), (dummy, dummy, dummy)
    def get_max(self, obs):
        return torch.max(self.forward(obs), dim=-1)[0].detach().unsqueeze(-1)
    def get_q(self, obs, a):
        bsize = obs.shape[0]
        frst = torch.tensor(list(range(bsize)), device=a.device)
        calc = self.forward(obs)
        calc = calc[frst, a.long().squeeze(-1)].unsqueeze(-1)
        return calc
    def get_action(self, obs):
        return torch.max(self.forward(obs), dim=-1)[1].detach().unsqueeze(-1)

        
        
def polyak(a, b, alfa=0.99):
    for namea, parama in a.named_parameters():
        for nameb, paramb in b.named_parameters():
            if namea == nameb:
                paramb.data = paramb.data*alfa + parama.data*(1-alfa)
    return b 
