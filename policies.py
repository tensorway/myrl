import gym
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from myrl.utils import get_batch_a

class GaussianPolicy(nn.Module):
    def __init__(self, net_arch, std=None, std_start=1, std_min=0.1, clip_range=[-1, 1]):
        super().__init__()
        # net_arch = net_arch.append(net_arch[-1])
        l = [nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])]
        l2 = [nn.Linear(net_arch[-2], net_arch[-1])]
        l = l + l2
        self.init(l)
        self.layers = nn.ModuleList(l)
        self.std = std
        self.std_start = std_start
        self.std_min = std_min
        self.clip_range = clip_range
        self.last_std = None
    def init(self, l):
        with torch.no_grad():
            for lay in l[:-2]:
                nn.init.orthogonal_(lay._parameters['weight'])
            for lay in l[-2:]:
                lay._parameters['weight'] /= 100
            for lay in l:
                lay._parameters['bias'] *= 0
    def forward(self, x):
        h = torch.tensor(x, dtype=torch.float)
        for lay in self.layers[:-2]:
            h = F.tanh(lay(h))
        m = self.layers[-2](h)

        if self.std is not None:
        	std = torch.zeros_like(m)+self.std
        else:
            std = nn.functional.softplus(self.layers[-1](h))*self.std_start+self.std_min
        self.last_std = std
        return m, std
    def act(self, o, smpl=None, debug=False):
        o = torch.tensor(o, dtype=torch.float)
        h, std = self.forward(o)
        h = h.clamp(self.clip_range[0],self.clip_range[1])
        if debug:
            print(std.mean().item(), "Std")
        dis = torch.distributions.Normal(h, std)
        if smpl is None:
            smpl = dis.rsample()
        a = smpl.detach().numpy()
        return a, (dis.log_prob(smpl).sum(dim=-1).unsqueeze(-1), smpl, h, std)
        
        
        
class CategoricalPolicy(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.odim = net_arch[-1]
    def forward(self, h):
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        h = torch.softmax(h, dim=-1)
        return h
    def act(self, o, smpl=None, debug=False):
        o = torch.tensor(o, dtype=torch.float)
        h = self.forward(o)
        dis = torch.distributions.Categorical(h)
        if smpl == None:
            smpl = dis.sample().unsqueeze(-1)
        a = smpl.detach().numpy()
        return a, (dis.log_prob(smpl), smpl, h)
        
class LinearPolicy(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, x):
        h = torch.tensor(x, dtype=torch.float)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        h = F.tanh(h)*2
        return h
    def act(self, o, std=0.5, debug=False):
        o = torch.tensor(o, dtype=torch.float)
        h = self.forward(o) 
        mu = torch.zeros(h.shape)
        std = mu + std
        dis = torch.distributions.Normal(mu, std)
        smpl = dis.sample()
        a = h + smpl
        a = h.detach().numpy()
        return a, (h, smpl, h)

class RandomPolicy():
    def __init__(self, env):
        self.env = env
    def act(self, obs):
        dummy = torch.tensor([[1]])
        a = get_batch_a(self.env, obs.shape[0]).numpy()
        return a, (dummy, dummy, dummy)














        
       
class LinearPolicy_old(nn.Module):
    def __init__(self, idim, h1dim, odim):
        super().__init__()
        self.lin1 = nn.Linear(idim, h1dim)
        self.lin2 = nn.Linear(h1dim, odim)
        self.odim = odim
    def forward(self, x):
        h = self.lin1(x)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.tanh(h)*2
        return h
    def act(self, o, std=0.5, debug=False):
        o = torch.tensor(o, dtype=torch.float)
        h = self.forward(o) 
        mu = torch.zeros(h.shape)
        std = mu + std
        dis = torch.distributions.Normal(mu, std)
        smpl = dis.sample()
        a = h + smpl
        a = h.detach().numpy()
        return a, (h, smpl, h)

class GaussianPolicy_old(nn.Module):
    def __init__(self, idim, h1dim, odim, std=None, std_start=0.2, std_max=0.4):
        super().__init__()
        self.lin1 = nn.Linear(idim, h1dim)
        self.lin2 = nn.Linear(h1dim, odim)
        self.lin3 = nn.Linear(h1dim, odim)
        self.std = std
        self.std_start = std_start
        self.std_max = std_max
        self.odim = odim
    def forward(self, x):
        h = self.lin1(x)
        h = F.relu(h)
        m = self.lin2(h)
        # m = torch.clamp(m, -2, 2)
        # m = torch.tanh(m)*2
        if self.std is not None:
        	std = (self.lin3(h)+self.std_start)*0+self.std
        else:
        	std = (torch.sigmoid(self.lin3(h)+self.std_start)+1e-8)*self.std_max
        return m, std
    def act(self, o, smpl=None, debug=False):
        o = torch.tensor(o, dtype=torch.float)
        h, std = self.forward(o)
        if debug:
            print(std.mean().item(), "Std")
        dis = torch.distributions.Normal(h, std)
        if smpl is None:
            smpl = dis.sample()
            # smpl = torch.clamp(smpl, -2, 2)
    
        a = smpl.detach().numpy()
        return a, (dis.log_prob(smpl).sum(dim=-1).unsqueeze(-1), smpl, h)
