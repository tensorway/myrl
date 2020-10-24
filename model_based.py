import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from myrl.utils import get_batch_a, get_batch_obs
import gym
import torch


class ModelResidual(nn.Module):
    def __init__(self, net_arch, env):
        super().__init__()
        l = [nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])]
        l2 = [nn.Linear(net_arch[-2], 1)]
        self.layers = nn.ModuleList(l+l2)
        self.env = env

    def forward(self, oldobs, a):
        # h = torch.tensor(x, dtype=torch.float)
        h = torch.cat((oldobs, a), dim=1)
        for lay in self.layers[:-2]:
            h = F.relu(lay(h))
        obs = self.layers[-2](h) + oldobs
        r = self.layers[-1](h)
        return r, obs

    def random_shooting(self, obs, nsmpls=100, tlen=10, all=False):
        maxr = float('-inf')
        obs = obs.repeat(nsmpls, 1)
        rs = torch.zeros(nsmpls, 1)
        for t in range(tlen):
            a = get_batch_a(self.env, nsmpls)
            r, obs = self.forward(obs, a)
            rs += r
            a = a.unsqueeze(1)
            if t==0:
                ass = a
            else:
                ass = torch.cat((ass, a), dim=1)
        # ii = torch.argmax(rs)
        if all:
            return ass, rs
        else:
            ii = torch.argmax(rs)
            return ass[ii][0].unsqueeze(-1)

    def cross_entropy(self, obs, niter, nsmpls, tlen, pmax=0.1, all=False):
        k = int(nsmpls*pmax)
        aret = None
        # print(niter)
        ass, rs = self.random_shooting(obs, nsmpls, tlen, all=True)
        obs = obs.repeat(nsmpls, 1)
        for iter in range(niter):
            topi = rs.squeeze(-1).topk(k)[1]
            topass = ass[topi]
            m, std = topass.mean(dim=0), topass.std(dim=0)
            dis = torch.distributions.Normal(m, std)
            ass = dis.sample(torch.Size((nsmpls,)))
            rs = self._eval_actions(obs, ass)
        if all:
            return ass, rs
        else:
            ii = torch.argmax(rs)
            return ass[ii][0].unsqueeze(-1) 

    def gradient_optimize(self, obs, tlen, minimax, ngrad_steps=10, grad_step=1e-1, nsmpls=1, all=False):
        obs = torch.tensor(obs).float()
        actions = torch.randn(nsmpls, tlen, 1, requires_grad=True)
        obs = obs.repeat(nsmpls, 1)
        opt = torch.optim.Adam([actions], lr=grad_step)
        for grad_step in range(ngrad_steps):
            rs = 0
            lobs = obs
            for i in range(tlen):
                r, lobs = self.forward(lobs, actions[:, i])
                rs += r
            loss = -rs.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                actions = torch.clamp(actions, -2, 2)
            
        ii = rs.squeeze(-1).argmax()
        if all:
            return actions[ii], rs[ii], actions
        return actions[ii][0].unsqueeze(0).detach().numpy()

    def _eval_actions(self, obs, ass):
        rs = 0
        for i in range(ass.shape[1]):
            a = ass[:, i]
            r, obs = self.forward(obs, a)
            rs += r
        return rs