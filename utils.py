import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import os
import torch.nn as nn


class ExperimentWriter():
    def __init__(self, s):
        self.z = 0
        self.s = s
    def new(self):
        os.system('mkdir '+self.s + str(self.z))
        self.writer = SummaryWriter(self.s  + str(self.z))
        self.z += 1

class Discrete2Continuous():
    def __init__(self, rmin, rmax, ndisc, pi):
        self.pi = pi
        self.cvt = torch.zeros(ndisc, 1)
        tmp = rmin
        assert ndisc > 1, "broj podjela tria bit 2 il vise"
        for i in range(ndisc):
            self.cvt[i, 0] = tmp
            tmp += (-rmin+rmax)/(ndisc-1)

    def act(self, obs):
        a, other = self.pi(obs)
        a = a.squeeze(-1)
        a = torch.tensor(a)
        a2 = self.cvt[a.long()]
        a = a.unsqueeze(-1)
        # print(a2, a2.shape)
        return a2.numpy(), (other[0], other[1], other[2], a)

class RunningNormalizer():
    def __init__(self, alpha=0.97):
        self.m = None
        self.std = None
        self.alpha = alpha
    def step(self, tensor):
        m = tensor.mean()
        std = tensor.std()
        if self.m is None:
            self.m = m
            self.std = std
        else:
            self.m = self.m*self.alpha + m*(1-self.alpha)
            self.std = self.std*self.alpha + std*(1-self.alpha)
        return (tensor-self.m)/self.std




def get_batch_obs(env, bsize):
    bobs = [env.observation_space.sample() for i in range(bsize)]
    bobs = torch.tensor(bobs)
    try:
        bobs.shape[2]
    except :
        bobs = bobs.squeeze(1)
    return bobs

def get_batch_a(env, bsize):
    ba = [env.action_space.sample() for i in range(bsize)]
    ba = torch.tensor(ba)
    try:
        ba.shape[1]
    except :
        ba = ba.unsqueeze(-1)
    return ba

def check_output(env, pi):
    nes = np.expand_dims(env.observation_space.sample(), axis=0)
    a, (d, sm, h)=pi.act(nes)
    return h

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

def global_gradient_clip(model, maxv=0.5):
    with torch.no_grad():
        l2sum = 0
        for p in model.parameters():
            l2sum += (p.grad**2).sum()
        if l2sum.item() > maxv:
            mul = torch.sqrt(maxv/l2sum)
            for p in model.parameters():
                p.grad *= mul
    return l2sum

def gae(rs, obs, vfunc, gamma, lmbda):
    vals = vfunc(obs).detach()
    T = len(vals)
    lmbda_pow = [lmbda**i for i in range(T)]
    rts = torch.zeros_like(rs)
    flag = torch.zeros_like(rts)
    def Rt(t):
        if t == T-1:
            toret = rs[-1]
        else:
            toret = (rs[t]+  (1-lmbda)*gamma*vals[t]) + gamma*lmbda*Rt(t+1)
        rts[t, 0] = toret
 
        flag[t] += 1
        return toret
    Rt(0)
    return rts

def list_gae(list_all, vfunc, gamma, lmbda, rew_dim=2, obs_dim=3):
    toret = []
    for i in range(len(list_all)):
        obs = list_all[i][obs_dim]
        toret.append(gae(list_all[i][rew_dim], obs, vfunc, gamma, lmbda))
    return toret

def add_list2list(l1, l):
    for i in range(len(l)):
        l[i].append(l1[i])

def normal_noise(h, std):
    mu = torch.zeros(h.shape)
    std = mu + std
    dis = torch.distributions.Normal(mu, std)
    smpl = dis.sample()
    return smpl.detach()

