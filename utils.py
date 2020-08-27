import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import os

def normal_noise(h, std):
    mu = torch.zeros(h.shape)
    std = mu + std
    dis = torch.distributions.Normal(mu, std)
    smpl = dis.sample()
    return smpl.detach()

class ExperimentWriter():
    def __init__(self, s):
        self.z = 0
        self.s = s
    def new(self):
        os.system('mkdir '+self.s + str(self.z))
        self.writer = SummaryWriter(self.s  + str(self.z))
        self.z += 1

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
        a2 = self.cvt[a]
        a = a.unsqueeze(-1)
        # print(a2, a2.shape)
        return a2.numpy(), (other[0], a, other[1])

def check_output(env, pi):
    nes = np.expand_dims(env.observation_space.sample(), axis=0)
    a, (d, sm, h)=pi.act(nes)
    return h

def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)