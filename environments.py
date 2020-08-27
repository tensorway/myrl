import gym
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random





class Env():
    def __init__(self, envname):
        self.env = gym.make(envname)
        self.obs = self.env.reset()
        self.fix_obs()
    def step(self, act, sample=False, debug=False):
        if sample:
            a, other = self.env.action_space.sample(), -1
            asq = a
        else:
            a, other = act(self.obs)
            asq = a.squeeze(0)

        old_obs = self.obs
        if type(self.env.action_space) is gym.spaces.box.Box:
            obs, r, done, _ = self.env.step(asq)
        else:
            obs, r, done, _ = self.env.step(asq.squeeze(-1))
        self.obs = obs
        self.fix_obs()
        obs = self.obs

        if done > 1e-5:
            self.obs = self.env.reset()
            self.fix_obs()

        old_obs = torch.tensor(old_obs, dtype=torch.float)
        obs = torch.tensor(obs, dtype=torch.float)
        r = torch.tensor([[r]], dtype=torch.float)
        d = torch.tensor([[done]], dtype=torch.float)       
        a = torch.tensor(a, dtype=torch.float)       

        if debug:
            print(old_obs.shape, a.shape, r.shape, obs.shape, d.shape, other[0].shape, other[1].shape, other[2].shape)
        return old_obs, a, r, obs, d, other
    def fix_obs(self):
        try:
            self.obs.shape[1]
        except:
            self.obs = np.expand_dims(self.obs, axis=0)


class Envs():
    def __init__(self, envname, n_envs):
        self.envs = [Env(envname) for i in range(n_envs)]
        self.n_envs = n_envs
        
    def step(self, pi, debug=False):
        l = [[] for i in range(8)]
        for env in self.envs:
            l2 = env.step(pi, debug)
            for i, t in enumerate(l2):
                if type(t) == type((0, 0)):
                    for j, tt in enumerate(t):
                        l[i+j].append(tt)
                else:
                    l[i].append(t)

        toret = [torch.cat(l[i], dim=0) for i in range(len(l))]
        return toret

    def evenout(self, maxstep):
        pi = lambda x:(np.expand_dims(self.envs[0].action_space.sample(), axis=0), -1)
        for i, env in enumerate(self.envs):
            print(i, "/", self.n_envs, end='\r')
            for j in range(random.randint(0, maxstep)):
                _ = env.step(pi, sample=True)

    def rollout(self, pi, gamma=1, length=1e6, debug=False):
        l = [[] for i in range(8)]
        for step in range(int(length)):
            l2 = self.step(pi, debug)
            for i, t in enumerate(l2):
                l[i].append(t.unsqueeze(1))
            if l[4][step].sum() > 1e-5:
                break
        toret = [torch.cat(l[i], dim=1) for i in range(len(l))]
        return toret

    def discounted_sum(self, r, gamma):
        for j in range(r.shape[1]-2, -1, -1):
            r[:, j] += gamma*r[:, j+1]
        return r