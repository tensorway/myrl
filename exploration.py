from myrl.value_functions import ValueFunctionMLP
import torch

class RandomNetworkBonus():
    def __init__(self, random_net_arch, estimator_net_arch, lr, opt_steps=2, weight=1):
        self.random_net = ValueFunctionMLP(random_net_arch)
        self.estimator_net = ValueFunctionMLP(estimator_net_arch)
        self.opt = torch.optim.Adam(self.estimator_net.parameters(), lr=lr)
        self.opt_steps = opt_steps
        self.weight = weight
    def step(self, obs):
        for i in range(self.opt_steps):
            loss = ((self.random_net(obs) + self.estimator_net(obs))**2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
    def get_bonus(self, obs):
        loss = ((self.random_net(obs) + self.estimator_net(obs))**2).detach()
        if loss.shape[0] > 1:
            return loss
        return loss.item()