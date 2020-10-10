import torch
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


def get_torch_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    n = Normal(0, v)
    # return torch.Tensor(size).uniform_(-v, v)
    return n.sample(size)


def copy_parameters(target, source, gamma=1.):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1. - gamma) + param.data * gamma)


def save_network(net, filename):
    torch.save(net, filename + 'pth')


def save_parameters(net, filename):
    torch.save(net.state_dict(), filename + 'pth')


def save_param_opt(net, opt, filename):
    torch.save({'params': net.state_dict(),
                'opt': opt.state_dict()}, filename + 'pth')


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class Buffer:
    def __init__(self, max_size=1000, seed=None):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        random.seed(seed)

    @property
    def size(self):
        return len(self.buffer)

    def sample(self, ct):
        ct = min(ct, self.size)
        batch = random.sample(self.buffer, ct)
        s = np.float32([x[0] for x in batch])
        a = np.float32([x[1] for x in batch])
        r = np.float32([x[2] for x in batch])
        s1 = np.float32([x[3] for x in batch])
        a1 = np.float32([x[4] for x in batch])

        return s, a, r, s1, a1

    def sample_(self, ct):
        ct = min(ct, self.size)
        batch = random.sample(self.buffer, ct)
        s = [x[0] for x in batch]
        a = [x[1] for x in batch]
        r = [x[2] for x in batch]
        s1 = [x[3] for x in batch]
        a1 = [x[4] for x in batch]
        ano = [x[5] for x in batch]

        return s, a, r, s1, a1, ano

    def add(self, s, a, r, s1, a1=None, ano=None):
        arr = [s, a, r, s1, a1, ano]
        self.buffer.append(arr)


class PriortizedReplay(Buffer):
    def __init__(self, max_size=1000, seed=None, beta=1., eps=0.1):
        super(PriortizedReplay, self).__init__(max_size, seed)
        self.beta = beta
        self.probs = deque(maxlen=self.max_size)
        self.rg = np.random.RandomState(seed)
        self.eps = eps

    def add(self, s, a, r, s1, a1=None, ano=None, td=0):
        arr = [s, a, r, s1, a1, ano]
        self.probs.append(td + self.eps)
        self.buffer.append(arr)

    def sample(self, ct):
        ct = min(ct, self.size)
        probs = np.array(self.probs)
        probs = probs ** self.beta
        probs = probs / probs.sum()
        idx = [self.rg.choice(self.size, p=probs) for _ in range(ct)]
        s = np.float32([self.buffer[i][0] for i in idx])
        a = np.float32([self.buffer[i][1] for i in idx])
        r = np.float32([self.buffer[i][2] for i in idx])
        s1 = np.float32([self.buffer[i][3] for i in idx])
        a1 = np.float32([self.buffer[i][4] for i in idx])

        return s, a, r, s1, a1

    def sample_(self, ct):
        ct = min(ct, self.size)
        probs = np.array(self.probs)
        probs = probs.argsort() + 1
        probs = (1 / probs)
        probs = probs ** self.beta
        probs = probs / probs.sum()
        idx = [self.rg.choice(self.size, p=probs) for _ in range(ct)]
        s = [self.buffer[i][0] for i in idx]
        a = [self.buffer[i][1] for i in idx]
        r = [self.buffer[i][2] for i in idx]
        s1 = [self.buffer[i][3] for i in idx]
        a1 = [self.buffer[i][4] for i in idx]
        ano = [self.buffer[i][5] for i in idx]

        return s, a, r, s1, a1, ano
