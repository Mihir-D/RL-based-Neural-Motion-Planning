
import numpy as np
from collections import deque
import random

class OUNoise(object):
    def __init__(self, joint_bounds, mu=0.0, theta=0.05, max_sigma=0.1, min_sigma=0.1, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = len(joint_bounds[0]) - 1 # since actions consider only 4 joints
        self.low          = joint_bounds[0][1:] # since joint 0 is ignored
        self.high         = joint_bounds[1][1:]
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + ou_state
        for i in range(self.action_dim):
            action[i] = max(action[i], self.low[i])
            action[i] = min(action[i], self.high[i])
        return np.array(action)