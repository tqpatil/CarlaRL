import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
class ReplayBuffer:
    def __init__(self, batch_size, device, state_dim):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.device = device
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.probs.append(probs)
        self.vals.append(vals)


    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i+self.batch_size] for i in batch_start]
        return (
            torch.FloatTensor(self.states).to(self.device),
            torch.LongTensor(self.actions).to(self.device),
            torch.FloatTensor(self.probs).to(self.device),
            torch.FloatTensor(self.vals).to(self.device),
            torch.FloatTensor(self.rewards).to(self.device),
            torch.FloatTensor(self.dones).to(self.device),
            torch.FloatTensor(batches).to(self.device)
        )
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []