import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random
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
        batch_start = list(range(0, n_states, self.batch_size)) 
        indices = list(range(n_states))  
        random.shuffle(indices)  
        batches = [indices[i: i + self.batch_size] for i in batch_start]
        return (
            torch.stack(self.states).to(self.device),
            torch.tensor(self.actions, dtype=torch.long).to(self.device),
            torch.tensor(self.probs, dtype=torch.float).to(self.device),
            torch.tensor(self.vals, dtype=torch.float).to(self.device),
            torch.tensor(self.rewards, dtype=torch.float).to(self.device),
            torch.tensor(self.dones, dtype=torch.float).to(self.device),
            torch.tensor(batches, dtype=torch.float).to(self.device)
        )
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []