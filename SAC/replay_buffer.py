import torch
import numpy as np
import random
from collections import deque
class ReplayBuffer:
    def __init__(self, batch_size=64, buffer_size=1000000, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), state_dim=(3,224,224)):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.state_dim = state_dim
        self.buffer = deque(maxlen=buffer_size)
        
    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

# test_buffer()
