import torch
from agent import Agent
import matplotlib.pyplot as plt
from environment import CarlaEnv
import numpy as np
"""
Traceback (most recent call last):
  File "train.py", line 27, in <module>
    action,prob, val = agent.choose_action(state)
  File "/home/ubuntu/Documents/CarlaRL/agent.py", line 408, in choose_action
    dist = self.actor(state)
  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/Documents/CarlaRL/agent.py", line 329, in forward
    x7 = F.relu(self.fc1(x6))
  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/miniconda3/envs/carla/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x62720 and 224x1024
"""
if __name__ == '__main__':
    lr = 0.0001
    n_epochs = 15
    agent = Agent(n_actions=5, alpha = lr, n_epochs=n_epochs)
    env = CarlaEnv()
    N = 50
    n_steps = 0
    score_history = []
    num_episodes = 1000
    best_score = float('-inf')
    for episode in range(num_episodes):
        if episode == 0:
            state = torch.tensor(env.start(), dtype=torch.float).to(agent.actor.device).unsqueeze(0)
            state = state.permute(0, 3, 1, 2)
            print(state.shape)
        else:
            state = torch.tensor(env.reset(), dtype=torch.float).to(agent.actor.device).unsqueeze(0)
            state = state.permute(0, 3, 1, 2)
        done = False
        score = 0
        while not done:
            action,prob, val = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            state_ = torch.tensor(state_, dtype=torch.float).to(agent.actor.device).unsqueeze(0)
            state_ = state_.permute(0, 3, 1, 2)
            n_steps += 1
            score += reward
            agent.store_memory(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
            state = state_
        score_history.append(score)
        avg_score = np.mean(score_history[-100::])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f'episode: {episode}, score: {score:.2f}, avg_score: {avg_score:.2f}, time_steps: {n_steps}')
        plt.plot(score_history)
