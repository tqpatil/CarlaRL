import torch
from agent import Agent
import matplotlib.pyplot as plt
from environment import CarlaEnv
import numpy as np

if __name__ == '__main__':
    lr = 0.001
    n_epochs = 50
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
