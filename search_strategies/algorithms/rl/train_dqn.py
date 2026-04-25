"""Implements the DQN-based search agent and its training procedure."""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from itertools import combinations

from .dqn_model import DQN
from .env_bandit import BanditEnv


class DQNAgent:
    """A Deep Q-Network agent for the AP selection bandit task."""

    def __init__(self, input_dim, action_size, lr=1e-3, eps=1.0, eps_decay=0.995, eps_min=0.01, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.q_net = DQN(input_dim, action_size).to(self.device)
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state):
        if np.random.rand() <= self.eps:
            return np.random.randint(self.q_net.fc3.out_features)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(s).cpu().numpy().squeeze()
            return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        q_chosen = q_values.gather(1, torch.tensor([[action]], device=self.device))
        loss = self.loss_fn(q_chosen, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def train_dqn_for_k(
    baseline_backend, baseline_model,
    X_val, y_val, ap_columns, k,
    noise_is_scalar, noise_value,
    num_episodes=100, lr=1e-3, eps=1.0, eps_decay=0.995, eps_min=0.01
):
    """Trains the DQN-based search agent to find the most critical AP subset of size k."""
    attack_combinations = list(combinations(ap_columns, k))
    action_size = len(attack_combinations)
    env = BanditEnv(baseline_backend, baseline_model, X_val, y_val, attack_combinations, noise_is_scalar, noise_value)
    agent = DQNAgent(input_dim=action_size, action_size=action_size, lr=lr, eps=eps, eps_decay=eps_decay, eps_min=eps_min)

    best_mse = -float('inf')
    best_combo = None

    for _ in trange(num_episodes, desc=f'DQN-based search (k={k})'):
        state = env.reset()
        action = agent.act(state)
        next_state, reward, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        if reward > best_mse:
            best_mse = reward
            best_combo = attack_combinations[action]

    return best_combo, best_mse, agent.q_net
