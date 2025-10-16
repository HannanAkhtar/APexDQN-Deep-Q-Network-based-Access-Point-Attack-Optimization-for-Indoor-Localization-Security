"""Implements the APexDQN agent and its training procedure.

This module defines the DQN-based agent for the contextual bandit formulation
of the AP selection problem. The agent learns to select k-sized AP subsets
that maximize localization error upon perturbation. The learning process uses
an epsilon-greedy policy and updates the Q-network via regression on the
observed rewards (attack MSE).
"""
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
        """Initializes the DQNAgent.

        Args:
            input_dim (int): Dimensionality of the state space.
            action_size (int): The number of possible actions (AP combinations).
            lr (float): Learning rate for the Adam optimizer.
            eps (float): Initial value for epsilon in the epsilon-greedy policy.
            eps_decay (float): Multiplicative decay factor for epsilon.
            eps_min (float): Minimum value for epsilon.
            device (torch.device, optional): The device to run the model on.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.q_net = DQN(input_dim, action_size).to(self.device)
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.eps:
            # Exploration: choose a random action.
            return np.random.randint(self.q_net.fc3.out_features)
        
        # Exploitation: choose the best-known action.
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(s).cpu().numpy().squeeze()
            return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state):
        """Updates the Q-network based on an observed transition.

        For the one-step bandit, the learning target is simply the immediate reward.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict Q-value for the chosen action.
        q_values = self.q_net(state_tensor)
        q_chosen = q_values.gather(1, torch.tensor([[action]], device=self.device))
        
        # Update network weights.
        loss = self.loss_fn(q_chosen, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        # Decay exploration rate.
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

def train_dqn_for_k(
    baseline_backend, baseline_model,
    X_val, y_val, ap_columns, k,
    noise_is_scalar, noise_value,
    num_episodes=100, lr=1e-3, eps=1.0, eps_decay=0.995, eps_min=0.01
):
    """Trains the APexDQN agent to find the most critical AP subset of size k.

    Args:
        baseline_backend (str): The model backend ('nn' or 'xgb').
        baseline_model: The pre-trained baseline localization model.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.DataFrame): Validation target data.
        ap_columns (list): List of AP column names to select from.
        k (int): The number of APs to select for attack.
        noise_is_scalar (bool): Flag indicating if noise is a single value.
        noise_value (float or np.ndarray): The noise to apply.
        num_episodes (int): The number of training episodes.
        lr (float): Learning rate.
        eps (float): Initial epsilon.
        eps_decay (float): Epsilon decay rate.
        eps_min (float): Minimum epsilon.

    Returns:
        tuple: A tuple containing:
            - best_combo (tuple): The AP combination yielding the highest MSE.
            - best_mse (float): The highest MSE achieved.
            - q_net (DQN): The trained Q-network.
    """
    # Construct the action space from all k-sized AP combinations.
    attack_combinations = list(combinations(ap_columns, k))
    action_size = len(attack_combinations)

    # Initialize environment and agent.
    env = BanditEnv(baseline_backend, baseline_model, X_val, y_val, attack_combinations, noise_is_scalar, noise_value)
    agent = DQNAgent(input_dim=action_size, action_size=action_size, lr=lr, eps=eps, eps_decay=eps_decay, eps_min=eps_min)
    
    best_mse = -float('inf')
    best_combo = None

    for _ in trange(num_episodes, desc=f"APexDQN Training (k={k})"):
        # The state is a fixed zero vector, as per the contextual bandit formulation.
        state = env.reset()
        
        # An episode consists of a single action and reward.
        action = agent.act(state)
        next_state, reward, _ = env.step(action)
        
        agent.learn(state, action, reward, next_state)
        
        # Track the best-performing action discovered so far.
        if reward > best_mse:
            best_mse = reward
            best_combo = attack_combinations[action]

    return best_combo, best_mse, agent.q_net

