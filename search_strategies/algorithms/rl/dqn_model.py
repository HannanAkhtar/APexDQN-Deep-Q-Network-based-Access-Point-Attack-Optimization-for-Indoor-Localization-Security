import torch
import torch.nn as nn


class DQN(nn.Module):
    """DQN network used by the benchmarked DQN-based search method."""

    def __init__(self, input_dim, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
