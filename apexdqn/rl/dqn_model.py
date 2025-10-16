import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    DQN network as in your notebook: FC 128 -> FC 128 -> output action_size
    The input is a zero vector (context-less) of shape (action_space_size,)
    but we implement it with a single input dimension and fully connected layers.
    """
    def __init__(self, input_dim, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        return q
