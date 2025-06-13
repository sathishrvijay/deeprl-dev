import torch
import torch.nn as nn


class AgentNet(nn.Module):
    def __init__(self, state_space_dim: int, h_layer_dim: int, n_actions: int):
        super(AgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_space_dim, h_layer_dim),
            nn.ReLU(),
            nn.Linear(h_layer_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Agent():
    # Maybe not required for simple agents
    def __init__(self):
        return()
