import torch
import torch.nn as nn

class DQNOneHL(nn.Module):
    def __init__(self, state_space_dim: int, h_layer_dim: int, n_actions: int):
        super(DQNOneHL, self).__init__()
        self.state_space_dim = state_space_dim
        self.net = nn.Sequential(
            nn.Linear(state_space_dim, h_layer_dim),
            nn.LayerNorm(h_layer_dim),
            nn.ReLU(),
            nn.Linear(h_layer_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # FrozenLake uses discrete states, so we need to one hot encodes the states before use
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
        return self.net(x)


class DQNTwoHL(nn.Module):
    def __init__(self, state_space_dim: int, h_layer1_dim: int, h_layer2_dim: int,
        n_actions: int):
        super(DQNTwoHL, self).__init__()
        self.state_space_dim = state_space_dim
        self.net = nn.Sequential(
            nn.Linear(state_space_dim, h_layer1_dim),
            nn.LayerNorm(h_layer1_dim),
            nn.ReLU(),
            nn.Linear(h_layer1_dim, h_layer2_dim),
            nn.LayerNorm(h_layer2_dim),
            nn.ReLU(),
            nn.Linear(h_layer2_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # FrozenLake uses discrete states, so we need to one hot encodes the states before use
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
        #breakpoint()
        return self.net(x)
