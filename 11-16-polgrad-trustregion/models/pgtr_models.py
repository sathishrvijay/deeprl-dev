import torch
import torch.nn as nn

class A2CDiscreteAction(nn.Module):
    """Value and Action Proba (aka Policy) output heads with a shared backbone"""
    def __init__(self, state_space_dim: int, h_layer1_dim: int,
        h_layer2v_dim: int, h_layer2a_dim: int, n_actions: int):
        super(A2CDiscreteAction, self).__init__()
        self.state_space_dim = state_space_dim
        self.base_layer = nn.Sequential(
            nn.Linear(state_space_dim, h_layer1_dim),
            nn.LayerNorm(h_layer1_dim),
            nn.ReLU()
        )
        self.critic_head = nn.Sequential(
            nn.Linear(h_layer1_dim, h_layer2v_dim),
            nn.LayerNorm(h_layer2v_dim),
            nn.ReLU(),
            nn.Linear(h_layer2v_dim, 1)
        )
        self.actor_head = nn.Sequential(
            nn.Linear(h_layer1_dim, h_layer2a_dim),
            nn.LayerNorm(h_layer2a_dim),
            nn.ReLU(),
            nn.Linear(h_layer2a_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # FrozenLake uses discrete states, so we need to one hot encodes the states before use
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
        #breakpoint()
        x = self.base_layer(x)
        value, actions = self.critic_head(x), self.actor_head(x)
        return value, actions
