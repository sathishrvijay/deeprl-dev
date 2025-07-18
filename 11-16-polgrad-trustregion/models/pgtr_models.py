import torch
import torch.nn as nn

class DiscreteA2CCritic(nn.Module):
    """Separate Critic Network for A2C with discrete actions."""
    def __init__(self, state_space_dim: int, hidden1_dim: int = 128, hidden2_dim: int = 32):
        super(DiscreteA2CCritic, self).__init__()
        self.state_space_dim = state_space_dim
        self.network = nn.Sequential(
            nn.Linear(state_space_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        # Handle discrete state encoding if needed
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
        return self.network(x)


class DiscreteA2CActor(nn.Module):
    """Separate Actor Network for A2C with discrete actions."""
    def __init__(self, state_space_dim: int, n_actions: int, hidden1_dim: int = 128, hidden2_dim: int = 64):
        super(DiscreteA2CActor, self).__init__()
        self.state_space_dim = state_space_dim
        self.network = nn.Sequential(
            nn.Linear(state_space_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        # Handle discrete state encoding if needed
        if x.dtype == torch.long or x.dtype == torch.int:
            x = torch.nn.functional.one_hot(x, num_classes=self.state_space_dim).float()
        return self.network(x)


class A2CDiscreteActionSeparate(nn.Module):
    """Wrapper class for separate Actor and Critic networks to maintain PTAN compatibility."""
    def __init__(self, state_space_dim: int, n_actions: int,
                 critic_hidden1_dim: int = 128, critic_hidden2_dim: int = 32,
                 actor_hidden1_dim: int = 128, actor_hidden2_dim: int = 64):
        super(A2CDiscreteActionSeparate, self).__init__()
        self.critic = DiscreteA2CCritic(state_space_dim, critic_hidden1_dim, critic_hidden2_dim)
        self.actor = DiscreteA2CActor(state_space_dim, n_actions, actor_hidden1_dim, actor_hidden2_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass that returns (value, action_logits) for PTAN compatibility."""
        value = self.critic(x)
        action_logits = self.actor(x)
        return value, action_logits

    def get_actor_parameters(self):
        """Get actor network parameters for separate optimization."""
        return self.actor.parameters()

    def get_critic_parameters(self):
        """Get critic network parameters for separate optimization."""
        return self.critic.parameters()


class A2CDiscreteAction(nn.Module):
    """Advantage Actor-Critic Model for discrete actions.
    Value and Action Proba (aka Policy) output heads with a shared backbone"""
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
