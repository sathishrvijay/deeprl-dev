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


class ContinuousA2CActor(nn.Module):
    """Separate Actor Network for A2C with continuous actions using log variance parameterization."""
    def __init__(self, state_dim: int, action_dim: int, hidden1_dim: int = 128, hidden2_dim: int = 64):
        super(ContinuousA2CActor, self).__init__()
        self.action_dim = action_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(hidden2_dim, action_dim)
        self.log_var_head = nn.Linear(hidden2_dim, action_dim)
        
        # Initialize log_var to reasonable values (std ≈ 0.6)
        nn.init.constant_(self.log_var_head.bias, -1.0)  # log(0.37) ≈ -1
        nn.init.xavier_uniform_(self.log_var_head.weight, gain=0.01)  # Small initial weights
        
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        
        # Mean with tanh scaling to action bounds [-2, 2] for Pendulum
        mean = torch.tanh(self.mean_head(features)) * 2.0
        
        # Log variance with clamping to prevent extreme values
        log_var = self.log_var_head(features)
        log_var = torch.clamp(log_var, -5, 2)  # std in [0.007, 2.7] range
        
        return mean, log_var


class ContinuousA2CCritic(nn.Module):
    """Separate Critic Network for A2C with continuous actions."""
    def __init__(self, state_dim: int, hidden1_dim: int = 128, hidden2_dim: int = 32):
        super(ContinuousA2CCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


class ContinuousA2C(nn.Module):
    """Wrapper class for separate Actor and Critic networks to maintain PTAN compatibility."""
    def __init__(self, state_dim: int, action_dim: int,
                 critic_hidden1_dim: int = 128, critic_hidden2_dim: int = 32,
                 actor_hidden1_dim: int = 128, actor_hidden2_dim: int = 64):
        super(ContinuousA2C, self).__init__()
        self.action_dim = action_dim
        self.critic = ContinuousA2CCritic(state_dim, critic_hidden1_dim, critic_hidden2_dim)
        self.actor = ContinuousA2CActor(state_dim, action_dim, actor_hidden1_dim, actor_hidden2_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass that returns (value, action_params) for PTAN compatibility.
        action_params is a concatenated tensor of [mean, log_var] for each action dimension.
        """
        value = self.critic(x)
        action_mean, action_log_var = self.actor(x)
        
        # Concatenate mean and log_var along the last dimension for PTAN compatibility
        action_params = torch.cat([action_mean, action_log_var], dim=-1)
        return value, action_params
    
    def get_action_distribution(self, x: torch.Tensor):
        """Get separate action mean and log variance tensors."""
        action_mean, action_log_var = self.actor(x)
        return action_mean, action_log_var

    def get_actor_parameters(self):
        """Get actor network parameters for separate optimization."""
        return self.actor.parameters()

    def get_critic_parameters(self):
        """Get critic network parameters for separate optimization."""
        return self.critic.parameters()
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy. Used for evaluation and action selection."""
        with torch.no_grad():
            action_mean, action_log_var = self.get_action_distribution(state)
            
            if deterministic:
                return action_mean
            else:
                std = torch.exp(action_log_var / 2)
                eps = torch.randn_like(std)
                return action_mean + std * eps
    
    def compute_log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        """Compute log probability of given actions under current policy."""
        action_mean, action_log_var = self.get_action_distribution(states)
        
        # Gaussian log probability: -0.5 * (log(2π) + log_var + (x-μ)²/σ²)
        log_prob = -0.5 * (
            torch.log(torch.tensor(2 * torch.pi)) + 
            action_log_var + 
            (actions - action_mean).pow(2) / torch.exp(action_log_var)
        )
        return log_prob.sum(dim=-1)  # Sum over action dimensions
    
    def compute_entropy(self, states: torch.Tensor):
        """Compute entropy of the policy at given states."""
        _, action_log_var = self.get_action_distribution(states)
        
        # Gaussian entropy: 0.5 * log(2πe * σ²) = 0.5 * (log(2πe) + log_var)
        entropy = 0.5 * (action_log_var + torch.log(torch.tensor(2 * torch.pi * torch.e)))
        return entropy.sum(dim=-1)  # Sum over action dimensions
