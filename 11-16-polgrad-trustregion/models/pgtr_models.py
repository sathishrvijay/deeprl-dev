import torch
import torch.nn as nn
import math

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

        # Initialize log_var to reasonable values for stable exploration
        # Start with std ≈ 0.3 for conservative initial exploration
        nn.init.constant_(self.log_var_head.bias, -2.2)  # log(0.11) ≈ -2.2, std ≈ 0.33
        nn.init.xavier_uniform_(self.log_var_head.weight, gain=0.01)  # Small initial weights

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        mean = self.mean_head(features)

        # Log variance with soft bounds to prevent extreme values while preserving gradients
        log_var = self.log_var_head(features)
        # Use tanh-based soft clamping to avoid gradient death
        # Maps (-∞, ∞) → (-4, 1) smoothly, preserving gradients everywhere
        log_var = torch.tanh(log_var / 2.0) * 2.5 - 1.5  # std range ≈ [0.02, 1.6]
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
        # Sample raw actions once and keep reference (stable approach)
        actions_mu, actions_logvar = self.get_action_distribution(x)
        std = torch.exp(actions_logvar / 2)
        eps = torch.randn_like(std)
        raw_actions = actions_mu + std * eps
        # Compute log probabilities using raw actions (stable, no atanh)
        logproba_actions = self.compute_logproba_raw_actions(raw_actions, actions_mu, actions_logvar)
        value = self.critic(x)
        return logproba_actions, actions_mu, actions_logvar, value

    def get_action_distribution(self, x: torch.Tensor):
        actions_mean, actions_logvar = self.actor(x)
        return actions_mean, actions_logvar

    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy using the Reparametrizion trick for differentiability
        Used for evaluation and action selection."""
        with torch.no_grad():
            actions_mean, actions_logvar = self.get_action_distribution(state)

            if deterministic:
                # For deterministic evaluation, apply tanh to mean
                return torch.tanh(actions_mean)

            else:
                std = torch.exp(actions_logvar / 2)
                eps = torch.randn_like(std)
                raw_actions = actions_mean + std * eps
                # Apply tanh squashing for bounded actions
                actions = torch.tanh(raw_actions)
                return actions

    def compute_logproba_raw_actions(self, raw_actions, actions_mu, actions_logvar):
        """Compute log probabilities for raw actions with stable Jacobian correction
        This avoids the numerical instability of atanh() by working directly with raw actions"""
        
        # Gaussian log probability for raw actions
        log_proba_raw = -0.5 * (
            torch.log(torch.tensor(2 * math.pi)) +
            actions_logvar +
            (raw_actions - actions_mu).pow(2) / torch.exp(actions_logvar)
        )
        
        # Stable Jacobian correction for tanh squashing
        # d/du[tanh(u) * 2] = 2 * (1 - tanh^2(u))
        # log|J| = log(2 * (1 - tanh^2(u)))
        tanh_raw = torch.tanh(raw_actions)
        jacobian_correction = torch.log(2 * (1 - tanh_raw.pow(2)) + 1e-6)
        
        # Sum over action dimensions and apply correction
        log_proba_squashed = (log_proba_raw + jacobian_correction).sum(dim=-1)
        return log_proba_squashed

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critic_parameters(self):
        return self.critic.parameters()


class SAC(nn.Module):
    """Wrapper class for separate Actor and Critic networks to maintain PTAN compatibility.
        Note: Bascically identical to ContinuousA2C structure with 2 critic nets.
        - Actor net is same as A2C
        - Critic net is same as DDPG
        """
    def __init__(self, state_dim: int, action_dim: int,
                 critic_hidden1_dim: int = 128, critic_hidden2_dim: int = 32,
                 actor_hidden1_dim: int = 128, actor_hidden2_dim: int = 64):
        super(SAC, self).__init__()
        self.action_dim = action_dim
        self.critic_1 = DDPGCritic(state_dim, action_dim, critic_hidden1_dim, critic_hidden2_dim)
        self.critic_2 = DDPGCritic(state_dim, action_dim, critic_hidden1_dim, critic_hidden2_dim)
        self.actor = ContinuousA2CActor(state_dim, action_dim, actor_hidden1_dim,
                                        actor_hidden2_dim)

    def get_action_distribution(self, x: torch.Tensor):
        actions_mean, actions_logvar = self.actor(x)
        return actions_mean, actions_logvar

    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy using the Reparametrizion trick for differentiability
        Used for evaluation and action selection."""
        with torch.no_grad():
            actions_mean, actions_logvar = self.get_action_distribution(state)

            if deterministic:
                # For deterministic evaluation, apply tanh to mean
                return torch.tanh(actions_mean) * 2.0
            else:
                std = torch.exp(actions_logvar / 2)
                eps = torch.randn_like(std)
                raw_actions = actions_mean + std * eps
                # Apply tanh squashing for bounded actions
                actions = torch.tanh(raw_actions) * 2.0
                return actions

    def compute_logproba_raw_actions(self, raw_actions, actions_mu, actions_logvar):
        """Compute log probabilities for raw actions with stable Jacobian correction
        This avoids the numerical instability of atanh() by working directly with raw actions"""
        
        # Gaussian log probability for raw actions
        log_proba_raw = -0.5 * (
            torch.log(torch.tensor(2 * math.pi)) +
            actions_logvar +
            (raw_actions - actions_mu).pow(2) / torch.exp(actions_logvar)
        )
        
        # Stable Jacobian correction for tanh squashing
        # d/du[tanh(u) * 2] = 2 * (1 - tanh^2(u))
        # log|J| = log(2 * (1 - tanh^2(u)))
        tanh_raw = torch.tanh(raw_actions)
        jacobian_correction = torch.log(2 * (1 - tanh_raw.pow(2)) + 1e-6)
        
        # Sum over action dimensions and apply correction
        log_proba_squashed = (log_proba_raw + jacobian_correction).sum(dim=-1)
        return log_proba_squashed

    def forward(self, x: torch.Tensor):
        # Sample raw actions once and keep reference (stable approach)
        actions_mu, actions_logvar = self.get_action_distribution(x)
        std = torch.exp(actions_logvar / 2)
        eps = torch.randn_like(std)
        raw_actions = actions_mu + std * eps
        
        # Apply tanh squashing for bounded actions
        squashed_actions = torch.tanh(raw_actions) * 2.0
        
        # Get Q-values using squashed actions
        qvalue_1 = self.critic_1(x, squashed_actions)
        qvalue_2 = self.critic_2(x, squashed_actions)

        # Compute log probabilities using raw actions (stable, no atanh)
        logproba_actions = self.compute_logproba_raw_actions(raw_actions, actions_mu, actions_logvar)
        return logproba_actions, qvalue_1, qvalue_2

    def get_critic_net(self, critic_id: int = 1):
        if critic_id == 1:
            return self.critic_1
        else:
            assert(critic_id == 2)
            return self.critic_2

    def get_actor_net(self):
        return self.actor

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critic_parameters(self, critic_id: int = 1):
        if critic_id == 1:
            return self.critic_1.parameters()
        else:
            assert(critic_id == 2)
            return self.critic_2.parameters()


class DDPGActor(nn.Module):
    """Actor network in DDPG is deterministic (unlike CA A2C and PPO)"""
    def __init__(self, state_dim: int,
                 action_dim: int,
                 hidden1_dim: int = 128,
                 hidden2_dim: int = 64,
                 action_scale = 2.0):
        super(DDPGActor, self).__init__()
        self.action_scale = action_scale

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, action_dim)
        )

        # Initialize last actor layer w/ smaller stddev (DDPG paper trick)
        nn.init.uniform_(self.network[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-1].bias, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor):
        # tanh scaling to action bounds [-2, 2] for Pendulum
        actions_mu = torch.tanh(self.network(x)) * self.action_scale
        return actions_mu


class DDPGCritic(nn.Module):
    """Critic Network for DDPG takes current action as input and returns Q(s, a)"""
    def __init__(self, state_dim: int, action_dim: int, hidden1_dim: int = 128, hidden2_dim: int = 32):
        super(DDPGCritic, self).__init__()
        self.state_stack = nn.Sequential(
            nn.Linear(state_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU()
        )
        self.actionvalue_stack = nn.Sequential(
            nn.Linear(hidden1_dim + action_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1)
        )

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        # returns Q(s, a)
        x = self.state_stack(x)
        return self.actionvalue_stack(torch.concat((x, actions), dim=-1))


class DDPG(nn.Module):
    """Wrapper class for separate DDPG Actor and Critic networks to maintain PTAN compatibility."""
    def __init__(self, state_dim: int, action_dim: int,
                 critic_hidden1_dim: int = 128, critic_hidden2_dim: int = 32,
                 actor_hidden1_dim: int = 128, actor_hidden2_dim: int = 64):
        super(DDPG, self).__init__()
        self.action_dim = action_dim
        self.critic = DDPGCritic(state_dim, action_dim, critic_hidden1_dim, critic_hidden2_dim)
        self.actor = DDPGActor(state_dim, action_dim, actor_hidden1_dim, actor_hidden2_dim)

    def forward(self, x: torch.Tensor):
        actions_mean = self.actor(x)
        qvalue = self.critic(x, actions_mean)
        return actions_mean, qvalue

    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from the policy. Used for evaluation and action selection."""
        with torch.no_grad():
            actions_mean = self.actor(state)
            return actions_mean

    def get_critic_net(self):
        return self.critic

    def get_actor_net(self):
        return self.actor

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critic_parameters(self):
        return self.critic.parameters()
