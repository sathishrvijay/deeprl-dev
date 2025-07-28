import ptan
import numpy as np

"""The core purpose of an agent is to take actions according to a policy. An Agent class wrapper therefore consults
the internal models to make policy predictions and take a corresponding action."""

class AgentContinuousA2C(ptan.agent.BaseAgent):
    """PTAN agent wrapper class to convert network output to actions for experience
    collection.
    NOTE: we needed to write one because it doesn't exist for continuous A2C"""
    def __init__(self, net, deterministic=False):
        self.net = net
        self.deterministic = deterministic

    def __call__(self, states: ptan.agent.States, agent_states: ptan.agent.AgentStates):
        states_v = ptan.agent.float32_preprocessor(states)
        actions = self.net.sample_action(states_v, self.deterministic)
        return actions, agent_states


class AgentDDPG(ptan.agent.BaseAgent):
    """PTAN agent wrapper class to convert network output to actions for experience
    collection.
    NOTE**: DDPG uses OU noise injection to help with agent action exploration during
    training. This requires agents persisting memory across steps. Since the network
    itself is stateless, this processing has to be performed in the agent.
    PTAN provides a ptan.agent.AgentStates implementation to help with this.
    [vs] I have not understood the math or intuition for OU, implementing as is."""
    def __init__(self, net, deterministic=False,
                 ou_mu: float = 0.0, ou_theta: float = 0.15,
                 ou_sigma: float = 0.1, ou_epsilon: float = 1.0,
                 ou_eps_decay: float = 0.999):
        self.net = net
        self.deterministic = deterministic
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.ou_theta = ou_theta
        self.ou_eps_init = ou_epsilon
        self.ou_eps = ou_epsilon
        self.ou_eps_decay = ou_eps_decay

    def decay_ou_eps(self):
        # Decay EPS to lower exploration later
        self.ou_eps *= self.ou_eps_decay

    def __call__(self, states: ptan.agent.States, agent_states: ptan.agent.AgentStates):
        states_v = ptan.agent.float32_preprocessor(states)
        actions = self.net.sample_action(states_v).cpu().numpy()

        if self.deterministic:
            new_agent_states = agent_states
        else:
            # An OU noise process is a stateful recursion
            # x_{t+1} = x_t + theta * (mu - x_t) + sigma * mathcal N(0,1)
            if agent_states is None:
                # reset epsilon whenever episode terminates and new one begins
                self.ou_eps = self.ou_eps_init
                agent_states = [np.zeros(action.shape, np.float32) for action in actions]
            # Implement OU noise
            new_agent_states = []
            for a_state, action in zip(agent_states, actions):
                a_state += self.ou_theta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                action += self.ou_eps * a_state
                # stay w/in -2.0 to 2.0 for pendulum after OU noise added
                action = np.clip(action, -2.0, 2.0)
                new_agent_states.append(a_state.copy())
        return actions, new_agent_states
