import ptan

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
