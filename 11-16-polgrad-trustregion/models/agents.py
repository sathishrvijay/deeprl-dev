import ptan


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
        # mu_v, var_v, _ = self.net(states_v)
        # mu = mu_v.data.cpu().numpy()
        # if deterministic:
        #     actions = mu
        # else:
        #     std = torch.exp(var_v / 2).data.cpu().numpy()
        #     eps = torch.randn_like(std)
        #     actions = mu + std * eps
        # actions = np.clip(actions, -1, 1)
        return actions, agent_states
