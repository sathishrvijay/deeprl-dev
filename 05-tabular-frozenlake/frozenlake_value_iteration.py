import gymnasium as gym

from torch.utils.tensorboard.writer import SummaryWriter

from collections import defaultdict, Counter
import typing as tt


"""Tabular Offline sync DP Value Iteration method to solve the 4x4 Frozen Lake problem
with stochastic slipping, a problem that CE method struggled with due to sparse rewards and
stochastic state transitions
Note: Experience collection is decoupled from training (classic VE) is decoupled from agent eval
Note: We will init the env within the agent class as is typical

Data structures we need for VF based Tabular Learning
* s, a -> list(s', count)   due to stochastic transitions
* s, a, s' -> r   reward matrix
* v(s) -> estimated value

"""

RL_ENV = "FrozenLake-v1"   # default is the simple 4x4
#RL_ENV = "FrozenLake8x8-v1"
GAMMA = 0.9
NUM_STEPS_PER_RANDOM_PLAY = 100
MAX_EPOCHS = 500   # total number of epochs to collect experience/train/test on
NUM_TRIALS = 20    # trials per epoch to determine if agent succeeded

State = int
Action = int
TransitKey = tt.Tuple[State, Action]
RewardKey = tt.Tuple[State, Action, State]

class Agent:
    def __init__(self):
        self.env = gym.make(RL_ENV)
        self.state, _ = self.env.reset()

        self.rewards_table: tt.Dict[RewardKey, float] = defaultdict(float)
        # transit counts implemented as a nested dict for easy iteration thru next states
        self.transit_counts_table: tt.Dict[TransitKey, Counter] = defaultdict(Counter)
        self.values_table: tt.Dict[State, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        """Note that random play is fully decoupled from agent (policy) because the goal is to
        independently estimate environment dynamics empirically. Also playing full episodes is not
        required"""

        for _ in range(n):
            # advance one step
            action = self.env.action_space.sample()
            next_state, reward, is_done, is_trunc, _ = self.env.step(action)

            # update transition and reward matrices
            rewards_key = (self.state, action, next_state)
            transit_key = (self.state, action)
            self.rewards_table[rewards_key] = float(reward)
            self.transit_counts_table[transit_key][next_state] += 1

            # for debug
            # print(f"s: {self.state}; a: {action}; s': {next_state}, r: {reward}")
            # print(f"rewards_t[{rewards_key}]: {self.rewards_table[rewards_key]}")
            # print(f"transit_t[{transit_key}]: {self.transit_counts_table[transit_key]}")

            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = next_state

    def calc_action_value(self, state, action) -> float:
        """Note that policy wont affect this calculation by definition since action is taken"""

        action_value = 0.0
        transit_key = (state, action)
        # print(f"(s, a): {transit_key} transits: {self.transit_counts_table[transit_key]}")
        total_count_transits = sum([self.transit_counts_table[transit_key][next_state] for next_state
            in range(self.env.observation_space.n)])

        # Divide by zero error: If this (s, a) pair does not exist in experience buffer, then terminate
        if total_count_transits == 0:
            return action_value

        for next_state in range(self.env.observation_space.n):
            rewards_key = (state, action, next_state)
            reward = self.rewards_table[rewards_key]
            count_transits = self.transit_counts_table[transit_key][next_state]
            sars_return = reward + GAMMA * self.values_table[next_state]
            action_value += count_transits / total_count_transits * sars_return

        return action_value

    def select_optimal_action(self, state: State) -> Action:
        """return argmax_a of AVF. However, since AVF is not stored, we need to do a one step
        lookahead search for all possible actions from current state to determine best action"""
        opt_action = None
        opt_return = None

        for action in range(self.env.action_space.n):
            est_return = self.calc_action_value(state, action)
            if opt_action is None or est_return > opt_return:
                opt_action = action
                opt_return = est_return
        return opt_action

    def play_trial_episode(self, env: gym.Env) -> float:
        """Note that we want a separate environment that doesn't mess with sample collection env.
        We use a deterministic agent that makes the optimal moves during episode play w/o exploration
        because data collection is independent and already exploratory
        """
        state, _ = env.reset()
        total_reward: Counter = 0
        while True:
            action = self.select_optimal_action(state)
            next_state, reward, is_done, is_trunc, _ = env.step(action)
            total_reward += reward

            if is_done or is_trunc:
                break
            else:
                state = next_state
        return total_reward

    def value_iteration(self):
        """Recall that V*(s) = max_a Q*(s, a). Also recall that in classic VI, we perform exactly
        one sweep of the state space"""
        for state in range(self.env.observation_space.n):
            action_values = [self.calc_action_value(state, action) for action
                in range(self.env.action_space.n)]
            self.values_table[state] = max(action_values)


if __name__ == '__main__':
    agent = Agent()
    test_env = gym.make(RL_ENV)

    for iter_no in range(MAX_EPOCHS):
        # print(f"iter: {iter_no}")
        agent.play_n_random_steps(NUM_STEPS_PER_RANDOM_PLAY)
        agent.value_iteration()

        avg_return = 0.0
        for _ in range(NUM_TRIALS):
            avg_return += agent.play_trial_episode(test_env)
            # print(f"iter {iter_no}; trial: {_}; total_return: {avg_return}")
        avg_return /= NUM_TRIALS

        # check for early stopping condition
        print(f"iter {iter_no}: average return from {NUM_TRIALS}={avg_return}")
        if avg_return > 0.8:
            print(f"Solved in {iter_no} iterations!")
            break
