import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter

from collections import defaultdict, Counter
import typing as tt
import random


"""Efficient On-policy TD(0) learning using SARSA. Optimization using GPI which is an online process.
Perform trials periodically to test for stopping condition.

Problem is small enough that an EPS decay schedule may not be required
Note that we don't need to maintain state transition or rewards table anymore since we don't care
about estimating env dynamics in TD Learning (aka model free) which significantly reduces amount of
code required to be written

Data structures we need for VF based Tabular Learning
* q(s, a) -> estimated action values
"""

RL_ENV = "FrozenLake-v1"   # default is the simple 4x4
#RL_ENV = "FrozenLake8x8-v1"
GAMMA = 0.9
ALPHA = 0.1
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.995
MAX_EPOCHS = 1000   # total number of epochs to collect experience/train/test on
NUM_GPI_ITERS = 5000   # Number of GPI iterations before trial
NUM_TRIALS = 100    # trials per epoch to determine if agent succeeded
RANDOM_SEED = 12345

State = int
Action = int
TransitKey = tt.Tuple[State, Action]

class Agent:
    def __init__(self):
        #self.env = gym.make(RL_ENV)
        self.env = gym.make(RL_ENV, is_slippery=True)
        self.state = None
        self.action = None
        self.eps = 1.0

        self.action_values_table: tt.Dict[TransitKey, float] = defaultdict(float)

        # Important: Initialize states to random values to help with learning
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                self.action_values_table[(state, action)] = random.uniform(0.0, 0.05)

        self.state, _ = self.env.reset()
        self.action = self.select_action(self.state)

    def get_best_action_and_value(self, state: State) -> tt.Tuple:
        opt_action, opt_value = None, None
        for action in range(self.env.action_space.n):
            transit_key = (state, action)
            est_value = self.action_values_table[transit_key]
            if opt_action is None or est_value > opt_value:
                opt_action = action
                opt_value = est_value
        return (opt_action, opt_value)

    def select_action(self, state: State, explore=True) -> Action:
        """Uses epsilon greedy to select the optimal action.
        returns argmax_a of AVF if not random sampling. Trivial because Q(s, a) now exists"""
        if explore and random.random() <= self.eps:
            return self.env.action_space.sample()
        else:
            (opt_action, _) = self.get_best_action_and_value(state)
        return opt_action

    def play_trial_episode(self, env: gym.Env) -> float:
        """Note that we want a separate env for trials that doesn't mess with training env.
        We use a deterministic agent that makes the optimal moves during episode play w/o exploration
        because training is independent and already exploratory
        """
        state, _ = env.reset()

        total_reward: Counter = 0
        while True:
            action = self.select_action(state, explore=False)
            next_state, reward, is_done, is_trunc, _ = env.step(action)
            total_reward += reward

            if is_done or is_trunc:
                break
            else:
                state = next_state
        return total_reward

    def sarsa_gpi(self):
        """Alternate between Policy Evaluation and Policy Iteration in a loop aka GPI.
        The algorithm used is On policy TD(0) SARSA"""
        next_state, reward, is_done, is_trunc, _ = self.env.step(self.action)

        # Policy Iteration (SARSA)
        next_action = self.select_action(next_state)

        # Policy Evaluation (i.e. update Q(s, a))
        tkey = (self.state, self.action)
        tkey_next = (next_state, next_action)
        old_q = self.action_values_table[tkey]
        target_return = reward + GAMMA * self.action_values_table[tkey_next]
        new_q = old_q + ALPHA * (target_return - old_q)
        self.action_values_table[tkey] = new_q

        #print(f"(s, a, r, s', a'): ({self.state}, {action}, {reward}, {next_state}, {next_action})")
        #print(f"Update: Q({self.state}, {action}): {new_q}")
        if is_done or is_trunc:
            #print("env reset due to episode complete!")
            self.state, _ = self.env.reset()
            self.action = agent.select_action(self.state)
        else:
            self.state = next_state
            self.action = next_action


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    agent = Agent()
    # writer = SummaryWriter(comment="-sarsa")

    # For trials
    test_env = gym.make(RL_ENV, is_slippery=True)

    for iter_no in range(MAX_EPOCHS):
        agent.eps = max(MIN_EPSILON, agent.eps * EPSILON_DECAY_RATE)

        # Periodically check Optimal Policy
        if iter_no % 50 == 0:
            print(f"epoch {iter_no}: Current eps={agent.eps:.3f} - ")
            for state in range(agent.env.observation_space.n):
                opt_action, opt_value = agent.get_best_action_and_value(state)
                print(f"State {state}: Best Action {opt_action}, Q-values: {opt_value}")

        for _ in range(NUM_GPI_ITERS):
            agent.sarsa_gpi()

        avg_return = 0.0
        for _ in range(NUM_TRIALS):
            avg_return += agent.play_trial_episode(test_env)
            # print(f"iter {iter_no}; trial: {_}; total_return: {avg_return}")
        avg_return /= NUM_TRIALS
        # writer.add_scalar("reward", avg_return, iter_no)

        # check for early stopping condition
        print(f"iter {iter_no}: average return from {NUM_TRIALS}={avg_return}")
        if avg_return > 0.8:
            print(f"Solved in {iter_no} iterations!")
            break
    # writer.close()
