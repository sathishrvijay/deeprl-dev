import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from collections import defaultdict, Counter
import typing as tt
import random


"""Efficient On-policy TD learning using SARSA(lambda). Optimization using GPI which is an online process.
Perform trials periodically to test for stopping condition.
ChatGPT recommends replacing trace instead of accumulating trace for stability

Data structures we need for AVF based Tabular Learning
* q(s, a) -> estimated action values
Note: In this code, we also switch to numpy dictionaries to actually simplify code further
"""

#RL_ENV = "FrozenLake-v1"   # default is the simple 4x4
RL_ENV = "FrozenLake8x8-v1"
GAMMA = 0.99
LAMBDA = 0.9
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

        num_actions = self.env.action_space.n
        num_states = self.env.observation_space.n

        # Initialize AVF and Eligibility Traces
        self.Q = np.random.uniform(low=0.0, high=0.05, size=(num_states, num_actions))
        self.E = np.zeros((num_states, num_actions))

        self.state, _ = self.env.reset()
        self.action = self.select_action(self.state)

    def get_best_action_and_value(self, state: State) -> tt.Tuple:
        opt_action = np.argmax(self.Q[state])
        opt_value = np.max(self.Q[state])
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

    def sarsa_lambda_gpi(self, replacing_trace=True):
        """Alternate between Policy Evaluation and Policy Iteration in a loop aka GPI.
        The algorithm used is On policy SARSA(lambda) with replacing trace for training stability

        Note: The TD update equation is the only difference between SARSA and SARSA(lambda). The update
        is identical to SARSA for lambda=0.
        """
        next_state, reward, is_done, is_trunc, _ = self.env.step(self.action)

        # Policy Iteration (SARSA)
        next_action = self.select_action(next_state)

        # Policy Evaluation (i.e. update Q(s, a))
        tkey = (self.state, self.action)
        tkey_next = (next_state, next_action)
        if replacing_trace:
            self.E[tkey] = 1
        else:
            # accumulating trace (more unstable in training due to higher variance)
            self.E[tkey] += 1

        target_return = reward + GAMMA * self.Q[tkey_next] * (not is_done)
        td_error = target_return - self.Q[tkey]
        self.Q[tkey] += ALPHA * self.E[tkey] * td_error
        # Recall that trace decays as product of lambda and gamma
        self.E[tkey] *= GAMMA * LAMBDA

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
            agent.sarsa_lambda_gpi()

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
