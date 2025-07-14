#!/usr/bin/env python3
"""Quick test to verify LunarLander environment and network setup"""

import gymnasium as gym
import torch
from models import dqn_models

def test_lunarlander_setup():
    print("Testing LunarLander-v2 environment setup...")

    # Test environment
    env = gym.make("LunarLander-v2")
    obs, info = env.reset()
    print(f"✓ Environment created successfully")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Initial observation shape: {obs.shape}")
    print(f"  - Initial observation: {obs}")

    # Test network
    n_states = env.observation_space.shape[0]  # Should be 8 for LunarLander
    n_actions = env.action_space.n  # Should be 4 for LunarLander

    print(f"\nTesting network with {n_states} states and {n_actions} actions...")
    net = dqn_models.DQNTwoHL(n_states, 128, 64, n_actions)

    # Test forward pass
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    q_values = net(obs_tensor)
    print(f"✓ Network forward pass successful")
    print(f"  - Input shape: {obs_tensor.shape}")
    print(f"  - Output shape: {q_values.shape}")
    print(f"  - Q-values: {q_values}")

    # Test action selection
    action = torch.argmax(q_values, dim=1).item()
    print(f"  - Selected action: {action}")

    # Test environment step
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  - New observation shape: {obs.shape}")
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")

    env.close()
    print("\n✓ All tests passed! LunarLander setup is working correctly.")

if __name__ == "__main__":
    test_lunarlander_setup()
