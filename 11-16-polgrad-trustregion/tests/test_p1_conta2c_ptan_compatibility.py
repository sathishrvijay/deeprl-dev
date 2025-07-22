#!/usr/bin/env python3
"""
Test script to verify the PTAN compatibility fix works correctly.
This script tests the model architecture and action selector without full training.
"""

import torch
import numpy as np
from models.pgtr_models import ContinuousA2C

def test_ptan_compatibility():
    """Test that our continuous A2C model works with PTAN's expected tensor format."""
    print("Testing PTAN Compatibility Fix...")

    # Pendulum environment specs
    state_dim = 3  # [cos(θ), sin(θ), θ_dot]
    action_dim = 1  # torque
    batch_size = 16

    # Create model
    model = ContinuousA2C(
        state_dim=state_dim,
        action_dim=action_dim,
        critic_hidden1_dim=128,
        critic_hidden2_dim=32,
        actor_hidden1_dim=128,
        actor_hidden2_dim=64
    )

    # Test forward pass - should return concatenated tensor
    dummy_states = torch.randn(batch_size, state_dim)
    value, action_params = model(dummy_states)

    print(f"✅ Forward pass successful")
    print(f"   Value shape: {value.shape}")
    print(f"   Action params shape: {action_params.shape}")
    print(f"   Expected action params shape: ({batch_size}, {2 * action_dim})")

    # Verify action_params has correct shape (batch_size, 2*action_dim)
    expected_shape = (batch_size, 2 * action_dim)
    if action_params.shape == expected_shape:
        print("✅ Action parameters have correct concatenated shape")
    else:
        print(f"❌ ERROR: Action parameters shape mismatch: {action_params.shape} vs {expected_shape}")
        return False

    # Test get_action_distribution method
    action_mean, action_log_var = model.get_action_distribution(dummy_states)
    print(f"✅ get_action_distribution successful")
    print(f"   Action mean shape: {action_mean.shape}")
    print(f"   Action log_var shape: {action_log_var.shape}")

    # Verify shapes match
    if action_mean.shape == (batch_size, action_dim) and action_log_var.shape == (batch_size, action_dim):
        print("✅ Action distribution shapes are correct")
    else:
        print(f"❌ ERROR: Action distribution shape mismatch")
        return False

    # Test that concatenation and splitting are consistent
    reconstructed_params = torch.cat([action_mean, action_log_var], dim=-1)
    if torch.allclose(action_params, reconstructed_params):
        print("✅ Concatenation and splitting are consistent")
    else:
        print("❌ ERROR: Concatenation/splitting inconsistency")
        return False

    # Test action selector compatibility
    from pendulumv1_cont_a2c_ptan import ContinuousActionSelector

    # Convert to numpy to simulate PTAN's behavior
    action_params_np = action_params.cpu().data.numpy()

    # Test stochastic action selector
    stochastic_selector = ContinuousActionSelector(deterministic=False)
    stochastic_actions = stochastic_selector(action_params_np)

    print(f"✅ Stochastic action selection successful")
    print(f"   Actions shape: {stochastic_actions.shape}")
    print(f"   Actions range: [{stochastic_actions.min():.3f}, {stochastic_actions.max():.3f}]")

    # Test deterministic action selector
    deterministic_selector = ContinuousActionSelector(deterministic=True)
    deterministic_actions = deterministic_selector(action_params_np)

    print(f"✅ Deterministic action selection successful")
    print(f"   Actions shape: {deterministic_actions.shape}")
    print(f"   Actions range: [{deterministic_actions.min():.3f}, {deterministic_actions.max():.3f}]")

    # Verify action shapes
    expected_action_shape = (batch_size, action_dim)
    if (stochastic_actions.shape == expected_action_shape and
        deterministic_actions.shape == expected_action_shape):
        print("✅ Action shapes are correct")
    else:
        print(f"❌ ERROR: Action shape mismatch")
        return False

    # Test that deterministic actions match the mean
    action_mean_np = action_mean.cpu().data.numpy()
    if np.allclose(deterministic_actions, action_mean_np):
        print("✅ Deterministic actions match action mean")
    else:
        print("❌ ERROR: Deterministic actions don't match action mean")
        return False

    # Test action bounds (should be in [-2, 2] for Pendulum)
    if (deterministic_actions.min() >= -2.1 and deterministic_actions.max() <= 2.1):
        print("✅ Actions are within expected bounds [-2, 2]")
    else:
        print(f"⚠️  WARNING: Actions outside expected bounds: [{deterministic_actions.min():.3f}, {deterministic_actions.max():.3f}]")

    print("\n✅ All PTAN compatibility tests passed!")
    print("The model should now work correctly with PTAN's ActorCriticAgent.")
    return True

if __name__ == "__main__":
    success = test_ptan_compatibility()
    if success:
        print("\n🎉 Ready to run the full training script!")
    else:
        print("\n❌ Fix needed before running full training.")
