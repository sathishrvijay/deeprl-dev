#!/usr/bin/env python3
"""
Test script to verify the action selector handles both single env and batch cases correctly.
"""

import numpy as np
import sys
sys.path.append('/Users/vsathish/Development/deeprl-dev/11-16-polgrad-trustregion')

def test_action_selector_shapes():
    """Test that action selector handles different input shapes correctly."""
    print("🔍 Testing Action Selector Shape Handling...")
    
    from pendulumv1_cont_a2c_ptan import ContinuousActionSelector
    
    # Test data for Pendulum (action_dim = 1)
    action_dim = 1
    
    # Test Case 1: Single environment (1D input)
    print("\n📍 Test Case 1: Single Environment")
    single_mean = np.array([0.5])  # action mean
    single_log_var = np.array([-1.0])  # action log variance
    single_scores = np.concatenate([single_mean, single_log_var])  # Shape: (2,)
    
    print(f"Input shape: {single_scores.shape}")
    print(f"Input content: {single_scores}")
    
    selector = ContinuousActionSelector(deterministic=True)
    single_action = selector(single_scores)
    
    print(f"Output shape: {single_action.shape}")
    print(f"Output content: {single_action}")
    print(f"Expected for Pendulum: 1D array with shape (1,)")
    
    # Verify shape is correct for Pendulum
    if single_action.shape == (1,):
        print("✅ Single environment shape is correct")
    else:
        print(f"❌ Single environment shape is wrong: {single_action.shape}")
        return False
    
    # Test Case 2: Batch of environments (2D input)
    print("\n📍 Test Case 2: Batch of Environments")
    batch_size = 16
    batch_mean = np.random.randn(batch_size, action_dim) * 0.5
    batch_log_var = np.random.randn(batch_size, action_dim) * 0.1 - 1.0
    batch_scores = np.concatenate([batch_mean, batch_log_var], axis=1)  # Shape: (16, 2)
    
    print(f"Input shape: {batch_scores.shape}")
    print(f"Input content (first 3): {batch_scores[:3]}")
    
    batch_actions = selector(batch_scores)
    
    print(f"Output shape: {batch_actions.shape}")
    print(f"Output content (first 3): {batch_actions[:3]}")
    print(f"Expected for batch: 2D array with shape ({batch_size}, {action_dim})")
    
    # Verify shape is correct for batch
    if batch_actions.shape == (batch_size, action_dim):
        print("✅ Batch environment shape is correct")
    else:
        print(f"❌ Batch environment shape is wrong: {batch_actions.shape}")
        return False
    
    # Test Case 3: Stochastic vs Deterministic
    print("\n📍 Test Case 3: Stochastic vs Deterministic")
    stoch_selector = ContinuousActionSelector(deterministic=False)
    det_selector = ContinuousActionSelector(deterministic=True)
    
    stoch_action = stoch_selector(single_scores)
    det_action = det_selector(single_scores)
    
    print(f"Stochastic action: {stoch_action}")
    print(f"Deterministic action: {det_action}")
    print(f"Expected deterministic: {single_mean}")  # Should match the mean
    
    if np.allclose(det_action, single_mean, atol=1e-6):
        print("✅ Deterministic action matches mean")
    else:
        print(f"❌ Deterministic action doesn't match mean")
        return False
    
    # Test Case 4: Edge cases
    print("\n📍 Test Case 4: Edge Cases")
    
    # Empty array
    try:
        empty_scores = np.array([])
        empty_action = selector(empty_scores)
        print(f"Empty input handled: {empty_action}")
        if empty_action.shape == (1,):
            print("✅ Empty input handled correctly")
        else:
            print("❌ Empty input not handled correctly")
    except Exception as e:
        print(f"❌ Empty input caused error: {e}")
        return False
    
    print("\n✅ All action selector shape tests passed!")
    return True

if __name__ == "__main__":
    success = test_action_selector_shapes()
    if success:
        print("\n🎉 Action selector is ready for both single env and batch processing!")
    else:
        print("\n❌ Action selector needs fixes.")