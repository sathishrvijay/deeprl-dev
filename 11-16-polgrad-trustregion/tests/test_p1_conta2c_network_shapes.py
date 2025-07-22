#!/usr/bin/env python3
"""
Test script to verify network output shapes and identify the issue.
"""

import torch
import sys
sys.path.append('/Users/vsathish/Development/deeprl-dev/11-16-polgrad-trustregion')

def test_network_shapes():
    """Test that the network produces correct output shapes."""
    print("🔍 Testing Network Output Shapes...")
    
    from models.pgtr_models import ContinuousA2C
    
    # Pendulum specs
    state_dim = 3
    action_dim = 1
    batch_size = 16
    
    # Create network
    net = ContinuousA2C(state_dim, action_dim)
    print(f"Network created with action_dim={action_dim}")
    
    # Test input
    states = torch.randn(batch_size, state_dim)
    print(f"Input states shape: {states.shape}")
    
    # Forward pass
    try:
        value, action_params = net(states)
        
        print(f"✅ Forward pass successful")
        print(f"Value shape: {value.shape}")
        print(f"Action params shape: {action_params.shape}")
        print(f"Expected action params shape: ({batch_size}, {2 * action_dim})")
        
        # Check if shapes are correct
        if action_params.shape == (batch_size, 2 * action_dim):
            print("✅ Action params shape is correct")
        else:
            print(f"❌ Action params shape is wrong!")
            print(f"   Expected: ({batch_size}, {2 * action_dim})")
            print(f"   Got: {action_params.shape}")
            return False
            
        # Test action distribution method
        action_mean, action_log_var = net.get_action_distribution(states)
        print(f"Action mean shape: {action_mean.shape}")
        print(f"Action log_var shape: {action_log_var.shape}")
        
        # Verify concatenation works
        manual_concat = torch.cat([action_mean, action_log_var], dim=-1)
        if torch.allclose(action_params, manual_concat):
            print("✅ Concatenation is working correctly")
        else:
            print("❌ Concatenation issue detected")
            return False
            
        # Test single state (like PTAN would use)
        single_state = torch.randn(1, state_dim)
        single_value, single_action_params = net(single_state)
        
        print(f"Single state test:")
        print(f"   Input shape: {single_state.shape}")
        print(f"   Value shape: {single_value.shape}")
        print(f"   Action params shape: {single_action_params.shape}")
        print(f"   Expected: (1, {2 * action_dim})")
        
        if single_action_params.shape == (1, 2 * action_dim):
            print("✅ Single state output is correct")
        else:
            print(f"❌ Single state output is wrong!")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All network shape tests passed!")
    return True

if __name__ == "__main__":
    success = test_network_shapes()
    if success:
        print("\n🎉 Network is producing correct output shapes!")
    else:
        print("\n❌ Network has shape issues that need fixing.")