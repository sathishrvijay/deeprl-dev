#!/usr/bin/env python3
"""Test the refactored performance tracking utilities"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import PerformanceTracker, print_training_header, print_final_summary
import time

def test_refactored_utils():
    """Test the refactored performance tracking utilities."""
    print("=== Testing Refactored Performance Utilities ===\n")
    
    # Test 1: Training header
    print("1. Testing print_training_header:")
    network_config = "128-64 network"
    hyperparams = {
        'lr': 1e-3,
        'batch_size': 32,
        'warmup_frames': 15000,
        'ramp_frames': 20000
    }
    print_training_header("LunarLander-v2", network_config, hyperparams)
    
    # Test 2: Performance tracker
    print("\n2. Testing PerformanceTracker:")
    perf_tracker = PerformanceTracker()
    
    # Simulate a few iterations
    for i in range(1, 4):
        time.sleep(0.1)  # Simulate training time
        training_time = 0.1
        frame_idx = i * 10
        avg_return = -300 + (i * 50)
        
        perf_metrics = perf_tracker.log_iteration(i, frame_idx, avg_return, training_time)
        
        print(f"Iter {i}: FPS={perf_metrics['current_fps']:.1f}, "
              f"Total time={perf_metrics['total_elapsed']:.1f}s")
        
        # Test checkpoint at iteration 2
        if i == 2:
            perf_tracker.print_checkpoint(i, frame_idx)
    
    # Test 3: Final summary
    print("\n3. Testing print_final_summary:")
    final_summary = perf_tracker.get_summary()
    print_final_summary(
        solved=True,
        average_return=215.5,
        target_reward=200.0,
        final_summary=final_summary,
        frame_idx=30,
        current_alpha=0.6,
        beta=0.8,
        epsilon=0.02,
        iter_no=3,
        tgt_net_sync_iters=100
    )
    
    print("\n✅ All utility functions working correctly!")
    print("✅ PerformanceTracker successfully moved to utils module")
    print("✅ Code is now modular and reusable")

if __name__ == "__main__":
    test_refactored_utils()