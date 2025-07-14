"""
Performance tracking utilities for deep reinforcement learning training.

This module provides tools to monitor and analyze training performance including:
- Wallclock time tracking
- Frames per second (FPS) calculation
- Training iteration timing
- Reward progression monitoring
- Comprehensive performance summaries
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class PerformanceTracker:
    """Track training performance metrics including timing and FPS.
    
    This class provides comprehensive performance monitoring for RL training,
    tracking wallclock time, processing speed, and reward progression.
    
    Attributes:
        start_time: Training start timestamp
        last_log_time: Last logging timestamp for FPS calculation
        last_frame_idx: Last frame index for FPS calculation
        episode_times: List of episode timing data
        episode_rewards: List of episode rewards for progression tracking
        training_times: List of training iteration times
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_frame_idx = 0
        self.episode_times: List[float] = []
        self.episode_rewards: List[float] = []
        self.training_times: List[float] = []
        
    def log_iteration(self, iter_no: int, frame_idx: int, avg_return: float, 
                     training_time: Optional[float] = None) -> Dict[str, float]:
        """Log performance metrics for current iteration.
        
        Args:
            iter_no: Current iteration number
            frame_idx: Current frame index
            avg_return: Average return/reward for this iteration
            training_time: Time taken for this training iteration
            
        Returns:
            Dictionary containing calculated performance metrics
        """
        current_time = time.time()
        
        # Calculate time metrics
        total_elapsed = current_time - self.start_time
        time_since_last_log = current_time - self.last_log_time
        frames_since_last_log = frame_idx - self.last_frame_idx
        
        # Calculate FPS
        if time_since_last_log > 0:
            current_fps = frames_since_last_log / time_since_last_log
        else:
            current_fps = 0
            
        overall_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0
        
        # Store metrics
        self.episode_rewards.append(avg_return)
        if training_time is not None:
            self.training_times.append(training_time)
        
        # Update tracking variables
        self.last_log_time = current_time
        self.last_frame_idx = frame_idx
        
        return {
            'total_elapsed': total_elapsed,
            'current_fps': current_fps,
            'overall_fps': overall_fps,
            'time_since_last': time_since_last_log
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary containing comprehensive performance metrics
        """
        total_time = time.time() - self.start_time
        avg_training_time = sum(self.training_times) / len(self.training_times) if self.training_times else 0
        
        return {
            'total_wallclock_time': total_time,
            'total_time_str': str(timedelta(seconds=int(total_time))),
            'avg_training_time_per_iter': avg_training_time,
            'max_reward_achieved': max(self.episode_rewards) if self.episode_rewards else -float('inf'),
            'final_avg_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'total_iterations': len(self.episode_rewards)
        }
    
    def reset(self):
        """Reset all tracking metrics."""
        self.__init__()
    
    def get_reward_progression(self) -> List[float]:
        """Get the complete reward progression history.
        
        Returns:
            List of all recorded episode rewards
        """
        return self.episode_rewards.copy()
    
    def get_training_times(self) -> List[float]:
        """Get the complete training time history.
        
        Returns:
            List of all recorded training times
        """
        return self.training_times.copy()
    
    def print_checkpoint(self, iter_no: int, frame_idx: int):
        """Print a formatted checkpoint summary.
        
        Args:
            iter_no: Current iteration number
            frame_idx: Current frame index
        """
        current_summary = self.get_summary()
        print(f"📊 [Checkpoint] Iter {iter_no}: "
              f"Time={current_summary['total_time_str']}, "
              f"Max_reward={current_summary['max_reward_achieved']:.1f}, "
              f"Avg_FPS={frame_idx / current_summary['total_wallclock_time']:.1f}")


def print_training_header(env_name: str, network_config: str, hyperparams: Dict[str, any]):
    """Print formatted training header with configuration details.
    
    Args:
        env_name: Name of the environment being trained on
        network_config: Description of network architecture
        hyperparams: Dictionary of key hyperparameters
    """
    print(f"=== Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Environment: {env_name}")
    print(f"Network: {network_config}")
    print(f"Hyperparameters: {hyperparams}")
    print("=" * 80)


def print_final_summary(solved: bool, average_return: float, target_reward: float,
                       final_summary: Dict[str, any], frame_idx: int, 
                       current_alpha: float, beta: float, epsilon: float,
                       iter_no: int, tgt_net_sync_iters: int):
    """Print comprehensive final training summary.
    
    Args:
        solved: Whether the environment was solved
        average_return: Final average return achieved
        target_reward: Target reward for solving
        final_summary: Performance summary from PerformanceTracker
        frame_idx: Total frames processed
        current_alpha: Final priority buffer alpha
        beta: Final importance sampling beta
        epsilon: Final exploration epsilon
        iter_no: Total iterations completed
        tgt_net_sync_iters: Target network sync interval
    """
    print("\n" + "=" * 80)
    if solved:
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"✅ Environment SOLVED with average reward: {average_return:.2f}")
    else:
        print("⏹️  TRAINING STOPPED")
        print(f"❌ Final average reward: {average_return:.2f} (target: {target_reward})")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Total wallclock time: {final_summary['total_time_str']}")
    print(f"   Total iterations: {final_summary['total_iterations']:,}")
    print(f"   Total frames processed: {frame_idx:,}")
    print(f"   Average FPS: {frame_idx / final_summary['total_wallclock_time']:.1f}")
    print(f"   Average training time per iteration: {final_summary['avg_training_time_per_iter']:.3f}s")
    print(f"   Maximum reward achieved: {final_summary['max_reward_achieved']:.2f}")
    print(f"   Final network sync iterations: {iter_no // tgt_net_sync_iters}")
    
    print(f"\n🔧 FINAL HYPERPARAMETER STATE:")
    print(f"   Priority Alpha: {current_alpha:.3f}")
    print(f"   Beta (IS correction): {beta:.3f}")
    print(f"   Epsilon (exploration): {epsilon:.3f}")
    
    print(f"\n📈 TRAINING EFFICIENCY:")
    if final_summary['total_wallclock_time'] > 0:
        frames_per_minute = (frame_idx / final_summary['total_wallclock_time']) * 60
        iterations_per_minute = final_summary['total_iterations'] / (final_summary['total_wallclock_time'] / 60)
        print(f"   Frames per minute: {frames_per_minute:.0f}")
        print(f"   Iterations per minute: {iterations_per_minute:.1f}")
        if solved:
            print(f"   Time to solve: {final_summary['total_time_str']}")
    
    print("=" * 80)