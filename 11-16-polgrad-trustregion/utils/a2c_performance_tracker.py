"""
Performance tracking utilities for A2C training.
Provides comprehensive metrics tracking, logging, and summary reporting.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class PerformanceTracker:
    """Tracks and logs performance metrics during A2C training."""

    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time

        # Training metrics
        self.iteration_times: List[float] = []
        self.training_times: List[float] = []
        self.eval_times: List[float] = []
        self.returns: List[float] = []
        self.iterations: List[int] = []
        self.frame_indices: List[int] = []

        # Performance statistics
        self.best_return = float('-inf')
        self.best_iteration = 0
        self.total_frames = 0

        # Loss tracking (if provided)
        self.losses: List[float] = []
        self.critic_losses: List[float] = []
        self.actor_losses: List[float] = []
        self.entropy_losses: List[float] = []

    def log_iteration(self, iteration: int, frame_idx: int, avg_return: float,
                     training_time: float, eval_time: float = 0.0,
                     loss_dict: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Log metrics for a single training iteration."""
        current_time = time.time()

        # Update tracking lists
        self.iterations.append(iteration)
        self.frame_indices.append(frame_idx)
        self.returns.append(avg_return)
        self.training_times.append(training_time)
        self.eval_times.append(eval_time)

        # Track best performance
        if avg_return > self.best_return:
            self.best_return = avg_return
            self.best_iteration = iteration

        # Track losses if provided
        if loss_dict:
            self.losses.append(loss_dict.get('total_loss', 0.0))
            self.critic_losses.append(loss_dict.get('critic_loss', 0.0))
            self.actor_losses.append(loss_dict.get('actor_loss', 0.0))
            self.entropy_losses.append(loss_dict.get('entropy_loss', 0.0))

        # Calculate performance metrics
        total_elapsed = current_time - self.start_time
        iteration_time = training_time + eval_time
        self.iteration_times.append(iteration_time)

        # Calculate FPS (frames per second)
        if iteration_time > 0:
            current_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0
        else:
            current_fps = 0

        return {
            'avg_return': avg_return,
            'best_return': self.best_return,
            'current_fps': current_fps,
            'total_elapsed': total_elapsed,
            'iteration_time': iteration_time,
            'training_time': training_time,
            'eval_time': eval_time
        }

    def print_checkpoint(self, iteration: int, frame_idx: int):
        """Print a performance checkpoint summary."""
        if not self.returns:
            return

        current_time = time.time()
        total_elapsed = current_time - self.start_time
        checkpoint_elapsed = current_time - self.last_checkpoint_time

        # Calculate recent performance (last 100 iterations or all if less)
        recent_returns = self.returns[-100:] if len(self.returns) >= 100 else self.returns
        recent_times = self.iteration_times[-100:] if len(self.iteration_times) >= 100 else self.iteration_times

        avg_recent_return = np.mean(recent_returns)
        std_recent_return = np.std(recent_returns)
        avg_iteration_time = np.mean(recent_times) if recent_times else 0

        current_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        print(f"\n{'='*80}")
        print(f"CHECKPOINT - Iteration {iteration:,} | Frame {frame_idx:,}")
        print(f"{'='*80}")
        print(f"Performance:")
        print(f"  Current Return:     {self.returns[-1]:8.2f}")
        print(f"  Recent Avg (±std):  {avg_recent_return:8.2f} (±{std_recent_return:.2f})")
        print(f"  Best Return:        {self.best_return:8.2f} (iter {self.best_iteration:,})")
        print(f"")
        print(f"Timing:")
        print(f"  Total Time:         {total_elapsed/60:8.1f} minutes")
        print(f"  Since Last Check:   {checkpoint_elapsed/60:8.1f} minutes")
        print(f"  Avg Iter Time:      {avg_iteration_time:8.3f} seconds")
        print(f"  Current FPS:        {current_fps:8.1f}")

        if self.losses:
            print(f"")
            print(f"Recent Losses (avg of last 10):")
            recent_losses = self.losses[-10:]
            recent_critic = self.critic_losses[-10:] if self.critic_losses else [0]
            recent_actor = self.actor_losses[-10:] if self.actor_losses else [0]
            recent_entropy = self.entropy_losses[-10:] if self.entropy_losses else [0]

            print(f"  Total Loss:         {np.mean(recent_losses):8.4f}")
            print(f"  Critic Loss:        {np.mean(recent_critic):8.4f}")
            print(f"  Actor Loss:         {np.mean(recent_actor):8.4f}")
            print(f"  Entropy Loss:       {np.mean(recent_entropy):8.4f}")

        print(f"{'='*80}\n")

        self.last_checkpoint_time = current_time

    def get_summary(self) -> Dict[str, float]:
        """Get comprehensive training summary statistics."""
        if not self.returns:
            return {}

        total_time = time.time() - self.start_time

        summary = {
            'total_iterations': len(self.iterations),
            'total_time_minutes': total_time / 60,
            'total_frames': self.frame_indices[-1] if self.frame_indices else 0,
            'avg_fps': (self.frame_indices[-1] / total_time) if self.frame_indices and total_time > 0 else 0,
            'final_return': self.returns[-1],
            'best_return': self.best_return,
            'best_iteration': self.best_iteration,
            'avg_return': np.mean(self.returns),
            'std_return': np.std(self.returns),
            'avg_training_time': np.mean(self.training_times) if self.training_times else 0,
            'avg_eval_time': np.mean(self.eval_times) if self.eval_times else 0,
            'avg_iteration_time': np.mean(self.iteration_times) if self.iteration_times else 0,
        }

        if self.losses:
            summary.update({
                'final_total_loss': self.losses[-1],
                'avg_total_loss': np.mean(self.losses),
                'final_critic_loss': self.critic_losses[-1] if self.critic_losses else 0,
                'final_actor_loss': self.actor_losses[-1] if self.actor_losses else 0,
                'final_entropy_loss': self.entropy_losses[-1] if self.entropy_losses else 0,
            })

        return summary


def print_training_header(env_name: str, network_config: str, hyperparams: Dict[str, float]):
    """Print a formatted training header with configuration details."""
    print(f"\n{'='*80}")
    print(f"STARTING A2C TRAINING")
    print(f"{'='*80}")
    print(f"Environment:        {env_name}")
    print(f"Network:            {network_config}")
    print(f"Hyperparameters:")
    for key, value in hyperparams.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.6f}")
        else:
            print(f"  {key:15s}: {value}")
    print(f"Start Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


def print_final_summary(solved: bool, average_return: float, target_reward: float,
                       final_summary: Dict[str, float], frame_idx: int,
                       current_alpha, epsilon: float, iter_no: int):
    """Print comprehensive final training summary."""
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")

    # Training outcome
    status = "SOLVED" if solved else "NOT SOLVED"
    print(f"Status:             {status}")
    print(f"Final Return:       {average_return:8.2f}")
    print(f"Target Reward:      {target_reward:8.2f}")
    print(f"Best Return:        {final_summary.get('best_return', 0):8.2f} (iter {final_summary.get('best_iteration', 0):,})")
    print(f"")

    # Training statistics
    print(f"Training Statistics:")
    print(f"  Total Iterations:   {iter_no:8.0f}")
    print(f"  Total Frames:       {frame_idx:8,}")
    print(f"  Total Time:         {final_summary.get('total_time_minutes', 0):8.1f} minutes")
    print(f"  Average FPS:        {final_summary.get('avg_fps', 0):8.1f}")
    print(f"")

    # Performance statistics
    print(f"Performance Statistics:")
    print(f"  Average Return:     {final_summary.get('avg_return', 0):8.2f}")
    print(f"  Return Std Dev:     {final_summary.get('std_return', 0):8.2f}")
    print(f"  Avg Training Time:  {final_summary.get('avg_training_time', 0):8.3f}s")
    print(f"  Avg Eval Time:      {final_summary.get('avg_eval_time', 0):8.3f}s")
    print(f"")

    # Final hyperparameters
    print(f"Final Hyperparameters:")
    if isinstance(current_alpha, tuple):
        print(f"  Actor Learning Rate:      {current_alpha[0]:.6f}")
        print(f"  Critic Learning Rate:      {current_alpha[1]:.6f}")
    else:
        print(f"  Learning Rate:      {current_alpha:.6f}")
    print(f"  Epsilon:            {epsilon:.6f}")

    # Loss information (if available)
    if 'final_total_loss' in final_summary:
        print(f"")
        print(f"Final Losses:")
        print(f"  Total Loss:         {final_summary['final_total_loss']:8.4f}")
        print(f"  Critic Loss:        {final_summary.get('final_critic_loss', 0):8.4f}")
        print(f"  Actor Loss:         {final_summary.get('final_actor_loss', 0):8.4f}")
        print(f"  Entropy Loss:       {final_summary.get('final_entropy_loss', 0):8.4f}")

    print(f"")
    print(f"Completed:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
