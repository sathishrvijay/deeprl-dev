#!/usr/bin/env python3
"""Visualize the gradual alpha ramp and epsilon decay schedules"""

import matplotlib.pyplot as plt
import numpy as np

# Parameters from the modified code
EPSILON_DECAY_FRAMES = 50000
PRIORITY_BUF_WARMUP_FRAMES = 15000
PRIORITY_BUF_RAMP_FRAMES = 20000
PRIORITY_BUF_ALPHA = 0.6
MIN_EPSILON = 0.02

def get_priority_alpha(frame_idx: int) -> float:
    """Calculate priority buffer alpha with gradual ramp-up."""
    if frame_idx < PRIORITY_BUF_WARMUP_FRAMES:
        return 0.0
    elif frame_idx < PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES:
        ramp_progress = (frame_idx - PRIORITY_BUF_WARMUP_FRAMES) / PRIORITY_BUF_RAMP_FRAMES
        return ramp_progress * PRIORITY_BUF_ALPHA
    else:
        return PRIORITY_BUF_ALPHA

def get_epsilon(frame_idx: int) -> float:
    """Calculate epsilon value."""
    return max(MIN_EPSILON, 1.0 - float(frame_idx) / float(EPSILON_DECAY_FRAMES))

# Generate data
frames = np.arange(0, 60000, 100)
alphas = [get_priority_alpha(f) for f in frames]
epsilons = [get_epsilon(f) for f in frames]

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Alpha ramp plot
ax1.plot(frames, alphas, 'b-', linewidth=2, label='Priority Alpha')
ax1.axvline(x=PRIORITY_BUF_WARMUP_FRAMES, color='r', linestyle='--', alpha=0.7, label='Warmup End')
ax1.axvline(x=PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES, color='g', linestyle='--', alpha=0.7, label='Ramp End')
ax1.set_ylabel('Priority Alpha')
ax1.set_title('Gradual Priority Buffer Alpha Ramp')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(-0.05, 0.65)

# Add phase annotations
ax1.annotate('Phase 1: Pure Exploration\n(Uniform Sampling)', 
             xy=(7500, 0.3), ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
ax1.annotate('Phase 2: Guided Learning\n(Gradual Prioritization)', 
             xy=(25000, 0.3), ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
ax1.annotate('Phase 3: Exploitation\n(Full Prioritization)', 
             xy=(47500, 0.3), ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

# Epsilon decay plot
ax2.plot(frames, epsilons, 'r-', linewidth=2, label='Epsilon')
ax2.axvline(x=PRIORITY_BUF_WARMUP_FRAMES, color='r', linestyle='--', alpha=0.7, label='Warmup End')
ax2.axvline(x=PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES, color='g', linestyle='--', alpha=0.7, label='Ramp End')
ax2.set_xlabel('Training Frames')
ax2.set_ylabel('Epsilon')
ax2.set_title('Epsilon Decay Schedule')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('priority_alpha_epsilon_schedules.png', dpi=150, bbox_inches='tight')
plt.show()

# Print key transition points
print("=== Training Phase Transitions ===")
print(f"Phase 1 (Pure Exploration): 0 - {PRIORITY_BUF_WARMUP_FRAMES:,} frames")
print(f"  - Alpha: 0.0 (uniform sampling)")
print(f"  - Epsilon: 1.0 → {get_epsilon(PRIORITY_BUF_WARMUP_FRAMES):.2f}")

print(f"\nPhase 2 (Guided Learning): {PRIORITY_BUF_WARMUP_FRAMES:,} - {PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES:,} frames")
print(f"  - Alpha: 0.0 → {PRIORITY_BUF_ALPHA}")
print(f"  - Epsilon: {get_epsilon(PRIORITY_BUF_WARMUP_FRAMES):.2f} → {get_epsilon(PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES):.2f}")

print(f"\nPhase 3 (Exploitation): {PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES:,}+ frames")
print(f"  - Alpha: {PRIORITY_BUF_ALPHA} (full prioritization)")
print(f"  - Epsilon: {get_epsilon(PRIORITY_BUF_WARMUP_FRAMES + PRIORITY_BUF_RAMP_FRAMES):.2f} → {MIN_EPSILON}")