#!/usr/bin/env python3
"""
Run KalmanTrack Hybrid Experiments
Tests hybrid CoTracker3 + Kalman tracker on synthetic data
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import imageio.v3 as iio
from typing import Dict, List

from synthetic_simulator import BouncingBallSimulator, create_test_scenarios
from test_synthetic_tracking import SyntheticVideoGenerator
from kalman_hybrid import KalmanTrackHybrid, HybridTrackingResult


def run_hybrid_tracking(video_path: str, initial_points: np.ndarray,
                       N: int = 5, scenario_name: str = 'test') -> Dict:
    """
    Run hybrid tracker on video
    
    Args:
        video_path: Path to video
        initial_points: Initial point positions [N_points, 2]
        N: Run CoTracker3 every N frames
        scenario_name: Scenario name
        
    Returns:
        Dictionary with tracking results
    """
    print(f"\n{'='*60}")
    print(f"Running Hybrid Tracker (N={N})")
    print(f"{'='*60}")
    
    # Load video
    print("Loading video...")
    video_frames = iio.imread(video_path, plugin="FFMPEG")
    T, H, W = video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]
    
    # Convert to tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)[None].float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_tensor = video_tensor.to(device)
    
    # Prepare queries [1, N_points, 3] where each row is [t, x, y]
    n_points = initial_points.shape[0]
    queries = np.zeros((n_points, 3))
    queries[:, 0] = 0  # Start frame
    queries[:, 1:] = initial_points
    queries_tensor = torch.from_numpy(queries)[None].float().to(device)
    
    print(f"Video: {T} frames, {H}x{W}")
    print(f"Tracking {n_points} points")
    print(f"CoTracker3 frequency: every {N} frames")
    
    # Initialize hybrid tracker
    tracker = KalmanTrackHybrid(
        N=N,
        process_var_pos=1e-4,
        process_var_vel=1e-2,
        meas_var_pos=0.1,
        device=device
    )
    
    tracker.initialize(video_tensor, initial_points)
    
    # Track all frames
    results = []
    print("\nTracking frames...")
    
    for frame_idx in range(T):
        result = tracker.track_frame(video_tensor, queries_tensor, frame_idx)
        results.append(result)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{T} - "
                  f"{'CoTracker3' if result.used_cotracker else 'Kalman'} - "
                  f"{result.processing_time*1000:.1f}ms")
    
    # Get statistics
    stats = tracker.get_statistics()
    
    print(f"\n{'='*60}")
    print("Tracking Statistics:")
    print(f"{'='*60}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"CoTracker3 calls: {stats['cotracker_calls']}")
    print(f"Kalman predictions: {stats['kalman_predictions']}")
    print(f"Speedup factor: {stats['speedup_factor']:.2f}×")
    print(f"CoTracker3 frequency: {stats['cotracker_frequency']}")
    
    # Extract tracking data
    positions = np.array([r.positions[0] for r in results])  # [T, 2] for first point
    velocities = np.array([r.velocities[0] for r in results])
    used_cotracker = np.array([r.used_cotracker for r in results])
    processing_times = np.array([r.processing_time for r in results])
    
    return {
        'positions': positions,
        'velocities': velocities,
        'used_cotracker': used_cotracker,
        'processing_times': processing_times,
        'stats': stats,
        'N': N,
        'scenario': scenario_name
    }


def compare_hybrid_vs_baseline(simulation_result: Dict,
                               cotracker_result: Dict,
                               hybrid_results: List[Dict],
                               scenario_name: str):
    """
    Compare hybrid tracker with different N values against CoTracker3 baseline
    
    Args:
        simulation_result: Ground truth
        cotracker_result: CoTracker3 baseline
        hybrid_results: List of hybrid results with different N
        scenario_name: Scenario name
    """
    print(f"\n{'='*60}")
    print("Comparison: Hybrid vs Baseline")
    print(f"{'='*60}")
    
    # Ground truth
    true_states = simulation_result['true_states']
    time = simulation_result['time']
    
    # Convert to pixels
    video_gen = SyntheticVideoGenerator()
    true_px = np.array([video_gen.meters_to_pixels(s[0], s[1]) for s in true_states])
    
    # CoTracker3 baseline
    cotracker_px = cotracker_result['tracks'][:, 0, :]
    cotracker_error = np.sqrt((true_px[:, 0] - cotracker_px[:, 0])**2 + 
                             (true_px[:, 1] - cotracker_px[:, 1])**2)
    cotracker_rmse = np.sqrt(np.mean(cotracker_error**2))
    
    # Create comparison figure
    n_hybrid = len(hybrid_results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different N values
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: 2D Trajectories
    ax = axes[0, 0]
    ax.plot(true_px[:, 0], true_px[:, 1], 'k-', label='Ground Truth', 
            linewidth=3, alpha=0.7)
    ax.plot(cotracker_px[:, 0], cotracker_px[:, 1], 'gray', 
            label='CoTracker3 (baseline)', linewidth=2, linestyle='--', alpha=0.5)
    
    for i, hybrid in enumerate(hybrid_results):
        hybrid_px = hybrid['positions']
        ax.plot(hybrid_px[:, 0], hybrid_px[:, 1], 
               color=colors[i % len(colors)], 
               label=f"Hybrid (N={hybrid['N']})", 
               linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title('2D Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Plot 2: Error over time
    ax = axes[0, 1]
    ax.plot(time, cotracker_error, 'gray', label='CoTracker3', 
            linewidth=2, linestyle='--', alpha=0.5)
    
    for i, hybrid in enumerate(hybrid_results):
        hybrid_px = hybrid['positions']
        hybrid_error = np.sqrt((true_px[:, 0] - hybrid_px[:, 0])**2 + 
                              (true_px[:, 1] - hybrid_px[:, 1])**2)
        ax.plot(time, hybrid_error, color=colors[i % len(colors)],
               label=f"Hybrid (N={hybrid['N']})", linewidth=2, alpha=0.7)
        
        # Mark CoTracker3 update frames
        update_frames = np.where(hybrid['used_cotracker'])[0]
        ax.scatter(time[update_frames], hybrid_error[update_frames], 
                  color=colors[i % len(colors)], s=20, alpha=0.3, marker='o')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tracking Error (pixels)', fontsize=12)
    ax.set_title('Tracking Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMSE Comparison
    ax = axes[1, 0]
    methods = ['CoTracker3\n(baseline)']
    rmses = [cotracker_rmse]
    
    for hybrid in hybrid_results:
        hybrid_px = hybrid['positions']
        hybrid_error = np.sqrt((true_px[:, 0] - hybrid_px[:, 0])**2 + 
                              (true_px[:, 1] - hybrid_px[:, 1])**2)
        hybrid_rmse = np.sqrt(np.mean(hybrid_error**2))
        methods.append(f"Hybrid\n(N={hybrid['N']})")
        rmses.append(hybrid_rmse)
    
    bars = ax.bar(methods, rmses, color=['gray'] + colors[:n_hybrid], alpha=0.7)
    ax.set_ylabel('RMSE (pixels)', fontsize=12)
    ax.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rmse:.1f}px', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Speedup vs Accuracy
    ax = axes[1, 1]
    speedups = []
    accuracies = []
    
    for hybrid in enumerate(hybrid_results):
        speedup = hybrid['stats']['speedup_factor']
        hybrid_px = hybrid['positions']
        hybrid_error = np.sqrt((true_px[:, 0] - hybrid_px[:, 0])**2 + 
                              (true_px[:, 1] - hybrid_px[:, 1])**2)
        hybrid_rmse = np.sqrt(np.mean(hybrid_error**2))
        accuracy_retention = (1 - (hybrid_rmse - cotracker_rmse) / cotracker_rmse) * 100
        
        speedups.append(speedup)
        accuracies.append(accuracy_retention)
        
        ax.scatter(speedup, accuracy_retention, 
                  color=colors[i % len(colors)], s=200, alpha=0.7,
                  label=f"N={hybrid['N']}")
        ax.text(speedup, accuracy_retention, f"N={hybrid['N']}", 
               ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Speedup Factor', fontsize=12)
    ax.set_ylabel('Accuracy Retention (%)', fontsize=12)
    ax.set_title('Speedup vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f'outputs/hybrid_comparison_{scenario_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'RMSE (px)':<15} {'Speedup':<15} {'Accuracy':<15}")
    print(f"{'='*80}")
    print(f"{'CoTracker3 (baseline)':<25} {cotracker_rmse:<15.2f} {'1.00×':<15} {'100.0%':<15}")
    
    for i, hybrid in enumerate(hybrid_results):
        hybrid_px = hybrid['positions']
        hybrid_error = np.sqrt((true_px[:, 0] - hybrid_px[:, 0])**2 + 
                              (true_px[:, 1] - hybrid_px[:, 1])**2)
        hybrid_rmse = np.sqrt(np.mean(hybrid_error**2))
        speedup = hybrid['stats']['speedup_factor']
        accuracy_retention = (1 - (hybrid_rmse - cotracker_rmse) / cotracker_rmse) * 100
        
        print(f"{'Hybrid (N=' + str(hybrid['N']) + ')':<25} "
              f"{hybrid_rmse:<15.2f} "
              f"{speedup:<15.2f}× "
              f"{accuracy_retention:<15.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    
    print("="*60)
    print("KalmanTrack Hybrid Experiments")
    print("="*60)
    
    # Create scenario
    scenarios = create_test_scenarios()
    scenario_name = 'nominal'
    config = scenarios[scenario_name]
    
    print(f"\nScenario: {scenario_name}")
    
    # Generate or load synthetic data
    print("\n1. Generating synthetic data...")
    simulator = BouncingBallSimulator(config)
    result = simulator.simulate()
    
    # Generate video
    video_path = f'outputs/synthetic_{scenario_name}.mp4'
    if not os.path.exists(video_path):
        print("\n2. Generating video...")
        video_gen = SyntheticVideoGenerator()
        video_gen.generate_video(result, video_path, fps=30)
    else:
        print(f"\n2. Using existing video: {video_path}")
    
    # Get initial position
    initial_x, initial_y = result['true_states'][0, :2]
    video_gen = SyntheticVideoGenerator()
    initial_px = video_gen.meters_to_pixels(initial_x, initial_y)
    initial_points = np.array([initial_px])
    
    # Load CoTracker3 baseline (if not already done)
    print("\n3. Loading CoTracker3 baseline...")
    from test_cotracker3_synthetic import track_with_cotracker3
    cotracker_result = track_with_cotracker3(video_path, initial_points)
    
    # Run hybrid tracker with different N values
    N_values = [3, 5, 10]
    hybrid_results = []
    
    for N in N_values:
        print(f"\n4. Running Hybrid Tracker (N={N})...")
        hybrid_result = run_hybrid_tracking(video_path, initial_points, N, scenario_name)
        hybrid_results.append(hybrid_result)
    
    # Compare results
    print("\n5. Generating comparison plots...")
    compare_hybrid_vs_baseline(result, cotracker_result, hybrid_results, scenario_name)
    
    print("\n✓ Experiments complete!")
    print("Check outputs/ directory for results.")
