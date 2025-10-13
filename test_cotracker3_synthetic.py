#!/usr/bin/env python3
"""
Test CoTracker3 on Synthetic Bouncing Ball Data
Uses the hybrid CoTracker3 implementation to track synthetic data
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add hybrid tracker to path
HYBRID_PATH = os.path.join(os.path.dirname(__file__), 'klamantrack_hybrid_v1')
if HYBRID_PATH not in sys.path:
    sys.path.insert(0, HYBRID_PATH)

from synthetic_simulator import BouncingBallSimulator, SimulationConfig, create_test_scenarios
from test_synthetic_tracking import SyntheticVideoGenerator
import imageio.v3 as iio


def track_with_cotracker3(video_path: str, initial_points: np.ndarray,
                         device: str = 'cuda') -> dict:
    """
    Track points using CoTracker3
    
    Args:
        video_path: Path to video file
        initial_points: Initial point positions [N, 2] in (x, y) format
        device: Device to run on
        
    Returns:
        Dictionary with tracking results
    """
    print(f"\n=== Tracking with CoTracker3 ===")
    print(f"Video: {video_path}")
    print(f"Initial points: {initial_points.shape}")
    
    # Load CoTracker3 model (use offline for synthetic data - simpler API)
    print("Loading CoTracker3 offline model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    cotracker = cotracker.to(device)
    cotracker.eval()
    print(f"CoTracker3 offline model loaded on {device}")
    
    # Load video
    print("Loading video...")
    video_frames = iio.imread(video_path, plugin="FFMPEG")
    print(f"Video shape: {video_frames.shape}")
    
    # Convert to tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)[None].float()
    video_tensor = video_tensor.to(device)
    
    T, H, W = video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]
    
    # Prepare queries: [1, N, 3] where each row is [t, x, y]
    # Track from frame 0
    N = initial_points.shape[0]
    queries = np.zeros((N, 3))
    queries[:, 0] = 0  # Start frame
    queries[:, 1:] = initial_points  # x, y positions
    queries_tensor = torch.from_numpy(queries)[None].float().to(device)
    
    print(f"Queries shape: {queries_tensor.shape}")
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Run CoTracker3 offline model
    print("Running CoTracker3 offline...")
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video_tensor, queries=queries_tensor)
    
    # Convert to numpy
    tracks = pred_tracks[0].cpu().numpy()  # [T, N, 2]
    visibility = pred_visibility[0].cpu().numpy()  # [T, N]
    
    print(f"Tracks shape: {tracks.shape}")
    print(f"Visibility shape: {visibility.shape}")
    
    return {
        'tracks': tracks,
        'visibility': visibility,
        'video_shape': (T, H, W)
    }


def create_tracking_videos(video_path: str,
                          simulation_result: dict,
                          cotracker_result: dict,
                          kalman_result: dict,
                          scenario_name: str):
    """
    Create videos showing CoTracker3 and KalmanTrack tracking
    
    Args:
        video_path: Path to original video
        simulation_result: Ground truth data
        cotracker_result: CoTracker3 results
        kalman_result: KalmanTrack results
        scenario_name: Scenario name
    """
    print("\nCreating tracking visualization videos...")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    cotracker_video_path = f'outputs/cotracker3_{scenario_name}.mp4'
    kalman_video_path = f'outputs/kalmantrack_{scenario_name}.mp4'
    comparison_video_path = f'outputs/comparison_{scenario_name}.mp4'
    
    out_cotracker = cv2.VideoWriter(cotracker_video_path, fourcc, fps, (width, height))
    out_kalman = cv2.VideoWriter(kalman_video_path, fourcc, fps, (width, height))
    out_comparison = cv2.VideoWriter(comparison_video_path, fourcc, fps, (width * 2, height))
    
    # Get tracking data
    cotracker_tracks = cotracker_result['tracks'][:, 0, :]  # [T, 2]
    kalman_positions = kalman_result['tracked_positions']
    
    # Convert ground truth to pixels
    video_gen = SyntheticVideoGenerator()
    true_states = simulation_result['true_states']
    true_px = np.array([video_gen.meters_to_pixels(s[0], s[1]) for s in true_states])
    
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create frames for each tracker
        frame_cotracker = frame.copy()
        frame_kalman = frame.copy()
        
        # Draw ground truth (green)
        gt_pos = (int(true_px[frame_idx, 0]), int(true_px[frame_idx, 1]))
        cv2.circle(frame_cotracker, gt_pos, 8, (0, 255, 0), 2)
        cv2.circle(frame_kalman, gt_pos, 8, (0, 255, 0), 2)
        
        # Draw CoTracker3 prediction (blue)
        ct_pos = (int(cotracker_tracks[frame_idx, 0]), int(cotracker_tracks[frame_idx, 1]))
        cv2.circle(frame_cotracker, ct_pos, 6, (255, 0, 0), -1)
        cv2.line(frame_cotracker, gt_pos, ct_pos, (255, 255, 0), 1)
        
        # Draw KalmanTrack prediction (red)
        if kalman_positions[frame_idx] is not None:
            kt_pos = kalman_positions[frame_idx]
            cv2.circle(frame_kalman, kt_pos, 6, (0, 0, 255), -1)
            cv2.line(frame_kalman, gt_pos, kt_pos, (255, 255, 0), 1)
        
        # Add labels
        cv2.putText(frame_cotracker, "CoTracker3", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_cotracker, f"Frame: {frame_idx}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame_kalman, "KalmanTrack (DINO)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_kalman, f"Frame: {frame_idx}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend
        cv2.putText(frame_cotracker, "Green=GT, Blue=Predicted", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_kalman, "Green=GT, Red=Predicted", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write individual videos
        out_cotracker.write(frame_cotracker)
        out_kalman.write(frame_kalman)
        
        # Create side-by-side comparison
        comparison_frame = np.hstack([frame_cotracker, frame_kalman])
        out_comparison.write(comparison_frame)
        
        frame_idx += 1
    
    cap.release()
    out_cotracker.release()
    out_kalman.release()
    out_comparison.release()
    
    print(f"  CoTracker3 video: {cotracker_video_path}")
    print(f"  KalmanTrack video: {kalman_video_path}")
    print(f"  Comparison video: {comparison_video_path}")


def compare_trackers(simulation_result: dict, 
                    cotracker_result: dict,
                    kalman_result: dict,
                    scenario_name: str):
    """
    Compare CoTracker3 and KalmanTrack against ground truth
    
    Args:
        simulation_result: Ground truth from simulator
        cotracker_result: CoTracker3 tracking results
        kalman_result: KalmanTrack results
        scenario_name: Name for saving
    """
    true_states = simulation_result['true_states']
    time = simulation_result['time']
    
    # Convert ground truth to pixels
    video_gen = SyntheticVideoGenerator()
    true_px = np.array([video_gen.meters_to_pixels(s[0], s[1]) for s in true_states])
    
    # CoTracker3 tracks (already in pixels, tracking first point)
    cotracker_px = cotracker_result['tracks'][:, 0, :]  # [T, 2]
    cotracker_vis = cotracker_result['visibility'][:, 0]  # [T]
    
    # KalmanTrack positions
    kalman_px = []
    for pos in kalman_result['tracked_positions']:
        if pos is not None:
            kalman_px.append(pos)
        else:
            kalman_px.append((np.nan, np.nan))
    kalman_px = np.array(kalman_px)
    
    # Compute errors
    cotracker_errors = np.sqrt((true_px[:, 0] - cotracker_px[:, 0])**2 + 
                               (true_px[:, 1] - cotracker_px[:, 1])**2)
    kalman_errors = np.sqrt((true_px[:, 0] - kalman_px[:, 0])**2 + 
                           (true_px[:, 1] - kalman_px[:, 1])**2)
    
    # Compute metrics
    cotracker_rmse = np.sqrt(np.mean(cotracker_errors**2))
    kalman_valid_errors = kalman_errors[~np.isnan(kalman_errors)]
    kalman_rmse = np.sqrt(np.mean(kalman_valid_errors**2)) if len(kalman_valid_errors) > 0 else np.inf
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 2D Trajectory
    ax = axes[0, 0]
    ax.plot(true_px[:, 0], true_px[:, 1], 'k-', label='Ground Truth', linewidth=3, alpha=0.7)
    ax.plot(cotracker_px[:, 0], cotracker_px[:, 1], 'b--', label='CoTracker3', linewidth=2)
    ax.plot(kalman_px[:, 0], kalman_px[:, 1], 'r:', label='KalmanTrack', linewidth=2)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title('2D Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # X position over time
    ax = axes[0, 1]
    ax.plot(time, true_px[:, 0], 'k-', label='Ground Truth', linewidth=3, alpha=0.7)
    ax.plot(time, cotracker_px[:, 0], 'b--', label='CoTracker3', linewidth=2)
    ax.plot(time, kalman_px[:, 0], 'r:', label='KalmanTrack', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('X Position (pixels)', fontsize=12)
    ax.set_title('X Position vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Y position over time
    ax = axes[1, 0]
    ax.plot(time, true_px[:, 1], 'k-', label='Ground Truth', linewidth=3, alpha=0.7)
    ax.plot(time, cotracker_px[:, 1], 'b--', label='CoTracker3', linewidth=2)
    ax.plot(time, kalman_px[:, 1], 'r:', label='KalmanTrack', linewidth=2)
    # Mark bounces
    for bounce_time in simulation_result['bounce_times']:
        ax.axvline(x=bounce_time, color='orange', alpha=0.2, linestyle=':')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title('Y Position vs Time (orange lines = bounces)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Error comparison
    ax = axes[1, 1]
    ax.plot(time, cotracker_errors, 'b-', label='CoTracker3', linewidth=2)
    ax.plot(time, kalman_errors, 'r-', label='KalmanTrack', linewidth=2, alpha=0.7)
    ax.axhline(y=cotracker_rmse, color='b', linestyle='--', alpha=0.5,
              label=f'CoTracker3 RMSE: {cotracker_rmse:.1f}px')
    ax.axhline(y=kalman_rmse, color='r', linestyle='--', alpha=0.5,
              label=f'KalmanTrack RMSE: {kalman_rmse:.1f}px')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tracking Error (pixels)', fontsize=12)
    ax.set_title('Tracking Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(500, max(np.nanmax(cotracker_errors), np.nanmax(kalman_errors)) * 1.1))
    
    # Add comparison table
    comparison_text = f"""
    Comparison Metrics:
    
    CoTracker3:
      RMSE: {cotracker_rmse:.2f} px
      Mean Error: {np.mean(cotracker_errors):.2f} px
      Max Error: {np.max(cotracker_errors):.2f} px
      Success Rate: 100.0%
    
    KalmanTrack:
      RMSE: {kalman_rmse:.2f} px
      Mean Error: {np.mean(kalman_valid_errors):.2f} px
      Max Error: {np.nanmax(kalman_errors):.2f} px
      Success Rate: {kalman_result['metrics']['success_rate']:.1f}%
    
    Winner: {'CoTracker3' if cotracker_rmse < kalman_rmse else 'KalmanTrack'}
    Improvement: {abs(cotracker_rmse - kalman_rmse):.2f} px
    """
    
    fig.text(0.02, 0.5, comparison_text, fontsize=9, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0.15, 0, 1, 1])
    
    save_path = f'outputs/comparison_{scenario_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TRACKING COMPARISON SUMMARY")
    print("="*60)
    print(f"\nCoTracker3:")
    print(f"  RMSE: {cotracker_rmse:.2f} pixels")
    print(f"  Mean Error: {np.mean(cotracker_errors):.2f} pixels")
    print(f"  Max Error: {np.max(cotracker_errors):.2f} pixels")
    print(f"\nKalmanTrack (DINO + Kalman):")
    print(f"  RMSE: {kalman_rmse:.2f} pixels")
    print(f"  Mean Error: {np.mean(kalman_valid_errors):.2f} pixels")
    print(f"  Max Error: {np.nanmax(kalman_errors):.2f} pixels")
    print(f"  Success Rate: {kalman_result['metrics']['success_rate']:.1f}%")
    print(f"\nWinner: {'CoTracker3' if cotracker_rmse < kalman_rmse else 'KalmanTrack'}")
    print(f"Improvement: {abs(cotracker_rmse - kalman_rmse):.2f} pixels")
    print("="*60)


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("=== CoTracker3 vs KalmanTrack on Synthetic Data ===\n")
    
    # Load previous KalmanTrack results
    from test_synthetic_tracking import track_synthetic_video
    
    # Create scenario
    scenarios = create_test_scenarios()
    scenario_name = 'nominal'
    config = scenarios[scenario_name]
    
    print(f"Scenario: {scenario_name}")
    
    # Generate synthetic data
    print("\n1. Generating synthetic trajectory...")
    simulator = BouncingBallSimulator(config)
    result = simulator.simulate()
    
    # Generate video
    print("\n2. Generating synthetic video...")
    video_gen = SyntheticVideoGenerator()
    video_path = f'outputs/synthetic_{scenario_name}.mp4'
    
    if not os.path.exists(video_path):
        video_gen.generate_video(result, video_path, fps=30)
    else:
        print(f"Using existing video: {video_path}")
    
    # Get initial ball position
    initial_x, initial_y = result['true_states'][0, :2]
    initial_px = video_gen.meters_to_pixels(initial_x, initial_y)
    initial_points = np.array([initial_px])
    
    print(f"\nInitial ball position: {initial_px}")
    
    # Track with CoTracker3
    print("\n3. Tracking with CoTracker3...")
    cotracker_result = track_with_cotracker3(video_path, initial_points)
    
    # Track with KalmanTrack
    print("\n4. Tracking with KalmanTrack...")
    kalman_result = track_synthetic_video(video_path, result, scenario_name)
    
    # Create tracking videos
    print("\n5. Creating tracking videos...")
    create_tracking_videos(video_path, result, cotracker_result, kalman_result, scenario_name)
    
    # Compare results
    print("\n6. Comparing results...")
    compare_trackers(result, cotracker_result, kalman_result, scenario_name)
    
    print("\n✓ Done! Check outputs/ directory for:")
    print("  - comparison_nominal.mp4 (side-by-side video)")
    print("  - cotracker3_nominal.mp4 (CoTracker3 only)")
    print("  - kalmantrack_nominal.mp4 (KalmanTrack only)")
    print("  - comparison_nominal.png (plots)")
