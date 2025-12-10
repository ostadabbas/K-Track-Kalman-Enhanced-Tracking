#!/usr/bin/env python3
"""
Basic usage example for K-track hybrid point tracker.

This example demonstrates how to use K-track to accelerate CoTracker3
by running it only on keyframes and using Kalman filtering for intermediate frames.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalman_hybrid import KalmanTrackHybrid


def load_video(video_path: str, max_frames: int = None):
    """
    Load video from file and convert to tensor format.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)
        
    Returns:
        video_tensor: [1, T, 3, H, W] tensor
        fps: Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and len(frames) >= max_frames:
            break
            
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames loaded from video")
    
    # Stack into [T, 3, H, W] and add batch dimension
    video_tensor = torch.stack(frames).unsqueeze(0)  # [1, T, 3, H, W]
    
    return video_tensor, fps


def example_basic_tracking():
    """Basic example: Track points in a video using K-track."""
    
    print("=" * 60)
    print("K-track Basic Usage Example")
    print("=" * 60)
    
    # Configuration
    video_path = "videos/helicopter.mp4"  # Change to your video
    N = 5  # Run tracker every 5 frames
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Video: {video_path}")
    print(f"  Keyframe frequency (N): {N}")
    print(f"  Device: {device}")
    
    # Load video
    print(f"\nLoading video...")
    try:
        video_tensor, fps = load_video(video_path, max_frames=100)
        print(f"  Loaded {video_tensor.shape[1]} frames at {fps:.1f} FPS")
        print(f"  Resolution: {video_tensor.shape[3]}x{video_tensor.shape[4]}")
    except Exception as e:
        print(f"Error loading video: {e}")
        print("\nNote: Place a video file in the 'videos/' directory or update video_path")
        return
    
    # Move to device
    video_tensor = video_tensor.to(device)
    
    # Define initial points to track (center region)
    H, W = video_tensor.shape[3], video_tensor.shape[4]
    initial_points = np.array([
        [W // 2, H // 2],           # Center
        [W // 2 - 50, H // 2],      # Left of center
        [W // 2 + 50, H // 2],      # Right of center
        [W // 2, H // 2 - 50],      # Above center
        [W // 2, H // 2 + 50],      # Below center
    ])
    
    print(f"\nInitializing tracker with {len(initial_points)} points...")
    
    # Create hybrid tracker
    tracker = KalmanTrackHybrid(
        N=N,
        warmup=3,
        device=device,
        tracker_type='cotracker3'
    )
    
    # Create query tensor [1, N_points, 3] where each point is [t, x, y]
    queries = torch.tensor([[[0, p[0], p[1]] for p in initial_points]], 
                          device=device).float()
    
    # Initialize tracker
    tracker.initialize(video_tensor, initial_points)
    
    print("\nTracking through video...")
    print("-" * 60)
    
    # Track through video
    results = []
    for frame_idx in range(1, video_tensor.shape[1]):
        result = tracker.track_frame(video_tensor, queries, frame_idx)
        results.append(result)
        
        if frame_idx % 10 == 0 or result.used_cotracker:
            status = "TRACKER" if result.used_cotracker else "KALMAN"
            print(f"Frame {frame_idx:3d}: {status:6s} | "
                  f"Time: {result.processing_time*1000:.1f}ms | "
                  f"Points: {result.positions.shape[0]}")
    
    # Print statistics
    stats = tracker.get_statistics()
    print("\n" + "=" * 60)
    print("Tracking Statistics:")
    print("=" * 60)
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Tracker calls: {stats['cotracker_calls']}")
    print(f"  Kalman predictions: {stats['kalman_predictions']}")
    print(f"  Speedup factor: {stats['speedup_factor']:.2f}×")
    print(f"  Tracker frequency: {stats['cotracker_frequency']}")
    
    avg_time = np.mean([r.processing_time for r in results])
    print(f"\n  Average processing time: {avg_time*1000:.1f}ms")
    print(f"  Estimated FPS: {1.0/avg_time:.1f}")
    
    print("\nExample completed successfully!")
    print("\nNext steps:")
    print("  - Try different N values (3, 5, 10) to see speed/accuracy tradeoff")
    print("  - Experiment with different trackers (tapir, spatracker, trackon)")
    print("  - See examples/ directory for more advanced usage")


if __name__ == "__main__":
    example_basic_tracking()

