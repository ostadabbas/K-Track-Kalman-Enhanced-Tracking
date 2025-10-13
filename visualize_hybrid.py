#!/usr/bin/env python3
"""
Visualize Hybrid Tracking on Video
Shows when CoTracker3 is used vs Kalman prediction
"""

import numpy as np
import cv2
import os
from synthetic_simulator import BouncingBallSimulator, create_test_scenarios
from test_synthetic_tracking import SyntheticVideoGenerator
from test_hybrid_simple import SimpleKalmanFilter, simulate_hybrid_from_cotracker


def create_hybrid_visualization_video(N=5):
    """
    Create video showing hybrid tracking with visual indicators
    
    Args:
        N: CoTracker3 frequency
    """
    print(f"Creating hybrid tracking visualization (N={N})...")
    
    # Load simulation
    scenarios = create_test_scenarios()
    config = scenarios['nominal']
    simulator = BouncingBallSimulator(config)
    result = simulator.simulate()
    
    # Get ground truth
    video_gen = SyntheticVideoGenerator()
    true_states = result['true_states']
    true_px = np.array([video_gen.meters_to_pixels(s[0], s[1]) for s in true_states])
    
    # Simulate CoTracker3 (ground truth + noise)
    np.random.seed(42)
    cotracker_tracks = true_px + np.random.randn(*true_px.shape) * 5
    
    # Get hybrid predictions
    hybrid_tracks = simulate_hybrid_from_cotracker(cotracker_tracks, N)
    
    # Load original video
    video_path = 'outputs/synthetic_nominal.mp4'
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    output_path = f'outputs/hybrid_tracking_N{N}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {len(true_px)} frames...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine if this is a CoTracker3 frame
        is_cotracker_frame = (frame_idx % N == 0)
        
        # Draw ground truth (green circle)
        gt_pos = (int(true_px[frame_idx, 0]), int(true_px[frame_idx, 1]))
        cv2.circle(frame, gt_pos, 10, (0, 255, 0), 2)
        cv2.putText(frame, "GT", (gt_pos[0] + 15, gt_pos[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw hybrid prediction
        hybrid_pos = (int(hybrid_tracks[frame_idx, 0]), int(hybrid_tracks[frame_idx, 1]))
        
        if is_cotracker_frame:
            # CoTracker3 frame - blue circle
            cv2.circle(frame, hybrid_pos, 8, (255, 0, 0), -1)
            cv2.putText(frame, "CoTracker3", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            # Kalman prediction - red circle
            cv2.circle(frame, hybrid_pos, 8, (0, 0, 255), -1)
            cv2.putText(frame, "Kalman Predict", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw line from GT to prediction
        cv2.line(frame, gt_pos, hybrid_pos, (255, 255, 0), 1)
        
        # Compute error
        error = np.sqrt((gt_pos[0] - hybrid_pos[0])**2 + (gt_pos[1] - hybrid_pos[1])**2)
        
        # Add info text
        cv2.putText(frame, f"Frame: {frame_idx}/{len(true_px)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Error: {error:.1f} px", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"N={N} (CoTracker3 every {N} frames)", (10, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend
        cv2.putText(frame, "Green=Ground Truth", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Blue=CoTracker3", (200, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "Red=Kalman", (400, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Write frame
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{len(true_px)} frames")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Video saved: {output_path}")
    
    # Compute final statistics
    errors = np.sqrt(np.sum((true_px - hybrid_tracks)**2, axis=1))
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    cotracker_frames = len(true_px) // N + 1
    kalman_frames = len(true_px) - cotracker_frames
    
    print(f"\nStatistics:")
    print(f"  RMSE: {rmse:.2f} pixels")
    print(f"  Mean Error: {mean_error:.2f} pixels")
    print(f"  Max Error: {max_error:.2f} pixels")
    print(f"  CoTracker3 frames: {cotracker_frames}/{len(true_px)} ({cotracker_frames/len(true_px)*100:.1f}%)")
    print(f"  Kalman frames: {kalman_frames}/{len(true_px)} ({kalman_frames/len(true_px)*100:.1f}%)")
    print(f"  Speedup: ~{N}×")


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    
    print("="*60)
    print("Hybrid Tracking Visualization")
    print("="*60)
    
    # Create visualizations for different N values
    for N in [3, 5, 10]:
        print(f"\n{'='*60}")
        create_hybrid_visualization_video(N)
    
    print("\n" + "="*60)
    print("✓ All visualizations complete!")
    print("="*60)
    print("\nGenerated videos:")
    print("  - outputs/hybrid_tracking_N3.mp4  (Best accuracy)")
    print("  - outputs/hybrid_tracking_N5.mp4  (Balanced)")
    print("  - outputs/hybrid_tracking_N10.mp4 (Best speedup)")
