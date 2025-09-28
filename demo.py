#!/usr/bin/env python3
"""
KalmanTrack Demo Script
Demonstrates DINO + Kalman Filter tracking on video files
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

from kalman_track import KalmanTrack, InteractiveROISelector


def main():
    parser = argparse.ArgumentParser(description='KalmanTrack Demo - DINO + Kalman Filter Tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    parser.add_argument('--dino_model', type=str, default='dino_vits16', 
                       choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                       help='DINO model variant')
    parser.add_argument('--n_keypoints', type=int, default=50, help='Number of keypoints to extract')
    parser.add_argument('--process_noise', type=float, default=0.03, help='Kalman process noise')
    parser.add_argument('--measurement_noise', type=float, default=0.5, help='Kalman measurement noise')
    parser.add_argument('--match_threshold', type=float, default=0.7, help='Feature matching threshold')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--no_display', action='store_true', help='Disable real-time display')
    parser.add_argument('--save_stats', action='store_true', help='Save performance statistics')
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    print(f"Loading video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize tracker
    print("Initializing KalmanTrack...")
    tracker = KalmanTrack(
        dino_model=args.dino_model,
        n_keypoints=args.n_keypoints,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        match_threshold=args.match_threshold,
        device=args.device
    )
    
    # Read first frame for ROI selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Interactive ROI selection
    print("Select target object by dragging mouse...")
    roi_selector = InteractiveROISelector()
    roi = roi_selector.select_roi(first_frame)
    
    if roi is None:
        print("No ROI selected. Exiting.")
        return
    
    print(f"ROI selected: {roi}")
    
    # Set target
    tracker.set_target(first_frame, roi)
    
    # Setup output video writer if requested
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {args.output}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Tracking loop
    frame_count = 0
    start_time = time.time()
    
    print("Starting tracking... Press 'q' to quit, 'r' to reset, 'p' to pause")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
        
        # Track object
        if not paused:
            result = tracker.track(frame)
        
        # Visualize results
        if not args.no_display or args.output:
            vis_frame = tracker.visualize_tracking(frame, result, show_prediction=True, show_roi=True)
            
            # Add progress bar
            progress = frame_count / total_frames
            bar_width = width - 40
            bar_height = 10
            bar_x, bar_y = 20, height - 30
            
            cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            progress_text = f"{frame_count}/{total_frames} ({progress*100:.1f}%)"
            cv2.putText(vis_frame, progress_text, (bar_x, bar_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            if not args.no_display:
                cv2.imshow('KalmanTrack Demo', vis_frame)
            
            # Write to output video
            if out_writer:
                out_writer.write(vis_frame)
        
        # Handle keyboard input
        if not args.no_display:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('r'):
                print("Resetting tracker...")
                tracker.reset()
                tracker.set_target(frame, roi)
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed {frame_count} frames, {fps_actual:.1f} FPS")
    
    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    total_time = time.time() - start_time
    stats = tracker.get_performance_stats()
    
    print("\n" + "="*50)
    print("TRACKING STATISTICS")
    print("="*50)
    print(f"Total frames processed: {stats.get('total_frames', 0)}")
    print(f"Successful tracks: {stats.get('successful_tracks', 0)}")
    print(f"Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"Average processing time: {stats.get('avg_processing_time', 0)*1000:.1f} ms")
    print(f"Average FPS: {stats.get('fps', 0):.1f}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Real-time factor: {total_frames/fps/total_time:.2f}x")
    
    # Save statistics if requested
    if args.save_stats:
        stats_file = Path(args.video).stem + "_kalmantrack_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("KalmanTrack Performance Statistics\n")
            f.write("="*40 + "\n")
            f.write(f"Video: {args.video}\n")
            f.write(f"DINO Model: {args.dino_model}\n")
            f.write(f"Keypoints: {args.n_keypoints}\n")
            f.write(f"Process Noise: {args.process_noise}\n")
            f.write(f"Measurement Noise: {args.measurement_noise}\n")
            f.write(f"Match Threshold: {args.match_threshold}\n")
            f.write(f"Device: {args.device}\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nTotal processing time: {total_time:.2f} seconds\n")
            f.write(f"Real-time factor: {total_frames/fps/total_time:.2f}x\n")
        
        print(f"Statistics saved to: {stats_file}")
    
    # Plot tracking history
    try:
        plot_file = Path(args.video).stem + "_kalmantrack_plot.png"
        tracker.plot_tracking_history(save_path=plot_file)
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
