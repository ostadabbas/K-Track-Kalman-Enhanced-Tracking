#!/usr/bin/env python3
"""
Improved KalmanTrack example with better parameters and ROI selection
"""

import cv2
import numpy as np
from kalman_track import KalmanTrack, InteractiveROISelector

def improved_tracking_example():
    """Improved example with interactive ROI selection and better parameters"""
    
    # Initialize tracker with stable corner-based approach
    print("Initializing KalmanTrack with stable parameters...")
    tracker = KalmanTrack(
        dino_model='dino_vits16',      # Fastest model
        n_keypoints=10,                # More keypoints for better matching
        process_noise=0.01,            # Lower process noise for stability
        measurement_noise=0.3,         # Lower measurement noise
        match_threshold=0.50,           # Fixed threshold (works better)
        device='auto',
        use_multiscale=True,          # Disable multi-scale (caused issues)
        adaptive_threshold=True       # Disable adaptive (too strict)
    )
    
    # Load video
    video_path = 'videos/plane.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Setup output files
    import os
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/tracking_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writer setup
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(output_dir, "tracked_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Tracking data storage
    tracking_data = {
        'video_info': {
            'path': video_path,
            'fps': fps,
            'width': width,
            'height': height
        },
        'tracker_config': {
            'dino_model': 'dino_vits16',
            'n_keypoints': 10,
            'process_noise': 0.005,
            'measurement_noise': 0.3,
            'match_threshold': 0.2
        },
        'frames': []
    }
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    print(f"Video loaded: {frame.shape}")
    
    # Interactive ROI selection
    print("Please select a region with a clear, trackable object...")
    roi_selector = InteractiveROISelector()
    roi = roi_selector.select_roi(frame)
    
    if roi is None:
        print("No ROI selected. Using default center ROI.")
        # Fallback to center ROI but smaller
        height, width = frame.shape[:2]
        roi_size = 80
        center_x, center_y = width // 2, height // 2
        roi = (
            center_x - roi_size // 2,
            center_y - roi_size // 2,
            center_x + roi_size // 2,
            center_y + roi_size // 2
        )
    
    print(f"Using ROI: {roi}")
    
    # Show ROI on first frame
    roi_frame = frame.copy()
    x_min, y_min, x_max, y_max = roi
    cv2.rectangle(roi_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(roi_frame, "Selected ROI", (x_min, y_min - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Selected ROI', roi_frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyWindow('Selected ROI')
    
    # Set target
    tracker.set_target(frame, roi)
    
    # Process video
    frame_count = 0
    successful_tracks = 0
    
    print("Starting tracking... Press 'q' to quit, 'r' to reset ROI")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Track object
        result = tracker.track(frame)
        
        if result.success:
            successful_tracks += 1
        
        # Save tracking data for this frame
        frame_data = {
            'frame_number': frame_count,
            'success': result.success,
            'center_position': result.position,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'num_matches': result.num_matches,
            'processing_time': result.processing_time,
            'points': []
        }
        
        # Save individual point data
        if result.positions and result.point_ids:
            for i, (pos, point_id) in enumerate(zip(result.positions, result.point_ids)):
                frame_data['points'].append({
                    'id': int(point_id),
                    'position': [int(pos[0]), int(pos[1])],
                    'index': i
                })
        
        tracking_data['frames'].append(frame_data)
        
        # Visualize
        vis_frame = tracker.visualize_tracking(frame, result, show_prediction=True, show_roi=True)
        
        # Add success rate and point count info
        success_rate = successful_tracks / frame_count * 100
        point_count = len(result.positions) if result.positions else 0
        cv2.putText(vis_frame, f"Success: {success_rate:.1f}%, Points: {point_count}", 
                   (10, vis_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(vis_frame)
        
        # Display
        cv2.imshow('Improved KalmanTrack', vis_frame)
        
        # Print detailed status every 30 frames
        if frame_count % 30 == 0:
            point_count = len(result.positions) if result.positions else 0
            print(f"Frame {frame_count}: Success={result.success}, "
                  f"Points={point_count}, "
                  f"Confidence={result.confidence:.2f}, "
                  f"Success Rate={success_rate:.1f}%, "
                  f"Time={result.processing_time*1000:.1f}ms")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting tracker with new ROI...")
            # Reset and select new ROI
            tracker.reset()
            new_roi = roi_selector.select_roi(frame)
            if new_roi is not None:
                roi = new_roi
                tracker.set_target(frame, roi)
                successful_tracks = 0  # Reset success counter
                print(f"New ROI set: {roi}")
    
    # Cleanup
    cap.release()
    out.release()  # Close video writer
    cv2.destroyAllWindows()
    
    # Add final statistics to tracking data
    stats = tracker.get_performance_stats()
    tracking_data['final_stats'] = {
        'total_frames': frame_count,
        'successful_tracks': successful_tracks,
        'success_rate': successful_tracks/frame_count*100 if frame_count > 0 else 0,
        **stats
    }
    
    # Save tracking data to JSON
    json_path = os.path.join(output_dir, "tracking_data.json")
    with open(json_path, 'w') as f:
        json.dump(tracking_data, f, indent=2, default=str)
    
    # Save summary report
    report_path = os.path.join(output_dir, "tracking_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"KalmanTrack Tracking Report\n")
        f.write(f"===========================\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        for key, value in tracking_data['tracker_config'].items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total frames: {frame_count}\n")
        f.write(f"  Successful tracks: {successful_tracks}\n")
        f.write(f"  Success rate: {successful_tracks/frame_count*100:.1f}%\n")
        for key, value in stats.items():
            if key not in ['total_frames', 'successful_tracks']:
                f.write(f"  {key}: {value}\n")
    
    # Print final stats
    print("\nFinal Statistics:")
    print(f"  Total frames: {frame_count}")
    print(f"  Successful tracks: {successful_tracks}")
    print(f"  Success rate: {successful_tracks/frame_count*100:.1f}%")
    for key, value in stats.items():
        if key not in ['total_frames', 'successful_tracks']:
            print(f"  {key}: {value}")
    
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - Video: {output_video_path}")
    print(f"  - Data: {json_path}")
    print(f"  - Report: {report_path}")


def test_different_parameters():
    """Test different parameter combinations to find optimal settings"""
    
    video_path = 'videos/helicopter.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not load test frame")
        return
    
    # Test parameters
    test_configs = [
        {'match_threshold': 0.3, 'n_keypoints': 30, 'name': 'Lenient + Few KP'},
        {'match_threshold': 0.4, 'n_keypoints': 50, 'name': 'Balanced'},
        {'match_threshold': 0.5, 'n_keypoints': 70, 'name': 'Strict + Many KP'},
        {'match_threshold': 0.6, 'n_keypoints': 100, 'name': 'Very Strict'},
    ]
    
    # Simple ROI in center
    height, width = frame.shape[:2]
    roi = (width//2 - 50, height//2 - 50, width//2 + 50, height//2 + 50)
    
    print("Testing different parameter configurations...")
    print("ROI:", roi)
    
    for config in test_configs:
        print(f"\nTesting {config['name']}:")
        print(f"  match_threshold: {config['match_threshold']}")
        print(f"  n_keypoints: {config['n_keypoints']}")
        
        tracker = KalmanTrack(
            dino_model='dino_vits16',
            n_keypoints=config['n_keypoints'],
            match_threshold=config['match_threshold'],
            device='auto'
        )
        
        tracker.set_target(frame, roi)
        
        # Test on a few synthetic frames
        test_frames = 10
        successes = 0
        
        for i in range(test_frames):
            # Add small random noise to simulate motion
            noisy_frame = frame.copy()
            noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
            noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            result = tracker.track(noisy_frame)
            if result.success:
                successes += 1
        
        success_rate = successes / test_frames * 100
        print(f"  Success rate: {success_rate:.1f}% ({successes}/{test_frames})")


if __name__ == "__main__":
    print("KalmanTrack Improved Example")
    print("1. Run improved tracking")
    print("2. Test parameter configurations")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        improved_tracking_example()
    elif choice == "2":
        test_different_parameters()
    else:
        print("Running improved tracking by default...")
        improved_tracking_example()
