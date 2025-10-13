#!/usr/bin/env python3
"""
Test KalmanTrack on Synthetic Bouncing Ball Data
Generates synthetic data and tracks it with the existing tracking system
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from synthetic_simulator import BouncingBallSimulator, SimulationConfig, create_test_scenarios
from kalman_track import KalmanTrack
import os
from typing import Dict, Tuple


class SyntheticVideoGenerator:
    """
    Converts synthetic trajectory data into video frames with a rendered ball
    """
    
    def __init__(self, width: int = 800, height: int = 600, 
                 pixels_per_meter: float = 50):
        """
        Initialize video generator
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            pixels_per_meter: Scale factor for converting meters to pixels
        """
        self.width = width
        self.height = height
        self.pixels_per_meter = pixels_per_meter
        self.ball_radius = 15  # pixels
        
    def meters_to_pixels(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Convert position from meters to pixel coordinates
        
        Args:
            x_m: X position in meters
            y_m: Y position in meters
            
        Returns:
            (x_px, y_px) in pixel coordinates
        """
        # Origin at bottom-left, convert to top-left for image coordinates
        x_px = int(x_m * self.pixels_per_meter + 50)  # 50px margin
        y_px = int(self.height - (y_m * self.pixels_per_meter + 50))  # Flip Y
        return x_px, y_px
    
    def render_frame(self, x: float, y: float, frame_number: int,
                    show_ground_truth: bool = True) -> np.ndarray:
        """
        Render a single frame with the ball at position (x, y)
        
        Args:
            x: X position in meters
            y: Y position in meters
            frame_number: Current frame number
            show_ground_truth: Whether to show ground truth marker
            
        Returns:
            RGB frame as numpy array
        """
        # Create blank frame (white background)
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw floor line
        floor_y = self.meters_to_pixels(0, 0)[1]
        cv2.line(frame, (0, floor_y), (self.width, floor_y), (100, 100, 100), 2)
        
        # Draw walls
        wall_left = self.meters_to_pixels(0, 0)[0]
        wall_right = self.meters_to_pixels(10, 0)[0]
        cv2.line(frame, (wall_left, 0), (wall_left, self.height), (100, 100, 100), 2)
        cv2.line(frame, (wall_right, 0), (wall_right, self.height), (100, 100, 100), 2)
        
        # Convert position to pixels
        x_px, y_px = self.meters_to_pixels(x, y)
        
        # Draw ball (red circle with texture for corner detection)
        cv2.circle(frame, (x_px, y_px), self.ball_radius, (0, 0, 255), -1)
        cv2.circle(frame, (x_px, y_px), self.ball_radius, (0, 0, 128), 2)
        
        # Add texture pattern (cross pattern for corners)
        cv2.line(frame, (x_px - self.ball_radius//2, y_px), 
                (x_px + self.ball_radius//2, y_px), (255, 255, 255), 2)
        cv2.line(frame, (x_px, y_px - self.ball_radius//2),
                (x_px, y_px + self.ball_radius//2), (255, 255, 255), 2)
        
        # Add small circles for more features
        cv2.circle(frame, (x_px - 5, y_px - 5), 2, (255, 255, 255), -1)
        cv2.circle(frame, (x_px + 5, y_px - 5), 2, (255, 255, 255), -1)
        cv2.circle(frame, (x_px - 5, y_px + 5), 2, (255, 255, 255), -1)
        cv2.circle(frame, (x_px + 5, y_px + 5), 2, (255, 255, 255), -1)
        
        # Add shadow for depth perception
        shadow_y = floor_y
        shadow_alpha = max(0.1, 1.0 - (floor_y - y_px) / 200.0)
        shadow_radius = int(self.ball_radius * shadow_alpha)
        overlay = frame.copy()
        cv2.ellipse(overlay, (x_px, shadow_y), (shadow_radius, shadow_radius // 2),
                   0, 0, 360, (150, 150, 150), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add position info
        cv2.putText(frame, f"Pos: ({x:.2f}m, {y:.2f}m)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def generate_video(self, simulation_result: Dict, 
                      output_path: str = 'outputs/synthetic_video.mp4',
                      fps: int = 30) -> str:
        """
        Generate video from simulation trajectory
        
        Args:
            simulation_result: Output from BouncingBallSimulator.simulate()
            output_path: Path to save video
            fps: Frames per second
            
        Returns:
            Path to generated video
        """
        true_states = simulation_result['true_states']
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        print(f"Generating video with {len(true_states)} frames...")
        
        for i, state in enumerate(true_states):
            x, y = state[0], state[1]
            frame = self.render_frame(x, y, i)
            out.write(frame)
            
            if (i + 1) % 50 == 0:
                print(f"  Rendered {i + 1}/{len(true_states)} frames")
        
        out.release()
        print(f"Video saved to: {output_path}")
        return output_path


def track_synthetic_video(video_path: str, simulation_result: Dict,
                         scenario_name: str = 'test') -> Dict:
    """
    Track the synthetic video with KalmanTrack and compare with ground truth
    
    Args:
        video_path: Path to synthetic video
        simulation_result: Ground truth data
        scenario_name: Name of scenario for saving results
        
    Returns:
        Dictionary with tracking results and metrics
    """
    print(f"\n=== Tracking Synthetic Video: {scenario_name} ===")
    
    # Initialize tracker
    tracker = KalmanTrack(
        dino_model='dino_vits16',
        n_keypoints=10,
        process_noise=0.005,
        measurement_noise=0.3,
        match_threshold=0.2,
        device='auto',
        use_multiscale=False,
        adaptive_threshold=False
    )
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    
    if not ret:
        print("Error: Could not read video")
        return None
    
    # Find ball position in first frame (look for red pixels)
    # Convert to HSV and find red ball
    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    # Red color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Get largest contour (the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Expand ROI slightly
        margin = 20
        roi = (max(0, x - margin), max(0, y - margin),
               min(first_frame.shape[1], x + w_box + margin),
               min(first_frame.shape[0], y + h_box + margin))
    else:
        # Fallback to center
        h, w = first_frame.shape[:2]
        roi_size = 100
        roi = (w//2 - roi_size//2, h//2 - roi_size//2,
               w//2 + roi_size//2, h//2 + roi_size//2)
    
    print(f"Setting ROI: {roi}")
    tracker.set_target(first_frame, roi)
    
    # Track through video
    tracked_positions = []
    frame_count = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = tracker.track(frame)
        
        if result.success and result.position:
            tracked_positions.append(result.position)
        else:
            tracked_positions.append(None)
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Tracked {frame_count} frames")
    
    cap.release()
    
    print(f"Tracking complete: {frame_count} frames")
    
    # Compute metrics
    metrics = compute_tracking_metrics(simulation_result, tracked_positions)
    
    # Visualize results
    visualize_tracking_results(simulation_result, tracked_positions, 
                              metrics, scenario_name)
    
    return {
        'tracked_positions': tracked_positions,
        'metrics': metrics,
        'scenario': scenario_name
    }


def compute_tracking_metrics(simulation_result: Dict, 
                            tracked_positions: list) -> Dict:
    """
    Compute tracking error metrics compared to ground truth
    
    Args:
        simulation_result: Ground truth from simulator
        tracked_positions: List of tracked positions (pixel coordinates)
        
    Returns:
        Dictionary of metrics
    """
    true_states = simulation_result['true_states']
    
    # Convert ground truth to pixel coordinates
    video_gen = SyntheticVideoGenerator()
    true_positions_px = []
    for state in true_states:
        x_px, y_px = video_gen.meters_to_pixels(state[0], state[1])
        true_positions_px.append((x_px, y_px))
    
    # Compute errors
    errors = []
    valid_frames = 0
    
    for i, (true_pos, tracked_pos) in enumerate(zip(true_positions_px, tracked_positions)):
        if tracked_pos is not None:
            error = np.sqrt((true_pos[0] - tracked_pos[0])**2 + 
                          (true_pos[1] - tracked_pos[1])**2)
            errors.append(error)
            valid_frames += 1
        else:
            errors.append(np.nan)
    
    errors = np.array(errors)
    valid_errors = errors[~np.isnan(errors)]
    
    metrics = {
        'rmse': np.sqrt(np.mean(valid_errors**2)) if len(valid_errors) > 0 else np.inf,
        'mean_error': np.mean(valid_errors) if len(valid_errors) > 0 else np.inf,
        'max_error': np.max(valid_errors) if len(valid_errors) > 0 else np.inf,
        'std_error': np.std(valid_errors) if len(valid_errors) > 0 else np.inf,
        'success_rate': valid_frames / len(tracked_positions) * 100,
        'total_frames': len(tracked_positions),
        'valid_frames': valid_frames
    }
    
    return metrics


def visualize_tracking_results(simulation_result: Dict, 
                               tracked_positions: list,
                               metrics: Dict,
                               scenario_name: str):
    """
    Create comprehensive visualization of tracking results
    
    Args:
        simulation_result: Ground truth data
        tracked_positions: Tracked positions
        metrics: Computed metrics
        scenario_name: Scenario name for saving
    """
    true_states = simulation_result['true_states']
    time = simulation_result['time']
    
    # Convert to pixel coordinates
    video_gen = SyntheticVideoGenerator()
    true_px = np.array([video_gen.meters_to_pixels(s[0], s[1]) for s in true_states])
    
    tracked_px = []
    for pos in tracked_positions:
        if pos is not None:
            tracked_px.append(pos)
        else:
            tracked_px.append((np.nan, np.nan))
    tracked_px = np.array(tracked_px)
    
    # Compute errors
    errors = np.sqrt((true_px[:, 0] - tracked_px[:, 0])**2 + 
                    (true_px[:, 1] - tracked_px[:, 1])**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2D trajectory comparison
    ax = axes[0, 0]
    ax.plot(true_px[:, 0], true_px[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(tracked_px[:, 0], tracked_px[:, 1], 'r--', label='Tracked', linewidth=2, alpha=0.7)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('2D Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Match image coordinates
    
    # X position over time
    ax = axes[0, 1]
    ax.plot(time, true_px[:, 0], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time, tracked_px[:, 0], 'r--', label='Tracked', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position (pixels)')
    ax.set_title('X Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y position over time
    ax = axes[1, 0]
    ax.plot(time, true_px[:, 1], 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time, tracked_px[:, 1], 'r--', label='Tracked', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Y Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tracking error over time
    ax = axes[1, 1]
    ax.plot(time, errors, 'g-', linewidth=2)
    ax.axhline(y=metrics['mean_error'], color='r', linestyle='--', 
              label=f"Mean: {metrics['mean_error']:.1f}px")
    ax.axhline(y=metrics['rmse'], color='orange', linestyle='--',
              label=f"RMSE: {metrics['rmse']:.1f}px")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (pixels)')
    ax.set_title('Tracking Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f"""Metrics:
RMSE: {metrics['rmse']:.2f} px
Mean Error: {metrics['mean_error']:.2f} px
Max Error: {metrics['max_error']:.2f} px
Success Rate: {metrics['success_rate']:.1f}%"""
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = f'outputs/tracking_results_{scenario_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results visualization saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    
    print("=== Synthetic Tracking Test ===\n")
    
    # Create scenario
    scenarios = create_test_scenarios()
    scenario_name = 'nominal'
    config = scenarios[scenario_name]
    
    print(f"Testing scenario: {scenario_name}")
    print(f"Configuration:")
    print(f"  Gravity: {config.gravity} m/s²")
    print(f"  Elasticity: {config.elasticity}")
    print(f"  Duration: {config.duration}s")
    print(f"  Process noise: {config.process_noise_std}")
    print(f"  Measurement noise: {config.measurement_noise_std}")
    
    # Generate synthetic data
    print("\n1. Generating synthetic trajectory...")
    simulator = BouncingBallSimulator(config)
    result = simulator.simulate()
    
    print(f"   Generated {len(result['true_states'])} states")
    print(f"   Number of bounces: {len(result['bounce_times'])}")
    
    # Generate video
    print("\n2. Generating synthetic video...")
    video_gen = SyntheticVideoGenerator()
    video_path = video_gen.generate_video(result, 
                                         f'outputs/synthetic_{scenario_name}.mp4',
                                         fps=30)
    
    # Track video
    print("\n3. Tracking with KalmanTrack...")
    tracking_results = track_synthetic_video(video_path, result, scenario_name)
    
    # Print final metrics
    print("\n=== Final Results ===")
    metrics = tracking_results['metrics']
    print(f"RMSE: {metrics['rmse']:.2f} pixels")
    print(f"Mean Error: {metrics['mean_error']:.2f} pixels")
    print(f"Max Error: {metrics['max_error']:.2f} pixels")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Valid Frames: {metrics['valid_frames']}/{metrics['total_frames']}")
    
    print("\nDone! Check outputs/ directory for results.")
