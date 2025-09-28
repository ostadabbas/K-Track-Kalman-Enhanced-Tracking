"""
KalmanTrack: Fast point tracking using DINO features and Kalman filtering
A faster alternative to CoTracker3 for object tracking
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dino_extractor import DINOFeatureExtractor
from kalman_filter import KalmanTracker, MultiObjectKalmanTracker


@dataclass
class TrackingResult:
    """Result of tracking operation"""
    success: bool
    position: Optional[Tuple[int, int]]  # Center position (backward compatibility)
    positions: Optional[List[Tuple[int, int]]]  # All tracked points
    point_ids: Optional[List[int]]  # IDs of tracked points for consistency
    prediction: Optional[Tuple[int, int]]
    confidence: float
    num_matches: int
    processing_time: float


class KalmanTrack:
    """
    Main tracking class combining DINO features with Kalman filtering
    """
    
    def __init__(self, 
                 dino_model: str = 'dino_vits16',
                 n_keypoints: int = 50,
                 process_noise: float = 0.03,
                 measurement_noise: float = 0.5,
                 match_threshold: float = 0.3,
                 min_matches: int = 5,
                 device: str = 'auto',
                 use_multiscale: bool = True,
                 adaptive_threshold: bool = True):
        """
        Initialize KalmanTrack
        
        Args:
            dino_model: DINO model variant
            n_keypoints: Number of keypoints to extract
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            match_threshold: Feature matching threshold
            min_matches: Minimum matches required for tracking
            device: Device to run on
        """
        print("Initializing KalmanTrack...")
        
        # Initialize DINO feature extractor
        self.feature_extractor = DINOFeatureExtractor(
            model_name=dino_model,
            n_keypoints=n_keypoints,
            device=device
        )
        
        # Configure advanced features
        self.feature_extractor.use_multiscale = use_multiscale
        self.adaptive_threshold = adaptive_threshold
        
        # Initialize multi-object Kalman tracker for individual points
        self.multi_tracker = MultiObjectKalmanTracker(
            max_objects=100,  # Allow up to 100 tracked points
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        # Keep single tracker for backward compatibility (center tracking)
        self.kalman_tracker = KalmanTracker(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        # Tracking parameters
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.adaptive_threshold = adaptive_threshold
        
        # State variables
        self.is_initialized = False
        self.roi = None
        self.frame_count = 0
        self.tracking_history = []
        
        # Point tracking state
        self.reference_point_ids = {}  # Map reference keypoint index to tracker ID
        self.active_point_ids = set()  # Currently active point IDs
        self.point_colors = {}  # Map reference point index to unique color
        self.color_palette = self._generate_color_palette(100)  # Pre-generate colors
        
        # Performance tracking
        self.processing_times = []
        
        print("KalmanTrack initialized successfully!")
    
    def set_target(self, image: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Set target object for tracking
        
        Args:
            image: First frame containing the target
            roi: Region of interest (x_min, y_min, x_max, y_max)
        """
        print(f"Setting target with ROI: {roi}")
        
        self.roi = roi
        x_min, y_min, x_max, y_max = roi
        
        # Extract ROI
        roi_image = image[y_min:y_max, x_min:x_max]
        
        # Set reference features with ROI coordinates
        self.feature_extractor.set_reference(image, roi)
        
        # Point tracking state with colors and consistency
        self.reference_point_ids = {}
        self.active_point_ids = set()
        self.point_colors = {}  # Map reference point index to unique color
        self.color_palette = self._generate_color_palette(100)  # Pre-generate colors
        self.consistent_points = {}  # Map ref_idx -> last known position for consistency
        self.point_history = {}  # Track point positions over time
        
        # Initialize center Kalman filter for backward compatibility
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        self.kalman_tracker.initialize(center_x, center_y)
        
        self.is_initialized = True
        self.frame_count = 0
        
        print(f"Target set for dynamic point tracking, center at ({center_x}, {center_y})")
    
    def track(self, image: np.ndarray, use_prediction: bool = True) -> TrackingResult:
        """
        Track object in current frame
        
        Args:
            image: Current frame
            use_prediction: Whether to use Kalman prediction
            
        Returns:
            TrackingResult with tracking information
        """
        if not self.is_initialized:
            return TrackingResult(
                success=False,
                position=None,
                prediction=None,
                confidence=0.0,
                num_matches=0,
                processing_time=0.0
            )
        
        start_time = time.time()
        
        # Get Kalman prediction
        prediction = None
        if use_prediction:
            pred_x, pred_y = self.kalman_tracker.predict()
            prediction = (int(pred_x), int(pred_y))
        
        # Track multiple points using DINO features with consistency
        tracked_points = self.feature_extractor.track_points(image)
        
        # Update consistent point tracking
        if tracked_points is not None and hasattr(self.feature_extractor, 'last_point_ref_indices'):
            ref_indices = self.feature_extractor.last_point_ref_indices
            
            # Update consistent points dictionary
            for i, ref_idx in enumerate(ref_indices):
                if i < len(tracked_points):
                    self.consistent_points[ref_idx] = tracked_points[i]
                    
                    # Update point history for trajectory
                    if ref_idx not in self.point_history:
                        self.point_history[ref_idx] = []
                    self.point_history[ref_idx].append(tracked_points[i])
                    
                    # Keep only last 10 positions for each point
                    if len(self.point_history[ref_idx]) > 10:
                        self.point_history[ref_idx] = self.point_history[ref_idx][-10:]
        
        # Calculate center from ALL consistent points (not just current matches)
        detection_result = None
        if len(self.consistent_points) > 0:
            all_positions = list(self.consistent_points.values())
            center_x = sum(p[0] for p in all_positions) / len(all_positions)
            center_y = sum(p[1] for p in all_positions) / len(all_positions)
            detection_result = (int(center_x), int(center_y))
        
        # Debug: print detection status
        if self.frame_count % 30 == 0:  # Print every 30 frames
            if tracked_points is not None:
                print(f"Frame {self.frame_count}: Tracking {len(tracked_points)} points, center at {detection_result}")
            else:
                print(f"Frame {self.frame_count}: Detection failed")
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        if detection_result is not None:
            # Successful detection
            detected_x, detected_y = detection_result
            
            # Update Kalman filter with measurement
            corrected_x, corrected_y = self.kalman_tracker.update(detected_x, detected_y)
            
            # Calculate confidence based on prediction accuracy (if available)
            confidence = 1.0
            if prediction is not None:
                pred_error = np.sqrt((prediction[0] - detected_x)**2 + (prediction[1] - detected_y)**2)
                confidence = max(0.0, 1.0 - pred_error / 100.0)  # Normalize by 100 pixels
            
            # Use ALL consistent points (not just current matches) for stable visualization
            result_positions = list(self.consistent_points.values())
            result_point_ids = list(self.consistent_points.keys())
            
            result = TrackingResult(
                success=True,
                position=(int(corrected_x), int(corrected_y)),
                positions=result_positions,
                point_ids=result_point_ids,
                prediction=prediction,
                confidence=confidence,
                num_matches=len(tracked_points) if tracked_points else 0,
                processing_time=processing_time
            )
        else:
            # Detection failed, use prediction if available
            if prediction is not None and not self.kalman_tracker.is_lost():
                # Use consistent points even when detection fails
                result_positions = list(self.consistent_points.values()) if self.consistent_points else ([prediction] if prediction else [])
                result_point_ids = list(self.consistent_points.keys()) if self.consistent_points else ([0] if prediction else [])
                
                result = TrackingResult(
                    success=True,
                    position=prediction,
                    positions=result_positions if result_positions else ([prediction] if prediction else None),
                    point_ids=result_point_ids if result_point_ids else None,
                    prediction=prediction,
                    confidence=0.3,  # Low confidence for prediction-only
                    num_matches=0,
                    processing_time=processing_time
                )
            else:
                result = TrackingResult(
                    success=False,
                    position=None,
                    positions=None,
                    point_ids=None,
                    prediction=prediction,
                    confidence=0.0,
                    num_matches=0,
                    processing_time=processing_time
                )
        
        # Store tracking history
        self.tracking_history.append(result)
        self.frame_count += 1
        
        return result
    
    def _generate_color_palette(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate a palette of distinct colors for point visualization
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of BGR color tuples
        """
        import colorsys
        
        colors = []
        for i in range(num_colors):
            # Use HSV color space for better color distribution
            hue = i / num_colors
            saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 + (i % 2) * 0.1       # Vary brightness slightly
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR for OpenCV (and scale to 0-255)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
    def _get_point_color(self, point_index: int) -> Tuple[int, int, int]:
        """
        Get consistent color for a point index
        
        Args:
            point_index: Index of the point
            
        Returns:
            BGR color tuple
        """
        return self.color_palette[point_index % len(self.color_palette)]
    
    def reset(self):
        """Reset tracker state"""
        self.kalman_tracker.reset()
        self.is_initialized = False
        self.roi = None
        self.frame_count = 0
        self.tracking_history.clear()
        self.processing_times.clear()
        print("Tracker reset")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        processing_times = np.array(self.processing_times)
        successful_tracks = sum(1 for result in self.tracking_history if result.success)
        
        return {
            'total_frames': self.frame_count,
            'successful_tracks': successful_tracks,
            'success_rate': successful_tracks / max(1, self.frame_count),
            'avg_processing_time': np.mean(processing_times),
            'fps': 1.0 / np.mean(processing_times) if len(processing_times) > 0 else 0,
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times)
        }
    
    def visualize_tracking(self, image: np.ndarray, result: TrackingResult, 
                          show_prediction: bool = True, show_roi: bool = False) -> np.ndarray:
        """
        Visualize tracking results on image
        
        Args:
            image: Current frame
            result: Tracking result
            show_prediction: Whether to show prediction
            show_roi: Whether to show initial ROI
            
        Returns:
            Image with tracking visualization
        """
        vis_image = image.copy()
        
        # Draw initial ROI (reference only)
        if show_roi and self.roi is not None:
            x_min, y_min, x_max, y_max = self.roi
            
            # Draw original ROI as reference
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(vis_image, "Initial ROI", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw prediction
        if show_prediction and result.prediction is not None:
            pred_x, pred_y = result.prediction
            cv2.circle(vis_image, (pred_x, pred_y), 8, (0, 255, 255), 2)  # Yellow circle
            cv2.putText(vis_image, "Pred", (pred_x + 10, pred_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw all tracked points
        if result.success and result.positions is not None:
            # Color based on confidence
            if result.confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif result.confidence > 0.3:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw trajectories and points with unique colors based on reference point
            for i, (pos_x, pos_y) in enumerate(result.positions):
                # Get consistent color based on reference point index
                ref_idx = result.point_ids[i] if result.point_ids and i < len(result.point_ids) else i
                point_color = self._get_point_color(ref_idx)
                
                # Draw trajectory (last few positions)
                if ref_idx in self.point_history and len(self.point_history[ref_idx]) > 1:
                    trajectory = self.point_history[ref_idx]
                    for j in range(1, len(trajectory)):
                        pt1 = trajectory[j-1]
                        pt2 = trajectory[j]
                        # Fade the trajectory line (older points are more transparent)
                        alpha = j / len(trajectory)
                        faded_color = tuple(int(c * alpha) for c in point_color)
                        cv2.line(vis_image, pt1, pt2, faded_color, 1)
                
                # Draw current point with unique color
                cv2.circle(vis_image, (pos_x, pos_y), 4, point_color, -1)
                cv2.circle(vis_image, (pos_x, pos_y), 6, point_color, 1)  # Outer ring
                
                # Show reference point ID with same color
                point_label = str(ref_idx)
                cv2.putText(vis_image, point_label, (pos_x + 8, pos_y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1)
            
            # Draw center point (larger)
            if result.position is not None:
                center_x, center_y = result.position
                cv2.circle(vis_image, (center_x, center_y), 8, color, 2)  # Hollow circle for center
                # Draw tracking info
                info_text = f"Points: {len(result.positions)}, Conf: {result.confidence:.2f}"
                cv2.putText(vis_image, info_text, (center_x + 15, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text with outline
        
        # Draw frame info
        info_y = 30
        cv2.putText(vis_image, f"Frame: {self.frame_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if result.processing_time > 0:
            fps = 1.0 / result.processing_time
            cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, info_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def plot_tracking_history(self, save_path: Optional[str] = None):
        """
        Plot tracking history and performance
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.tracking_history:
            print("No tracking history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract data
        frames = list(range(len(self.tracking_history)))
        positions_x = [r.position[0] if r.position else None for r in self.tracking_history]
        positions_y = [r.position[1] if r.position else None for r in self.tracking_history]
        confidences = [r.confidence for r in self.tracking_history]
        processing_times = [r.processing_time for r in self.tracking_history]
        
        # Plot trajectory
        valid_x = [x for x in positions_x if x is not None]
        valid_y = [y for y in positions_y if y is not None]
        if valid_x and valid_y:
            ax1.plot(valid_x, valid_y, 'b-', alpha=0.7, label='Trajectory')
            ax1.scatter(valid_x[0], valid_y[0], c='green', s=100, label='Start')
            ax1.scatter(valid_x[-1], valid_y[-1], c='red', s=100, label='End')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title('Object Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot confidence over time
        ax2.plot(frames, confidences, 'g-', alpha=0.7)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Tracking Confidence')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot processing time
        ax3.plot(frames, processing_times, 'r-', alpha=0.7)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Processing Time (s)')
        ax3.set_title('Processing Time per Frame')
        ax3.grid(True, alpha=0.3)
        
        # Plot FPS
        fps_values = [1.0/t if t > 0 else 0 for t in processing_times]
        ax4.plot(frames, fps_values, 'm-', alpha=0.7)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('FPS')
        ax4.set_title('Frames per Second')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class InteractiveROISelector:
    """
    Interactive ROI selection for target object
    """
    
    def __init__(self):
        self.roi = None
        self.drawing = False
        self.start_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection"""
        image = param['image']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw rectangle while dragging
            temp_image = image.copy()
            cv2.rectangle(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow('Select ROI', temp_image)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            
            # Calculate ROI bounds
            x_min = min(self.start_point[0], end_point[0])
            y_min = min(self.start_point[1], end_point[1])
            x_max = max(self.start_point[0], end_point[0])
            y_max = max(self.start_point[1], end_point[1])
            
            # Ensure minimum size
            if x_max - x_min > 10 and y_max - y_min > 10:
                self.roi = (x_min, y_min, x_max, y_max)
                
                # Draw final rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, "ROI Selected - Press SPACE to confirm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select ROI', image)
    
    def select_roi(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Interactive ROI selection
        
        Args:
            image: Image to select ROI from
            
        Returns:
            ROI coordinates (x_min, y_min, x_max, y_max) or None
        """
        print("Select ROI by dragging mouse. Press SPACE to confirm, ESC to cancel.")
        
        self.roi = None
        display_image = image.copy()
        
        cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select ROI', self.mouse_callback, {'image': display_image})
        
        cv2.imshow('Select ROI', display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to confirm
                if self.roi is not None:
                    cv2.destroyWindow('Select ROI')
                    return self.roi
                    
            elif key == 27:  # ESC to cancel
                cv2.destroyWindow('Select ROI')
                return None
        
        return None
