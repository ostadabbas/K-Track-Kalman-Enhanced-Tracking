"""
Kalman Filter implementation for object motion prediction
Based on constant velocity model for 2D tracking
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class KalmanTracker:
    """
    2D Kalman Filter for object tracking with constant velocity model
    State vector: [x, y, vx, vy] - position and velocity
    """
    
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.5):
        """
        Initialize Kalman filter
        
        Args:
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (observation uncertainty)
        """
        # Create Kalman filter: 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix H: we observe position (x, y) but not velocity
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 1, 0, 0]   # y measurement
        ], dtype=np.float32)
        
        # State transition matrix F: constant velocity model
        # x_new = x + vx*dt, y_new = y + vy*dt, vx_new = vx, vy_new = vy
        dt = 1.0  # time step (assuming 1 frame = 1 time unit)
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ], dtype=np.float32)
        
        # Process noise covariance Q
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance R
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance matrix P (initial uncertainty)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
        # Tracking state
        self.initialized = False
        self.last_measurement = None
        self.prediction_count = 0  # Count consecutive predictions without measurements
        self.max_prediction_count = 10  # Maximum predictions before considering lost
        
    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """
        Initialize the filter with initial position and velocity
        
        Args:
            x, y: Initial position
            vx, vy: Initial velocity (default: 0)
        """
        # Set initial state [x, y, vx, vy]
        self.kalman.statePre = np.array([x, y, vx, vy], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, vx, vy], dtype=np.float32)
        
        self.initialized = True
        self.last_measurement = (x, y)
        self.prediction_count = 0
        
        print(f"Kalman filter initialized at position ({x:.1f}, {y:.1f})")
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next state (position)
        
        Returns:
            Predicted (x, y) position
        """
        if not self.initialized:
            raise ValueError("Kalman filter not initialized. Call initialize() first.")
        
        # Predict next state
        prediction = self.kalman.predict()
        
        # Extract position from state vector
        pred_x, pred_y = prediction[0], prediction[1]
        
        self.prediction_count += 1
        
        return float(pred_x), float(pred_y)
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update filter with new measurement
        
        Args:
            x, y: Measured position
            
        Returns:
            Corrected (x, y) position
        """
        if not self.initialized:
            # Initialize with first measurement
            self.initialize(x, y)
            return x, y
        
        # Create measurement vector
        measurement = np.array([x, y], dtype=np.float32)
        
        # Correct prediction with measurement
        corrected_state = self.kalman.correct(measurement)
        
        # Extract corrected position
        corr_x, corr_y = corrected_state[0], corrected_state[1]
        
        self.last_measurement = (x, y)
        self.prediction_count = 0  # Reset prediction count
        
        return float(corr_x), float(corr_y)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate
        
        Returns:
            Velocity (vx, vy)
        """
        if not self.initialized:
            return 0.0, 0.0
        
        state = self.kalman.statePost
        return float(state[2]), float(state[3])
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Get current state estimate
        
        Returns:
            State (x, y, vx, vy)
        """
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        
        state = self.kalman.statePost
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])
    
    def is_lost(self) -> bool:
        """
        Check if tracking is likely lost (too many predictions without measurements)
        
        Returns:
            True if tracking is likely lost
        """
        return self.prediction_count > self.max_prediction_count
    
    def reset(self):
        """Reset the filter to uninitialized state"""
        self.initialized = False
        self.last_measurement = None
        self.prediction_count = 0


class MultiObjectKalmanTracker:
    """
    Multi-object tracker using multiple Kalman filters
    """
    
    def __init__(self, max_objects: int = 10, process_noise: float = 0.03, 
                 measurement_noise: float = 0.5):
        """
        Initialize multi-object tracker
        
        Args:
            max_objects: Maximum number of objects to track
            process_noise: Process noise for individual filters
            measurement_noise: Measurement noise for individual filters
        """
        self.max_objects = max_objects
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Dictionary to store trackers by ID
        self.trackers = {}
        self.next_id = 0
        
        # Tracking parameters
        self.max_distance = 50.0  # Maximum distance for association
        
    def add_tracker(self, x: float, y: float) -> int:
        """
        Add new tracker for object at given position
        
        Args:
            x, y: Initial position
            
        Returns:
            Tracker ID
        """
        if len(self.trackers) >= self.max_objects:
            print(f"Maximum number of trackers ({self.max_objects}) reached")
            return -1
        
        tracker_id = self.next_id
        self.next_id += 1
        
        tracker = KalmanTracker(self.process_noise, self.measurement_noise)
        tracker.initialize(x, y)
        
        self.trackers[tracker_id] = tracker
        
        print(f"Added tracker {tracker_id} at position ({x:.1f}, {y:.1f})")
        return tracker_id
    
    def update_trackers(self, measurements: list) -> dict:
        """
        Update all trackers with new measurements
        
        Args:
            measurements: List of (x, y) measurements
            
        Returns:
            Dictionary mapping tracker_id to updated position
        """
        # Predict all trackers
        predictions = {}
        for tracker_id, tracker in self.trackers.items():
            if not tracker.is_lost():
                pred_x, pred_y = tracker.predict()
                predictions[tracker_id] = (pred_x, pred_y)
        
        # Associate measurements with trackers
        associations = self._associate_measurements(predictions, measurements)
        
        # Update trackers with associated measurements
        updated_positions = {}
        for tracker_id, measurement_idx in associations.items():
            if measurement_idx is not None:
                x, y = measurements[measurement_idx]
                corr_x, corr_y = self.trackers[tracker_id].update(x, y)
                updated_positions[tracker_id] = (corr_x, corr_y)
            else:
                # No measurement, use prediction
                if tracker_id in predictions:
                    updated_positions[tracker_id] = predictions[tracker_id]
        
        # Remove lost trackers
        lost_trackers = [tid for tid, tracker in self.trackers.items() if tracker.is_lost()]
        for tid in lost_trackers:
            print(f"Removing lost tracker {tid}")
            del self.trackers[tid]
        
        return updated_positions
    
    def _associate_measurements(self, predictions: dict, measurements: list) -> dict:
        """
        Associate measurements with tracker predictions using nearest neighbor
        
        Args:
            predictions: Dictionary of tracker predictions
            measurements: List of measurements
            
        Returns:
            Dictionary mapping tracker_id to measurement index (or None)
        """
        associations = {}
        used_measurements = set()
        
        # Calculate distances between predictions and measurements
        for tracker_id, (pred_x, pred_y) in predictions.items():
            best_distance = float('inf')
            best_measurement_idx = None
            
            for i, (meas_x, meas_y) in enumerate(measurements):
                if i in used_measurements:
                    continue
                
                distance = np.sqrt((pred_x - meas_x)**2 + (pred_y - meas_y)**2)
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_measurement_idx = i
            
            if best_measurement_idx is not None:
                associations[tracker_id] = best_measurement_idx
                used_measurements.add(best_measurement_idx)
            else:
                associations[tracker_id] = None
        
        return associations
    
    def get_all_states(self) -> dict:
        """
        Get states of all active trackers
        
        Returns:
            Dictionary mapping tracker_id to (x, y, vx, vy)
        """
        states = {}
        for tracker_id, tracker in self.trackers.items():
            if not tracker.is_lost():
                states[tracker_id] = tracker.get_state()
        return states
    
    def reset_all(self):
        """Reset all trackers"""
        self.trackers.clear()
        self.next_id = 0
