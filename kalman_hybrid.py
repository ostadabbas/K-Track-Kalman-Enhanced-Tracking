#!/usr/bin/env python3
"""
KalmanTrack: Hybrid CoTracker3 + Kalman Filter Point Tracker
Reduces computational cost by running CoTracker3 every N frames
and using Kalman filtering for intermediate frame predictions.

EECE 7398: Bayesian Filtering - Fall 2025
"""

import numpy as np
import torch
import cv2
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import time

from point_tracker_base import PointTrackerBase
from cotracker_tracker import CoTracker3Tracker
# from tapir_tracker import TAPIRTracker
try:
    from spatracker_tracker import SpatialTrackerTracker
except ImportError:
    SpatialTrackerTracker = None
try:
    from trackon_tracker import TrackOnTracker
except ImportError:
    TrackOnTracker = None


@dataclass
class HybridTrackingResult:
    """Result from hybrid tracking"""
    frame_idx: int
    positions: np.ndarray  # [N_points, 2]
    velocities: np.ndarray  # [N_points, 2]
    covariances: np.ndarray  # [N_points, 4, 4]
    used_cotracker: bool  # Whether CoTracker3 was run this frame
    visibility: np.ndarray  # [N_points]
    processing_time: float


class ConstantVelocityKalmanFilter:
    """
    Kalman Filter with constant velocity model for point tracking
    
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    
    def __init__(self, dt: float = 1.0,
                 process_var_pos: float = 1e-4,
                 process_var_vel: float = 1e-2,
                 meas_var_pos: float = 0.1):
        """
        Initialize Kalman filter
        
        Args:
            dt: Time step between frames
            process_var_pos: Process noise variance for position
            process_var_vel: Process noise variance for velocity
            meas_var_pos: Measurement noise variance
        """
        self.dt = dt
        
        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.diag([
            process_var_pos,
            process_var_pos,
            process_var_vel,
            process_var_vel
        ]).astype(np.float32)
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * meas_var_pos
        
        # State and covariance (will be initialized per point)
        self.x = None  # [4] state vector
        self.P = None  # [4, 4] covariance matrix
        
    def initialize(self, position: np.ndarray, velocity: Optional[np.ndarray] = None):
        """
        Initialize filter state
        
        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy] (optional, defaults to zero)
        """
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float32)
        
        self.x = np.concatenate([position, velocity]).astype(np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0  # Initial uncertainty
        
    def predict(self, q_scale: float = 1.0) -> None:
        """
        Prediction step with optional scalar scaling of process noise.
        """
        # Predict state
        self.x = self.F @ self.x
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + (self.Q * float(q_scale))
    
    def update(self, measurement: np.ndarray, measurement_noise: Optional[float] = None):
        """
        Update step with measurement
        
        Args:
            measurement: Measured position [x, y]
            measurement_noise: Optional custom measurement noise variance
        """
        # Use custom measurement noise if provided
        R = self.R if measurement_noise is None else np.eye(2) * measurement_noise
        
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current state
        
        Returns:
            position [x, y], velocity [vx, vy], covariance [4, 4]
        """
        return self.x[:2].copy(), self.x[2:].copy(), self.P.copy()


class ConstantAccelerationKalmanFilter:
    """Unused (CA experiments removed); kept only for backward compatibility."""
    pass


class IMMCVCAFilter:
    """Unused (IMM experiments removed); kept only for backward compatibility."""
    pass


class KalmanTrackHybrid:
    """
    Hybrid tracker combining a deep point tracker with Kalman filtering.

    Originally this used CoTracker3; it is now pluggable and can work with
    other trackers (e.g., TAPIR) that implement `PointTrackerBase`.
    """
    
    def __init__(self,
                 N: int = 5,
                 warmup: int = 3,
                 process_var_pos: float = 1e-4,
                 process_var_vel: float = 1e-2,
                 meas_var_pos: float = 0.1,
                 device: str = 'cuda',
                 tracker_type: str = 'cotracker3',
                 grid_size: int = 40,
                 trackon_checkpoint: str = None,
                 trackon_config: str = None):
        """
        Initialize hybrid tracker
        
        Args:
            N: Run deep tracker every N frames (after warmup)
            warmup: Number of initial frames to always run tracker (for velocity initialization)
            process_var_pos: Kalman process noise for position
            process_var_vel: Kalman process noise for velocity
            meas_var_pos: Kalman measurement noise
            device: Device for the deep tracker
            tracker_type: Which deep tracker to use ('cotracker3', 'tapir', 'spatracker', 'trackon')
            grid_size: Grid size for SpatialTracker (only used if tracker_type='spatracker')
            trackon_checkpoint: Path to Track-On checkpoint (only used if tracker_type='trackon')
            trackon_config: Path to Track-On config file (only used if tracker_type='trackon')
        """
        self.configured_N = N
        self.N = max(1, N if N is not None else 1)
        self.warmup = max(0, warmup if warmup is not None else 0)
        self.cotracker_only = N == 0
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tracker_type = (tracker_type or 'cotracker3').lower()
        self.grid_size = grid_size
        self.trackon_checkpoint = trackon_checkpoint
        self.trackon_config = trackon_config
        
        # Kalman filter parameters
        self.process_var_pos = process_var_pos
        self.process_var_vel = process_var_vel
        self.meas_var_pos = meas_var_pos
        # Deep tracker wrapper (loaded lazily)
        self.tracker: Optional[PointTrackerBase] = None
        
        # Tracking state
        self.kalman_filters = []  # One filter per point
        self.frame_count = 0
        self.n_points = 0
        self.warmup_positions = []  # Store positions during warmup for velocity estimation
        
        # Performance tracking
        self.cotracker_calls = 0
        self.total_frames = 0

        # No IMM or noise scheduling in the clean CV configuration

    def _create_tracker(self) -> PointTrackerBase:
        """Instantiate the appropriate deep tracker wrapper."""
        if self.tracker_type in ("cotracker3", "cotracker"):
            return CoTracker3Tracker(self.device)
        if self.tracker_type == "tapir":
            return TAPIRTracker(self.device, model_type='tapir')
        if self.tracker_type in ("boosttapir", "boost_tapir"):
            return TAPIRTracker(self.device, model_type='bootstapir')
        if self.tracker_type in ("spatracker", "spatialtracker"):
            if SpatialTrackerTracker is None:
                raise ImportError("SpatialTrackerTracker not available. Make sure SpaTracker is set up.")
            # Get grid_size from kwargs if provided
            grid_size = getattr(self, 'grid_size', 40)
            return SpatialTrackerTracker(self.device, grid_size=grid_size)
        if self.tracker_type in ("trackon", "track-on", "trackon2"):
            if TrackOnTracker is None:
                raise ImportError("TrackOnTracker not available. Make sure track_on is set up.")
            return TrackOnTracker(
                self.device, 
                checkpoint_path=self.trackon_checkpoint,
                config_path=self.trackon_config
            )
        raise ValueError(f"Unknown tracker_type '{self.tracker_type}'")

    def _ensure_tracker_loaded(self) -> None:
        """Lazy construction and model loading for the deep tracker."""
        if self.tracker is None:
            self.tracker = self._create_tracker()
        # Allow tracker implementation to handle idempotent loading
        self.tracker.load_model()
    
    def initialize(self, video_tensor: torch.Tensor, initial_points: np.ndarray):
        """
        Initialize tracking with first frame
        
        Args:
            video_tensor: Video tensor [1, T, 3, H, W]
            initial_points: Initial point positions [N_points, 2] (x, y)
        """
        # Ensure the deep tracker is ready
        self._ensure_tracker_loaded()
        
        self.n_points = initial_points.shape[0]
        self.frame_count = 0
        self.warmup_positions = []
        
        # Initialize Kalman filters for each point (pure CV model)
        # Velocity will be estimated during warmup if warmup > 0
        self.kalman_filters = []
        for i in range(self.n_points):
            kf = ConstantVelocityKalmanFilter(
                dt=1.0,
                process_var_pos=self.process_var_pos,
                process_var_vel=self.process_var_vel,
                meas_var_pos=self.meas_var_pos,
            )
            kf.initialize(initial_points[i])  # Initialize with zero velocity for now
            self.kalman_filters.append(kf)
        
        print(f"Initialized {self.n_points} Kalman filters (warmup={self.warmup})")
    
    def track_frame(self, video_tensor: torch.Tensor, 
                   queries: torch.Tensor,
                   frame_idx: int) -> HybridTrackingResult:
        """
        Track points in a single frame
        
        Args:
            video_tensor: Video tensor [1, T, 3, H, W]
            queries: Query points [1, N_points, 3] (t, x, y)
            frame_idx: Current frame index
            
        Returns:
            HybridTrackingResult with positions and metadata
        """
        start_time = time.time()
        used_cotracker = False
        
        # During warmup, always run tracker to get measurements for velocity estimation
        # After warmup, run tracker every N frames
        in_warmup = frame_idx < self.warmup
        run_cotracker_this_frame = (self.cotracker_only or 
                                   in_warmup or 
                                   frame_idx % self.N == 0)
        
        if run_cotracker_this_frame:
            # Run deep tracker on this keyframe
            positions, visibility = self._run_tracker(video_tensor, queries, frame_idx)
            used_cotracker = True
            self.cotracker_calls += 1
            
            # Update Kalman filters with measurements
            for i in range(self.n_points):
                if visibility[i] > 0.5:  # Only update if visible
                    self.kalman_filters[i].update(positions[i])
        else:
            # Use Kalman prediction only (constant process noise)
            positions = np.zeros((self.n_points, 2))
            visibility = np.ones(self.n_points)

            for i in range(self.n_points):
                self.kalman_filters[i].predict()
                pos, _, _ = self.kalman_filters[i].get_state()
                positions[i] = pos
        
        # Get velocities and covariances
        velocities = np.zeros((self.n_points, 2))
        covariances = np.zeros((self.n_points, 4, 4))
        
        for i in range(self.n_points):
            pos, vel, cov = self.kalman_filters[i].get_state()
            velocities[i] = vel
            covariances[i] = cov
        
        processing_time = time.time() - start_time
        self.total_frames += 1
        
        return HybridTrackingResult(
            frame_idx=frame_idx,
            positions=positions,
            velocities=velocities,
            covariances=covariances,
            used_cotracker=used_cotracker,
            visibility=visibility,
            processing_time=processing_time
        )
    
    def _run_tracker(self, video_tensor: torch.Tensor,
                     queries: torch.Tensor,
                     frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the underlying deep tracker on the current frame.
        
        Args:
            video_tensor: Video tensor
            queries: Query points
            frame_idx: Current frame
            
        Returns:
            positions [N_points, 2], visibility [N_points]
        """
        if self.tracker is None:
            self._ensure_tracker_loaded()
        return self.tracker.track(video_tensor, queries, frame_idx)
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_frames': self.total_frames,
            'cotracker_calls': self.cotracker_calls,
            'kalman_predictions': self.total_frames - self.cotracker_calls,
            'speedup_factor': self.total_frames / max(self.cotracker_calls, 1),
            'cotracker_frequency': "every frame (baseline)" if self.cotracker_only else f"1/{self.N} frames"
        }


if __name__ == "__main__":
    print("KalmanTrack Hybrid Tracker")
    print("=" * 50)
    print("\nThis module provides hybrid CoTracker3 + Kalman filtering")
    print("for efficient point tracking.")
    print("\nUsage:")
    print("  from kalman_hybrid import KalmanTrackHybrid")
    print("  tracker = KalmanTrackHybrid(N=5)")
    print("  tracker.initialize(video, initial_points)")
    print("  result = tracker.track_frame(video, queries, frame_idx)")
