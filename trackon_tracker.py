#!/usr/bin/env python3
"""
Track-On wrapper implementing the generic PointTrackerBase interface.

This allows Track-On to be used with the hybrid Kalman tracking system.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import torch

from point_tracker_base import PointTrackerBase

# Add track_on to path - this must be done before any imports
TRACKON_PATH = os.path.join(os.path.dirname(__file__), 'track_on')
if TRACKON_PATH not in sys.path:
    sys.path.insert(0, TRACKON_PATH)

try:
    # Import modules - Python should be able to resolve relative imports
    # since track_on is in sys.path
    from model.trackon_predictor import Predictor
    from utils.train_utils import load_args_from_yaml
    IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as e:
    Predictor = None
    load_args_from_yaml = None
    IMPORT_ERROR = str(e)


class TrackOnTracker(PointTrackerBase):
    """
    Wrapper around Track-On for use with hybrid Kalman tracking.
    
    Track-On is an online point tracking model that processes videos
    frame-by-frame with a compact transformer memory.
    """

    def __init__(self, device: torch.device, 
                 checkpoint_path: str = None,
                 config_path: str = None):
        super().__init__(device)
        self._model = None
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Default paths
        if self.checkpoint_path is None:
            self.checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                'track-on-weights', 
                'trackon2_dinov2_checkpoint.pt'
            )
        
        if self.config_path is None:
            self.config_path = os.path.join(
                TRACKON_PATH, 
                'config', 
                'test_dinov2.yaml'
            )

    def load_model(self) -> None:
        """Load Track-On model on the configured device."""
        if self._model is not None:
            return

        if Predictor is None:
            raise ImportError(
                f"Track-On dependencies not available. Original error: {IMPORT_ERROR}\n"
                f"Make sure track_on is properly installed and dependencies are met."
            )

        print("Loading Track-On model...")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Track-On checkpoint not found at {self.checkpoint_path}\n"
                f"Please download trackon2_dinov2_checkpoint.pt and place it in track-on-weights/"
            )
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Track-On config not found at {self.config_path}\n"
                f"Expected config file at track_on/config/test_dinov2.yaml"
            )
        
        # Load config
        model_args = load_args_from_yaml(self.config_path)
        
        # Initialize model with support_grid_size=0 since we provide our own queries
        # The support grid is meant for when queries=None, but we always provide queries
        # Setting it to 0 prevents adding extra grid points that can cause jumpy behavior
        self._model = Predictor(model_args, checkpoint_path=self.checkpoint_path, support_grid_size=0)
        self._model = self._model.to(self.device)
        self._model.eval()
        
        print(f"Track-On loaded on {self.device}")

    def track(
        self,
        video_tensor: torch.Tensor,
        queries: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Track-On on video up to the current frame and return predictions.
        
        IMPORTANT: Track-On is designed to process the entire video at once.
        For incremental calls, we process the full video chunk each time.
        This is less efficient but necessary for correct tracking.
        
        Args:
            video_tensor: [1, T, 3, H, W] tensor on the correct device
            queries: [1, N_points, 3] tensor (t, x, y) on the correct device
            frame_idx: Index of the frame to return predictions for
            
        Returns:
            positions: [N_points, 2] numpy array (x, y)
            visibility: [N_points] numpy array with visibility scores in [0, 1]
        """
        if self._model is None:
            self.load_model()

        with torch.no_grad():
            # Track-On expects video in format (1, T, 3, H, W) - same as what we receive
            # Process the entire video chunk up to current frame
            video_chunk = video_tensor[:, :frame_idx + 1]  # [1, T', 3, H, W]
            
            # IMPORTANT: Make a deep copy of queries because Track-On modifies them in-place
            # Track-On normalizes query coordinates by dividing by video dimensions (line 297-298 in trackon.py)
            # If we reuse the same tensor, coordinates get corrupted after first call
            # Use .clone() to ensure we have a fresh copy each time
            queries_copy = queries.clone().detach()
            
            # Track-On processes the entire chunk at once
            # Each call processes from frame 0 to frame_idx and returns all frames
            # This matches how Track-On's forward_online works - it processes all frames in the chunk
            pred_tracks, pred_visibility = self._model(video_chunk, queries=queries_copy)
            
            # Extract predictions for the current frame
            # pred_tracks is [1, T', N, 2] where T' = frame_idx + 1
            # pred_visibility is [1, T', N]
            # frame_idx is the index within the chunk (0-indexed)
            if frame_idx >= pred_tracks.shape[1]:
                # Safety check: if frame_idx is out of bounds, use last frame
                frame_idx = pred_tracks.shape[1] - 1
            
            positions = pred_tracks[0, frame_idx].cpu().numpy()  # [N_points, 2]
            visibility = pred_visibility[0, frame_idx].cpu().numpy()  # [N_points]
            
            # Convert boolean visibility to float if needed
            if visibility.dtype == bool:
                visibility = visibility.astype(np.float32)
            elif visibility.dtype in (np.int32, np.int64):
                visibility = visibility.astype(np.float32)

        return positions, visibility

