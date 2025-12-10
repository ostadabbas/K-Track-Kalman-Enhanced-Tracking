#!/usr/bin/env python3
"""
SpatialTracker wrapper implementing the generic PointTrackerBase interface.

This allows SpatialTracker to be used with the hybrid Kalman tracking system.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import torch

from point_tracker_base import PointTrackerBase

# Add SpaTracker to path
SPATRACKER_PATH = os.path.join(os.path.dirname(__file__), 'SpaTracker')
if SPATRACKER_PATH not in sys.path:
    sys.path.insert(0, SPATRACKER_PATH)

from models.spatracker.predictor import SpaTrackerPredictor
from mde import MonoDEst
from easydict import EasyDict as edict


class SpatialTrackerTracker(PointTrackerBase):
    """
    Wrapper around SpatialTracker for use with hybrid Kalman tracking.
    
    Uses SpaTrackerPredictor with depth estimation for 3D tracking.
    """

    def __init__(self, device: torch.device, grid_size: int = 40, 
                 checkpoint_path: str = None, use_depth: bool = True):
        super().__init__(device)
        self._model = None
        self._depth_estimator = None
        self.grid_size = grid_size
        self.use_depth = use_depth
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(SPATRACKER_PATH, 'checkpoints', 'spaT_final.pth')
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        """Load SpatialTracker model on the configured device."""
        if self._model is not None:
            return

        print("Loading SpatialTracker model...")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"SpatialTracker checkpoint not found at {self.checkpoint_path}\n"
                f"Please download spaT_final.pth from:\n"
                f"https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ?usp=sharing"
            )
        
        S_length = 12
        self._model = SpaTrackerPredictor(
            checkpoint=self.checkpoint_path,
            interp_shape=(384, 512),
            seq_length=S_length
        )
        self._model = self._model.to(self.device)
        self._model.eval()
        
        # Initialize depth estimator if needed
        if self.use_depth:
            print("Initializing depth estimator (ZoeDepth)...")
            original_cwd = os.getcwd()
            os.chdir(SPATRACKER_PATH)
            try:
                cfg = edict({"mde_name": "zoedepth_nk"})
                MonoDEst_O = MonoDEst(cfg)
                self._depth_estimator = MonoDEst_O.model
                self._depth_estimator.eval()
            except Exception as e:
                print(f"Warning: Could not initialize depth estimator: {e}")
                self._depth_estimator = None
            finally:
                os.chdir(original_cwd)
        
        print(f"SpatialTracker loaded on {self.device}")

    def track(
        self,
        video_tensor: torch.Tensor,
        queries: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SpatialTracker on video up to the current frame and return predictions.
        
        Args:
            video_tensor: [1, T, 3, H, W] tensor on the correct device
            queries: [1, N_points, 3] tensor (t, x, y) on the correct device
            frame_idx: Index of the frame to return predictions for
            
        Returns:
            positions: [N_points, 2] numpy array (x, y) - only 2D for compatibility
            visibility: [N_points] numpy array with visibility scores in [0, 1]
        """
        if self._model is None:
            self.load_model()

        with torch.no_grad():
            # SpatialTracker works on full video sequences
            # We'll run it on the video up to frame_idx
            video_chunk = video_tensor[:, :frame_idx + 1]
            
            # Convert queries to grid_size if needed, or use queries directly
            # For now, we'll use the queries as-is
            # Note: SpatialTracker expects queries in format [1, N, 3] with (t, x, y)
            
            # Run SpatialTracker
            # If we have queries, use them; otherwise use grid_size
            if queries is not None and queries.shape[1] > 0:
                # Use provided queries
                pred_tracks, pred_visibility, T_Firsts = self._model(
                    video_chunk,
                    video_depth=None,
                    queries=queries,
                    segm_mask=None,
                    grid_size=0,  # Use queries instead of grid
                    backward_tracking=False,
                    depth_predictor=self._depth_estimator,
                    wind_length=12
                )
            else:
                # Use grid_size to generate points
                pred_tracks, pred_visibility, T_Firsts = self._model(
                    video_chunk,
                    video_depth=None,
                    queries=None,
                    segm_mask=None,
                    grid_size=self.grid_size,
                    backward_tracking=False,
                    depth_predictor=self._depth_estimator,
                    wind_length=12
                )
            
            # Extract predictions for the current frame
            # pred_tracks is [1, T, N, 3] (x, y, z)
            # We only need x, y for 2D tracking
            positions_3d = pred_tracks[0, frame_idx].cpu().numpy()  # [N_points, 3]
            positions = positions_3d[:, :2]  # [N_points, 2] - extract only x, y
            
            # Visibility
            visibility = pred_visibility[0, frame_idx].cpu().numpy()  # [N_points]
            # Convert boolean to float if needed
            if visibility.dtype == bool:
                visibility = visibility.astype(np.float32)

        return positions, visibility

