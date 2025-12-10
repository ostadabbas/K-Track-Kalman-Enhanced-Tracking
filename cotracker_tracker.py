#!/usr/bin/env python3
"""
CoTracker3 wrapper implementing the generic PointTrackerBase interface.

This isolates all direct CoTracker3 calls so that the hybrid Kalman tracker
can remain agnostic to the underlying deep model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from point_tracker_base import PointTrackerBase


class CoTracker3Tracker(PointTrackerBase):
    """
    Thin wrapper around the official CoTracker3 offline model.

    API matches the usage currently in `kalman_hybrid.KalmanTrackHybrid`:
    - Uses `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`
    - Runs the model on the video up to the requested frame.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
        self._model = None

    def load_model(self) -> None:
        """Load CoTracker3 offline model on the configured device."""
        if self._model is not None:
            return

        print("Loading CoTracker3 offline model (via CoTracker3Tracker)...")
        self._model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"CoTracker3 loaded on {self.device}")

    def track(
        self,
        video_tensor: torch.Tensor,
        queries: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run CoTracker3 on video up to the current frame and return predictions.
        """
        if self._model is None:
            self.load_model()

        with torch.no_grad():
            # Use the same protocol as the original implementation:
            # run on video up to frame_idx (inclusive).
            video_chunk = video_tensor[:, : frame_idx + 1]
            pred_tracks, pred_visibility = self._model(video_chunk, queries=queries)

        # Extract predictions for the current frame.
        positions = pred_tracks[0, frame_idx].cpu().numpy()  # [N_points, 2]
        visibility = pred_visibility[0, frame_idx].cpu().numpy()  # [N_points]

        return positions, visibility


