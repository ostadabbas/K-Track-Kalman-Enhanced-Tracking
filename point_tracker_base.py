#!/usr/bin/env python3
"""
Abstract interface for deep point trackers used in the hybrid Kalman system.

This lets us plug in different trackers (e.g., CoTracker3, TAPIR) behind a
common API so that `KalmanTrackHybrid` does not depend on a specific model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch


class PointTrackerBase(ABC):
    """
    Base interface for deep-learning point trackers.

    All implementations must:
    - Accept a full video tensor [1, T, 3, H, W]
    - Accept query points [1, N_points, 3] with (t, x, y)
    - Return positions [N_points, 2] and visibility scores [N_points]
      for the requested frame index.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def load_model(self) -> None:
        """Load the underlying model on `self.device` (idempotent)."""

    @abstractmethod
    def track(
        self,
        video_tensor: torch.Tensor,
        queries: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points for a specific frame.

        Args:
            video_tensor: [1, T, 3, H, W] tensor on the correct device.
            queries: [1, N_points, 3] tensor (t, x, y) on the correct device.
            frame_idx: Index of the frame to return predictions for.

        Returns:
            positions: [N_points, 2] numpy array (x, y)
            visibility: [N_points] numpy array with visibility scores in [0, 1]
        """


