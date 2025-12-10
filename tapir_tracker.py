#!/usr/bin/env python3
"""
TAPIR / BoostTAPIR wrapper implementing the PointTrackerBase interface.

Based on the tapir_demo.py implementation from google-deepmind/tapnet.
"""

from __future__ import annotations

import os
import site
from typing import Tuple, Optional

import numpy as np
import torch
import cv2

# Configure JAX - minimal setup, let it auto-detect
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Import JAX - let it use whatever backend is available
try:
    import jax
except ImportError:
    raise RuntimeError("JAX is not installed. Please install with: pip install 'jax[cuda12]'")

from point_tracker_base import PointTrackerBase


class TAPIRTracker(PointTrackerBase):
    """
    Wrapper around TAPIR/BoostTAPIR point tracker from google-deepmind/tapnet.
    
    Implements the PointTrackerBase interface to work with KalmanTrackHybrid.
    Based on the tapir_demo.py example code.
    """

    def __init__(
        self, 
        device: torch.device,
        model_type: str = 'tapir',
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Initialize TAPIR tracker.
        
        Args:
            device: Device (TAPIR uses JAX, but we keep this for interface compatibility)
            model_type: 'tapir' or 'bootstapir' (BoostTAPIR)
            checkpoint_path: Path to checkpoint .npy file. If None, uses default locations.
        """
        super().__init__(device)
        self._model = None
        self._model_type = model_type.lower()
        self._checkpoint_path = checkpoint_path
        self._inference_fn = None
        self._tapir_model_module = None
        self._model_utils_module = None
        
    def load_model(self) -> None:
        """Load TAPIR model from checkpoint."""
        if self._model is not None:
            return

        # Import modules directly to avoid tapnet.__init__ which imports evaluation_datasets
        # that requires tensorflow_datasets (which we don't need for tracking)
        if self._tapir_model_module is None or self._model_utils_module is None:
            try:
                import importlib
                # Import modules directly without going through tapnet.__init__
                self._tapir_model_module = importlib.import_module('tapnet.models.tapir_model')
                self._model_utils_module = importlib.import_module('tapnet.utils.model_utils')
            except ImportError as e:
                # If import fails due to tensorflow_datasets, provide helpful error
                if "tensorflow_datasets" in str(e):
                    raise RuntimeError(
                        "TAPIR requires tensorflow_datasets but it's not installed. "
                        "Please install with: pip install tensorflow_datasets\n"
                        "Or install TAPIR with all dependencies: pip install git+https://github.com/google-deepmind/tapnet.git"
                    ) from e
                raise RuntimeError(
                    "TAPIR (google-deepmind/tapnet) is not installed. "
                    "Please install with: pip install git+https://github.com/google-deepmind/tapnet.git"
                ) from e

        print(f"Loading TAPIR model (type={self._model_type})...")
        
        # Determine checkpoint path
        if self._checkpoint_path is None:
            if self._model_type == 'tapir':
                checkpoint_path = 'tapnet/checkpoints/tapir_checkpoint_panning.npy'
            else:  # bootstapir or boosttapir
                checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.npy'
        else:
            checkpoint_path = self._checkpoint_path
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"TAPIR checkpoint not found at {checkpoint_path}. "
                f"Please download the checkpoint file. See TAPIR_INSTALLATION.md"
            )
        
        # Load checkpoint
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']
        
        # Create model with appropriate kwargs
        kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
        if self._model_type in ('bootstapir', 'boosttapir'):
            kwargs.update(
                dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
            )
        
        self._model = self._tapir_model_module.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
        
        # Create JIT-compiled inference function
        self._inference_fn = self._create_inference_fn()
        
        print(f"TAPIR model loaded successfully")
    
    def _create_inference_fn(self):
        """Create JIT-compiled inference function."""
        model_utils = self._model_utils_module
        
        def inference(frames, query_points):
            """
            Inference on one video.
            
            Args:
                frames: [num_frames, height, width, 3], [0, 255], np.uint8
                query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]
            
            Returns:
                tracks: [num_points, num_frames, 3], [-1, 1], [t, y, x]
                visibles: [num_points, num_frames], bool
            """
            # Preprocess video to match model inputs format
            frames = model_utils.preprocess_frames(frames)
            query_points = query_points.astype(np.float32)
            frames, query_points = frames[None], query_points[None]  # Add batch dimension

            outputs = self._model(
                video=frames,
                query_points=query_points,
                is_training=False,
                query_chunk_size=32,
            )
            tracks, occlusions, expected_dist = (
                outputs['tracks'],
                outputs['occlusion'],
                outputs['expected_dist'],
            )

            # Binarize occlusions
            visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
            return tracks[0], visibles[0]
        
        # JIT compile for performance
        return jax.jit(inference)
    
    def track(
        self,
        video_tensor: torch.Tensor,
        queries: torch.Tensor,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points for a specific frame using TAPIR.
        
        Args:
            video_tensor: [1, T, 3, H, W] float32 tensor in [0, 1] range
            queries: [1, N_points, 3] tensor with (t, x, y) in pixel coordinates
            frame_idx: Frame index to extract results from (0-indexed)
            
        Returns:
            positions: [N_points, 2] numpy array with (x, y) in pixel coordinates
            visibility: [N_points] numpy array with visibility scores (0-1)
        """
        if self._model is None or self._inference_fn is None:
            self.load_model()
        
        # Import transforms directly to avoid tapnet.__init__
        import importlib
        transforms = importlib.import_module('tapnet.utils.transforms')
        
        # Convert PyTorch tensors to numpy
        # video_tensor: [1, T, 3, H, W] -> [T, H, W, 3] uint8 [0, 255]
        video_np = video_tensor[0].cpu().numpy()  # [T, 3, H, W]
        video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, 3]
        
        # Convert from [0, 1] float to [0, 255] uint8
        if video_np.max() <= 1.0:
            video_np = (video_np * 255.0).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)
        
        # Store original dimensions for coordinate conversion
        T_full, H_orig, W_orig = video_np.shape[:3]
        
        # Resize video to 256x256 for TAPIR (much faster and less memory)
        # TAPIR is optimized for this resolution
        TAPIR_RESIZE = 256
        
        # Resize all frames
        video_resized = np.zeros((T_full, TAPIR_RESIZE, TAPIR_RESIZE, 3), dtype=np.uint8)
        for t in range(T_full):
            video_resized[t] = cv2.resize(video_np[t], (TAPIR_RESIZE, TAPIR_RESIZE), interpolation=cv2.INTER_LINEAR)
        
        # Extract video up to frame_idx (inclusive)
        video_chunk = video_resized[:frame_idx + 1]  # [T_chunk, 256, 256, 3]
        T_chunk, H_resized, W_resized = video_chunk.shape[:3]
        
        # Convert queries from [1, N, 3] (t, x, y) to [N, 3] (t, y, x)
        # Scale query coordinates to resized video dimensions
        queries_np = queries[0].cpu().numpy()  # [N_points, 3] with (t, x, y) in original coords
        N_points = queries_np.shape[0]
        
        # Scale x, y coordinates from original to resized dimensions
        scale_x = W_resized / W_orig
        scale_y = H_resized / H_orig
        
        # TAPIR expects [t, y, x] format in resized coordinates
        query_points_tapir = np.zeros((N_points, 3), dtype=np.float32)
        query_points_tapir[:, 0] = queries_np[:, 0]  # t stays t
        query_points_tapir[:, 1] = queries_np[:, 2] * scale_y  # y scaled to resized
        query_points_tapir[:, 2] = queries_np[:, 1] * scale_x  # x scaled to resized
        
        # Run inference
        tracks, visibles = self._inference_fn(video_chunk, query_points_tapir)
        
        # Convert to numpy arrays
        tracks = np.array(tracks)  # [N_points, T_chunk, 3] or [N_points, T_chunk, 2] in [-1, 1] normalized
        visibles = np.array(visibles)  # [N_points, T_chunk] bool
        
        # Extract positions for the requested frame
        # frame_idx should be within [0, T_chunk-1] since we sliced the video
        frame_idx_in_chunk = min(frame_idx, tracks.shape[1] - 1)  # Ensure valid index
        
        # Get tracks for this frame
        tracks_frame = tracks[:, frame_idx_in_chunk, :]  # [N_points, 2] or [N_points, 3]
        
        # Handle different output formats
        # TAPIR can output [t, y, x] or just [y, x] depending on version
        if tracks_frame.shape[1] == 3:
            # Format: [t, y, x]
            y_normalized = tracks_frame[:, 1]  # [N_points] in [-1, 1]
            x_normalized = tracks_frame[:, 2]  # [N_points] in [-1, 1]
        elif tracks_frame.shape[1] == 2:
            # Format: [y, x]
            y_normalized = tracks_frame[:, 0]  # [N_points] in [-1, 1]
            x_normalized = tracks_frame[:, 1]  # [N_points] in [-1, 1]
        else:
            raise ValueError(f"Unexpected tracks shape: {tracks_frame.shape}, expected 2 or 3 dimensions")
        
        # Convert from [-1, 1] normalized to resized pixel coordinates
        # Formula: pixel = (normalized + 1.0) / 2.0 * size
        y_pixels_resized = (y_normalized + 1.0) / 2.0 * H_resized
        x_pixels_resized = (x_normalized + 1.0) / 2.0 * W_resized
        
        # Scale back to original video dimensions
        scale_x_back = W_orig / W_resized
        scale_y_back = H_orig / H_resized
        y_pixels = y_pixels_resized * scale_y_back
        x_pixels = x_pixels_resized * scale_x_back
        
        # Stack as [N_points, 2] with (x, y) format (matching CoTracker3)
        positions = np.stack([x_pixels, y_pixels], axis=1)  # [N_points, 2]
        
        # Get visibility for this frame
        visibility = visibles[:, frame_idx_in_chunk].astype(np.float32)  # [N_points]
        
        return positions, visibility


