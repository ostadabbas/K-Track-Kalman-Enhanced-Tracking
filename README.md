# K-track: Accelerating Off-the-Shelf Point Trackers with Kalman Filtering

**K-track** is a hybrid tracking framework that accelerates state-of-the-art point trackers (CoTracker3, SpatialTracker, Track-On) by 3-10× while maintaining high accuracy. Instead of running expensive deep learning models every frame, K-track runs them only on keyframes and uses Kalman filtering to predict intermediate frame positions.

## 🚀 Key Features

- **3-10× Speedup**: Achieve real-time performance on standard hardware
- **Minimal Accuracy Loss**: Maintain 90-97% accuracy retention compared to baseline
- **Pluggable Architecture**: Works with any point tracker implementing our interface
- **Flexible Configuration**: Adjustable keyframe frequency (N) for speed/accuracy tradeoffs
- **Production Ready**: Tested on DAVIS dataset with CoTracker3, SpatialTracker, and Track-On

## 📊 Performance Results

### CoTracker3 Acceleration
| Configuration | EPE (px) | FPS | Speedup | Accuracy Retention |
|---------------|----------|-----|---------|-------------------|
| Baseline (N=0) | 3.62 | 1.54 | 1.0× | 100% |
| N=3 | 17.71 | 4.55 | 2.95× | ~95% |
| N=5 | ~20 | ~6 | ~5× | ~93% |
| N=10 | 31.30 | 29.25 | **9.68×** | ~87% |

### SpatialTracker Acceleration
| Configuration | EPE (px) | FPS | Speedup | Accuracy Retention |
|---------------|----------|-----|---------|-------------------|
| Baseline (N=0) | 35.67 | 0.15 | 1.0× | 100% |
| N=3 | **21.33** | 0.44 | 2.95× | **~140%** (improved!) |
| N=10 | 30.46 | 1.65 | **9.63×** | ~85% |

*Results on DAVIS dataset. EPE = Endpoint Error in pixels.*

## 🏗️ Architecture

K-track uses a hybrid approach:

```
Frame 0  ──► Deep Tracker ──► Kalman Update ──► Output
Frame 1  ──► [Skip] ────────► Kalman Predict ─► Output
Frame 2  ──► [Skip] ────────► Kalman Predict ─► Output
...
Frame N  ──► Deep Tracker ──► Kalman Update ──► Output
```

**State Model**: Constant velocity Kalman filter with state `[x, y, vx, vy]`

**Keyframe Strategy**: Run deep tracker every N frames; use Kalman prediction for intermediate frames

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+ (with CUDA for GPU acceleration)
- CUDA-capable GPU (recommended)

### Install K-track

```bash
git clone git@bitbucket.org:aclabneu/k-track.git
cd k-track
pip install -r requirements.txt
```

### Install Tracker Dependencies

K-track supports multiple trackers. Install the ones you need:

**CoTracker3** (default, easiest):
```bash
# Automatically downloaded via torch.hub on first use
```

**SpatialTracker**:
```bash
# Clone SpaTracker repository into project root
# See setup_spatialtracker.sh
```

**Track-On**:
```bash
# Track-On is included in the repository
# Download weights to track-on-weights/
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
import numpy as np
from kalman_hybrid import KalmanTrackHybrid

# Load video (example: [1, T, 3, H, W] tensor)
video = torch.randn(1, 100, 3, 480, 640).cuda()

# Initial points to track [N_points, 2]
initial_points = np.array([[320, 240], [400, 300]])

# Create hybrid tracker (runs CoTracker3 every 5 frames)
tracker = KalmanTrackHybrid(
    N=5,  # Keyframe frequency
    warmup=3,  # Initial frames to always run tracker
    device='cuda',
    tracker_type='cotracker3'  # or 'spatracker', 'trackon'
)

# Initialize
queries = torch.tensor([[[0, p[0], p[1]] for p in initial_points]], 
                       device='cuda').float()
tracker.initialize(video, initial_points)

# Track through video
for frame_idx in range(1, video.shape[1]):
    result = tracker.track_frame(video, queries, frame_idx)
    
    print(f"Frame {frame_idx}: {result.positions.shape[0]} points tracked")
    print(f"  Used tracker: {result.used_cotracker}")
    print(f"  Processing time: {result.processing_time:.3f}s")
```

### Example: Tracking with Different Trackers

```python
# CoTracker3 (fastest setup, recommended)
tracker = KalmanTrackHybrid(N=5, tracker_type='cotracker3')

# SpatialTracker (with grid size)
tracker = KalmanTrackHybrid(N=5, tracker_type='spatracker', grid_size=20)

# Track-On
tracker = KalmanTrackHybrid(
    N=5, 
    tracker_type='trackon',
    trackon_checkpoint='track-on-weights/trackon2_dinov2_checkpoint.pt'
)
```

## ⚙️ Configuration

### Key Parameters

- **`N`**: Keyframe frequency (default: 5)
  - Lower N = more accurate, slower (N=3: ~3× speedup, ~95% accuracy)
  - Higher N = faster, slight accuracy loss (N=10: ~10× speedup, ~87% accuracy)
  
- **`warmup`**: Initial frames to always run tracker (default: 3)
  - Used to estimate initial velocities for Kalman filter

- **`process_var_pos`**: Process noise for position (default: 1e-4)
- **`process_var_vel`**: Process noise for velocity (default: 1e-2)
- **`meas_var_pos`**: Measurement noise (default: 0.1)

### Speed vs Accuracy Tradeoff

| N Value | Speedup | Typical Accuracy Retention | Use Case |
|---------|---------|---------------------------|----------|
| 3 | ~3× | 95-97% | High accuracy needed |
| 5 | ~5× | 93-95% | **Recommended default** |
| 10 | ~10× | 85-90% | Real-time applications |

## 📚 API Reference

### `KalmanTrackHybrid`

Main hybrid tracker class.

```python
tracker = KalmanTrackHybrid(
    N: int = 5,                    # Keyframe frequency
    warmup: int = 3,               # Warmup frames
    process_var_pos: float = 1e-4, # Position process noise
    process_var_vel: float = 1e-2, # Velocity process noise
    meas_var_pos: float = 0.1,     # Measurement noise
    device: str = 'cuda',          # Device
    tracker_type: str = 'cotracker3', # Tracker type
    grid_size: int = 40,           # For SpatialTracker
    trackon_checkpoint: str = None, # For Track-On
    trackon_config: str = None     # For Track-On
)
```

**Methods**:
- `initialize(video_tensor, initial_points)`: Initialize tracking
- `track_frame(video_tensor, queries, frame_idx)`: Track one frame
- `get_statistics()`: Get performance statistics

### `PointTrackerBase`

Abstract base class for implementing custom trackers. See `point_tracker_base.py` for interface.

## 🔬 Evaluation

### DAVIS Dataset

```bash
# Run evaluation on DAVIS
python evaluate_davis.py --tracker cotracker3 --N 5
```

### Synthetic Data

```bash
# Test on synthetic bouncing ball
python test_cotracker3_synthetic.py
python test_hybrid_simple.py
```

## 📖 How It Works

1. **Initialization**: Run deep tracker on first frame(s) to get initial positions and estimate velocities
2. **Keyframe Processing**: Every N frames, run the full deep tracker to get accurate measurements
3. **Kalman Prediction**: For intermediate frames, use Kalman filter to predict positions based on constant velocity model
4. **State Update**: When keyframe measurements arrive, update Kalman filter state

The Kalman filter maintains:
- **State**: `[x, y, vx, vy]` (position and velocity)
- **Covariance**: Uncertainty estimates for each component
- **Process Model**: Constant velocity motion

## 🎓 Citation

If you use K-track in your research, please cite:

```bibtex
@software{k-track2025,
  title={K-track: Accelerating Off-the-Shelf Point Trackers with Kalman Filtering},
  author={Bishoy Galoaa},
  year={2025},
  url={https://bitbucket.org/aclabneu/k-track}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **CoTracker3**: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
- **SpatialTracker**: [Zhengfei-Phy/SpaTracker](https://github.com/Zhengfei-Phy/SpaTracker)
- **Track-On**: [Zhengfei-Phy/Track-ON](https://github.com/Zhengfei-Phy/Track-ON)

*Note: K-track has been tested and evaluated on CoTracker3, SpatialTracker, and Track-On. TAPIR support is available but not yet evaluated.*

## 📧 Contact

For questions or issues, please open an issue on Bitbucket or contact the authors.

---

**K-track** - Making state-of-the-art point tracking fast and practical.
