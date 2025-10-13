# KalmanTrack: Accelerating CoTracker3 with Kalman Filtering

**EECE 7398: Bayesian Filtering - Fall 2025**  
**Project Phase II: Synthetic Data Evaluation**

---

## 🎯 Project Objective

Reduce the computational cost of CoTracker3 point tracking by using a **hybrid approach**:
- Run CoTracker3 every **N frames** (keyframes) for accurate measurements
- Use **Kalman filtering** for intermediate frames to predict point positions
- Achieve **3-10× speedup** while maintaining tracking accuracy

---

## 📊 Problem Statement

**CoTracker3 Limitations:**
- Requires GPU inference **every frame** (expensive)
- ~5-10 FPS on standard hardware
- Too slow for real-time applications (robotics, AR, video editing)

**Our Solution:**
- **Sparse CoTracker3**: Run only on keyframes (every N=5-10 frames)
- **Kalman Interpolation**: Predict positions for intermediate frames
- **Adaptive Measurement Noise**: Higher noise for predictions vs measurements

---

## 🏗️ System Architecture

```
Frame 0 ──► CoTracker3 ──► Kalman Update ──► Output
Frame 1 ──► [Skip] ──────► Kalman Predict ─► Output
Frame 2 ──► [Skip] ──────► Kalman Predict ─► Output
...
Frame N ──► CoTracker3 ──► Kalman Update ──► Output
```

### State-Space Model

**State Vector:** `x = [x, y, vx, vy]`
- Position: (x, y)
- Velocity: (vx, vy)

**State Transition (Constant Velocity):**
```
x_{k+1} = x_k + vx_k * dt
y_{k+1} = y_k + vy_k * dt
vx_{k+1} = vx_k
vy_{k+1} = vy_k
```

**Measurement Model:**
```
z_k = [x_k, y_k] + noise
```

**Dual Measurement Strategy:**
1. **CoTracker3 (keyframes)**: Low noise (σ² = 0.1)
2. **Kalman Prediction (intermediate)**: Higher uncertainty

---

## 📁 Project Structure

```
kalmantrack/
├── synthetic_simulator.py       # Bouncing ball simulator (HW2 Part 1)
├── test_cotracker3_synthetic.py # Baseline comparison
├── kalman_hybrid.py             # NEW: Main hybrid tracker
├── scenarios.py                 # NEW: Test scenarios (HW2 Part 2)
├── metrics.py                   # NEW: Performance metrics (HW2 Part 3)
├── outputs/                     # Results and visualizations
└── klamantrack_hybrid_v1/       # Reference implementation
    └── scripts/
        ├── run_hybrid_enhanced.py
        └── kalman/cv_kalman.py
```

---

## 🧪 Test Scenarios (HW2 Part 2)

### Scenario 1: Nominal (Baseline)
- **Motion**: Simple bouncing ball
- **Process Noise**: σ_p = 0.05
- **Measurement Noise**: σ_m = 0.1
- **Update Frequency**: N = 5 frames
- **Purpose**: Establish baseline performance

### Scenario 2: High Noise (Challenging)
- **Motion**: Same bouncing
- **Process Noise**: σ_p = 0.2 (4× higher)
- **Measurement Noise**: σ_m = 0.5 (5× higher)
- **Update Frequency**: N = 10 frames
- **Purpose**: Test robustness to noise

### Scenario 3: Fast Dynamics (Stress Test)
- **Motion**: Higher drop, more elastic bounces
- **Process Noise**: σ_p = 0.1
- **Measurement Noise**: σ_m = 0.2
- **Update Frequency**: N = 15 frames
- **Purpose**: Test with rapid motion and long prediction gaps

---

## 📈 Performance Metrics (HW2 Part 3)

### Primary Metrics
1. **RMSE (Root Mean Square Error)**
   ```
   RMSE = sqrt(mean((x_true - x_pred)²))
   ```

2. **Mean Absolute Error**
   ```
   MAE = mean(|x_true - x_pred|)
   ```

3. **Maximum Error**
   ```
   Max Error = max(|x_true - x_pred|)
   ```

### Secondary Metrics
4. **Success Rate**: % of frames with valid tracking
5. **Processing Speed**: FPS (frames per second)
6. **Speedup Factor**: FPS_hybrid / FPS_cotracker3
7. **Accuracy Retention**: (1 - RMSE_hybrid/RMSE_cotracker3) × 100%

---

## 🔬 Baseline Methods (HW2 Part 4)

### Baseline 1: Pure CoTracker3 (Clairvoyant)
- **Description**: Run CoTracker3 every frame
- **Purpose**: Upper bound on accuracy
- **Expected**: Best accuracy, slowest speed

### Baseline 2: Kalman-Only (Prediction-Only)
- **Description**: Run CoTracker3 once, then pure Kalman prediction
- **Purpose**: Lower bound on accuracy
- **Expected**: Fast but poor accuracy (drift)

### Baseline 3: KalmanTrack Hybrid (Our Method)
- **Description**: CoTracker3 every N frames + Kalman interpolation
- **Purpose**: Balance accuracy and speed
- **Expected**: 3-10× speedup, <15% accuracy loss

### Baseline 4: Optical Flow Hybrid (Enhanced)
- **Description**: CoTracker3 + Optical Flow + Kalman
- **Purpose**: Best hybrid approach
- **Expected**: Better accuracy, moderate speedup

---

## 🎯 Expected Outcomes

### Hypothesis 1: Speedup vs Accuracy Tradeoff
- **N=5**: ~5× speedup, <10% accuracy loss
- **N=10**: ~8× speedup, 10-15% accuracy loss
- **N=15**: ~12× speedup, 15-25% accuracy loss

### Hypothesis 2: Noise Robustness
- Kalman filtering provides **better robustness** to measurement noise
- Adaptive noise parameters improve performance

### Hypothesis 3: Motion Complexity
- Simple linear motion: Kalman works well
- Complex bouncing: Requires more frequent CoTracker3 updates

---

## 📊 Current Results (Nominal Scenario)

| Method | RMSE (px) | Speed (FPS) | Speedup | Accuracy |
|--------|-----------|-------------|---------|----------|
| **CoTracker3** | 86.86 | ~5 | 1× | 100% |
| **KalmanTrack (DINO)** | 550.53 | ~20 | 4× | 84% loss ❌ |
| **Hybrid (Target)** | <100 | ~25-30 | 5-6× | >90% ✅ |

---

## 🚀 Next Steps

1. ✅ **Synthetic Simulator** - Complete
2. ✅ **CoTracker3 Baseline** - Complete
3. ⏳ **Implement Hybrid Tracker** - In Progress
4. ⏳ **Run All Scenarios** - Pending
5. ⏳ **Compute Metrics** - Pending
6. ⏳ **Write Report** - Pending

---

## 📝 HW2 Deliverables

- [x] Part 1: Synthetic data simulator (`synthetic_simulator.py`)
- [x] Part 2: Test scenarios (3 scenarios defined)
- [x] Part 3: Performance metrics (7 metrics defined)
- [x] Part 4: Baseline methods (4 baselines identified)
- [ ] Part 5: Real data (optional)
- [ ] Report (4-5 pages)
- [ ] Code documentation

---

## 📚 References

1. **CoTracker3**: Karaev et al., "CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos"
2. **Kalman Filtering**: Welch & Bishop, "An Introduction to the Kalman Filter"
3. **Hybrid Tracking**: Memory from previous work on optical flow interpolation
