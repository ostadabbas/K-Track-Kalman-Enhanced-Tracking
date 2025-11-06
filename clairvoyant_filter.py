#!/usr/bin/env python3
"""
Clairvoyant Kalman Filter for Point Tracking
EECE 7398: Bayesian Filtering - HW3

This module implements a clairvoyant (oracle) Kalman filter that has perfect
knowledge of all system parameters for baseline comparison with PCRB.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from synthetic_simulator import BouncingBallSimulator, SimulationConfig
from pcrb_computation import PCRBConfig, PCRBComputer


@dataclass
class FilterState:
    """State of the Kalman filter at one time step"""
    x_hat: np.ndarray  # State estimate [4x1]
    P: np.ndarray      # Covariance matrix [4x4]
    x_pred: np.ndarray # Predicted state [4x1]
    P_pred: np.ndarray # Predicted covariance [4x4]
    innovation: np.ndarray  # Innovation [2x1]
    S: np.ndarray      # Innovation covariance [2x2]
    K: np.ndarray      # Kalman gain [4x2]


class ClairvoyantKalmanFilter:
    """
    Clairvoyant Kalman Filter with perfect knowledge of system parameters
    
    This filter knows:
    - Exact process and measurement noise covariances
    - Perfect state transition model including gravity and air resistance
    - Collision detection and elastic collision handling
    """
    
    def __init__(self, config: PCRBConfig):
        self.config = config
        self.dt = config.dt
        self.g = config.gravity
        self.alpha = config.air_resistance
        self.sigma_p = config.process_noise_std
        self.sigma_m = config.measurement_noise_std
        self.elasticity = config.elasticity
        self.floor_y = config.floor_y
        
        # State and measurement dimensions
        self.nx = 4  # [x, y, vx, vy]
        self.ny = 2  # [x_obs, y_obs]
        
        # System matrices
        self.F = self._get_state_transition_matrix()
        self.Q = self._get_process_noise_covariance()
        self.H = self._get_measurement_matrix()
        self.R = self._get_measurement_noise_covariance()
        self.b = self._get_control_input()  # Gravity effect
        
        # Filter state history
        self.state_history: List[FilterState] = []
        
    def _get_state_transition_matrix(self) -> np.ndarray:
        """Get state transition matrix F"""
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1 - self.alpha * self.dt, 0],
            [0, 0, 0, 1 - self.alpha * self.dt]
        ])
        return F
    
    def _get_process_noise_covariance(self) -> np.ndarray:
        """Get process noise covariance matrix Q"""
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        Q = self.sigma_p**2 * np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2]
        ])
        return Q
    
    def _get_measurement_matrix(self) -> np.ndarray:
        """Get measurement matrix H (observe position only)"""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return H
    
    def _get_measurement_noise_covariance(self) -> np.ndarray:
        """Get measurement noise covariance matrix R"""
        R = self.sigma_m**2 * np.eye(2)
        return R
    
    def _get_control_input(self) -> np.ndarray:
        """Get control input vector (gravity effect)"""
        b = np.array([0, 0, 0, -self.g * self.dt])
        return b
    
    def _detect_collision(self, x_pred: np.ndarray) -> bool:
        """
        Detect if collision with floor occurs
        
        Args:
            x_pred: Predicted state [x, y, vx, vy]
            
        Returns:
            True if collision detected
        """
        y_pos = x_pred[1]
        y_vel = x_pred[3]
        
        # Collision if below floor and moving downward
        return y_pos <= self.floor_y and y_vel < 0
    
    def _handle_collision(self, x_pred: np.ndarray, P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle elastic collision with floor
        
        Args:
            x_pred: Predicted state before collision
            P_pred: Predicted covariance before collision
            
        Returns:
            x_post: State after collision
            P_post: Covariance after collision
        """
        x_post = x_pred.copy()
        
        # Set position to floor level
        x_post[1] = self.floor_y
        
        # Reverse and dampen y-velocity
        x_post[3] = -self.elasticity * x_pred[3]
        
        # Update covariance - collision affects y-velocity uncertainty
        # The collision is deterministic, so we can reduce velocity uncertainty
        P_post = P_pred.copy()
        
        # Collision transformation matrix
        T = np.eye(4)
        T[3, 3] = -self.elasticity  # Velocity reversal and damping
        
        # Transform covariance: P_post = T * P_pred * T^T
        P_post = T @ P_pred @ T.T
        
        # Position is now known exactly at floor level
        P_post[1, :] = 0
        P_post[:, 1] = 0
        P_post[1, 1] = 1e-12  # Small numerical value
        
        return x_post, P_post
    
    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """
        Initialize filter with initial state and covariance
        
        Args:
            x0: Initial state estimate [4x1]
            P0: Initial covariance matrix [4x4]
        """
        # Create initial filter state
        initial_state = FilterState(
            x_hat=x0.copy(),
            P=P0.copy(),
            x_pred=x0.copy(),
            P_pred=P0.copy(),
            innovation=np.zeros(2),
            S=np.eye(2),
            K=np.zeros((4, 2))
        )
        
        self.state_history = [initial_state]
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter prediction step
        
        Returns:
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        if not self.state_history:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        # Get previous state
        prev_state = self.state_history[-1]
        x_prev = prev_state.x_hat
        P_prev = prev_state.P
        
        # Prediction equations
        x_pred = self.F @ x_prev + self.b
        P_pred = self.F @ P_prev @ self.F.T + self.Q
        
        # Handle collision if detected
        if self._detect_collision(x_pred):
            x_pred, P_pred = self._handle_collision(x_pred, P_pred)
        
        return x_pred, P_pred
    
    def update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter update step
        
        Args:
            y: Measurement [2x1]
            x_pred: Predicted state [4x1]
            P_pred: Predicted covariance [4x4]
            
        Returns:
            x_hat: Updated state estimate
            P: Updated covariance
        """
        # Innovation
        innovation = y - self.H @ x_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # State update
        x_hat = x_pred + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.nx) - K @ self.H
        P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        
        return x_hat, P, innovation, S, K
    
    def step(self, y: np.ndarray) -> FilterState:
        """
        Perform one complete filter step (predict + update)
        
        Args:
            y: Measurement [2x1]
            
        Returns:
            FilterState object with all intermediate results
        """
        # Prediction step
        x_pred, P_pred = self.predict()
        
        # Update step
        x_hat, P, innovation, S, K = self.update(y, x_pred, P_pred)
        
        # Create filter state
        state = FilterState(
            x_hat=x_hat,
            P=P,
            x_pred=x_pred,
            P_pred=P_pred,
            innovation=innovation,
            S=S,
            K=K
        )
        
        # Store in history
        self.state_history.append(state)
        
        return state
    
    def get_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all state estimates and covariances
        
        Returns:
            estimates: Array of state estimates [T+1, 4]
            covariances: Array of covariances [T+1, 4, 4]
        """
        T = len(self.state_history)
        estimates = np.zeros((T, 4))
        covariances = np.zeros((T, 4, 4))
        
        for k, state in enumerate(self.state_history):
            estimates[k] = state.x_hat
            covariances[k] = state.P
        
        return estimates, covariances


def run_monte_carlo_filter(config: PCRBConfig, sim_config: SimulationConfig, 
                          n_trials: int = 1000) -> Dict:
    """
    Run Monte Carlo evaluation of clairvoyant Kalman filter
    
    Args:
        config: PCRB configuration
        sim_config: Simulation configuration
        n_trials: Number of Monte Carlo trials
        
    Returns:
        Dictionary with Monte Carlo results
    """
    print(f"Running {n_trials} Monte Carlo trials...")
    
    # Initialize simulator
    simulator = BouncingBallSimulator(sim_config)
    T = sim_config.n_steps
    
    # Storage for results
    all_estimates = np.zeros((n_trials, T + 1, 4))
    all_true_states = np.zeros((n_trials, T + 1, 4))
    all_covariances = np.zeros((n_trials, T + 1, 4, 4))
    
    for trial in range(n_trials):
        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")
        
        # Generate synthetic data
        simulator.reset()
        result = simulator.simulate(
            add_process_noise=True,
            add_measurement_noise=True
        )
        
        true_states = result['true_states']
        measurements = result['measurements']
        
        # Initialize filter
        kf = ClairvoyantKalmanFilter(config)
        
        # Initial state: use true initial state with some uncertainty
        x0 = true_states[0].copy()
        P0 = np.diag([0.1, 0.1, 0.5, 0.5])  # Initial uncertainty
        kf.initialize(x0, P0)
        
        # Run filter
        for k in range(T):
            y = measurements[k + 1]  # measurements[0] is initial, skip it
            kf.step(y)
        
        # Store results
        estimates, covariances = kf.get_estimates()
        all_estimates[trial] = estimates
        all_true_states[trial] = true_states
        all_covariances[trial] = covariances
    
    # Compute empirical MSE
    errors = all_estimates - all_true_states
    empirical_mse = np.zeros((T + 1, 4, 4))
    
    for k in range(T + 1):
        error_k = errors[:, k, :]  # [n_trials, 4]
        empirical_mse[k] = np.cov(error_k.T)
    
    # Compute average covariance from filter
    avg_covariance = np.mean(all_covariances, axis=0)
    
    return {
        'n_trials': n_trials,
        'estimates': all_estimates,
        'true_states': all_true_states,
        'covariances': all_covariances,
        'empirical_mse': empirical_mse,
        'avg_covariance': avg_covariance,
        'time': np.arange(T + 1) * sim_config.dt
    }


def compute_efficiency_ratio(pcrb_results: Dict, mc_results: Dict) -> Dict:
    """
    Compute efficiency ratio between PCRB and empirical performance
    
    Args:
        pcrb_results: Results from PCRB computation
        mc_results: Results from Monte Carlo filter evaluation
        
    Returns:
        Dictionary with efficiency analysis
    """
    pcrb_history = pcrb_results['pcrb_history']
    empirical_mse = mc_results['empirical_mse']
    
    T = len(pcrb_history)
    efficiency_ratio = np.zeros(T)
    
    for k in range(T):
        pcrb_trace = np.trace(pcrb_history[k])
        mse_trace = np.trace(empirical_mse[k])
        
        if mse_trace > 0:
            efficiency_ratio[k] = pcrb_trace / mse_trace
        else:
            efficiency_ratio[k] = 1.0
    
    return {
        'efficiency_ratio': efficiency_ratio,
        'mean_efficiency': np.mean(efficiency_ratio),
        'final_efficiency': efficiency_ratio[-1]
    }


def plot_filter_vs_pcrb(pcrb_results: Dict, mc_results: Dict, 
                       scenario_name: str = "nominal") -> None:
    """
    Plot filter performance vs PCRB bounds
    
    Args:
        pcrb_results: PCRB computation results
        mc_results: Monte Carlo filter results
        scenario_name: Name of scenario
    """
    time = pcrb_results['time']
    bounds = pcrb_results['bounds']
    empirical_mse = mc_results['empirical_mse']
    
    # Compute empirical RMSE
    empirical_rmse = {
        'x': np.sqrt(empirical_mse[:, 0, 0]),
        'y': np.sqrt(empirical_mse[:, 1, 1]),
        'vx': np.sqrt(empirical_mse[:, 2, 2]),
        'vy': np.sqrt(empirical_mse[:, 3, 3])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # X position
    ax = axes[0, 0]
    ax.plot(time, bounds['x'], 'b-', label='PCRB Bound', linewidth=2)
    ax.plot(time, empirical_rmse['x'], 'r--', label='Filter RMSE', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position RMSE (m)')
    ax.set_title('X Position: PCRB vs Filter Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y position
    ax = axes[0, 1]
    ax.plot(time, bounds['y'], 'b-', label='PCRB Bound', linewidth=2)
    ax.plot(time, empirical_rmse['y'], 'r--', label='Filter RMSE', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position RMSE (m)')
    ax.set_title('Y Position: PCRB vs Filter Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X velocity
    ax = axes[1, 0]
    ax.plot(time, bounds['vx'], 'b-', label='PCRB Bound', linewidth=2)
    ax.plot(time, empirical_rmse['vx'], 'r--', label='Filter RMSE', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Velocity RMSE (m/s)')
    ax.set_title('X Velocity: PCRB vs Filter Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y velocity
    ax = axes[1, 1]
    ax.plot(time, bounds['vy'], 'b-', label='PCRB Bound', linewidth=2)
    ax.plot(time, empirical_rmse['vy'], 'r--', label='Filter RMSE', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Velocity RMSE (m/s)')
    ax.set_title('Y Velocity: PCRB vs Filter Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = f'outputs/pcrb_vs_filter_{scenario_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCRB vs Filter plot saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Clairvoyant Kalman Filter Evaluation")
    print("=" * 50)
    
    # Test with nominal scenario
    from synthetic_simulator import create_test_scenarios
    from pcrb_computation import run_pcrb_analysis
    
    scenario_name = "nominal"
    scenarios = create_test_scenarios()
    sim_config = scenarios[scenario_name]
    
    # Create PCRB config
    pcrb_config = PCRBConfig(
        dt=sim_config.dt,
        gravity=sim_config.gravity,
        air_resistance=sim_config.air_resistance,
        process_noise_std=sim_config.process_noise_std,
        measurement_noise_std=sim_config.measurement_noise_std,
        elasticity=sim_config.elasticity,
        floor_y=sim_config.floor_y
    )
    
    # Run PCRB analysis
    print("Computing PCRB...")
    pcrb_results = run_pcrb_analysis(scenario_name)
    
    # Run Monte Carlo filter evaluation
    print("Running Monte Carlo filter evaluation...")
    mc_results = run_monte_carlo_filter(pcrb_config, sim_config, n_trials=100)
    
    # Compute efficiency ratio
    efficiency = compute_efficiency_ratio(pcrb_results, mc_results)
    
    print(f"\nEfficiency Analysis:")
    print(f"  Mean efficiency ratio: {efficiency['mean_efficiency']:.3f}")
    print(f"  Final efficiency ratio: {efficiency['final_efficiency']:.3f}")
    
    # Plot comparison
    plot_filter_vs_pcrb(pcrb_results, mc_results, scenario_name)
