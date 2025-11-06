#!/usr/bin/env python3
"""
Posterior Cramér-Rao Bound (PCRB) Computation for Point Tracking
EECE 7398: Bayesian Filtering - HW3

This module implements the recursive PCRB computation for the 4D point tracking
state-space model from HW2.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from synthetic_simulator import BouncingBallSimulator, SimulationConfig


@dataclass
class PCRBConfig:
    """Configuration for PCRB computation"""
    dt: float = 0.033  # Time step (30 FPS)
    gravity: float = 9.81  # m/s^2
    air_resistance: float = 0.01  # Drag coefficient
    process_noise_std: float = 0.05  # Process noise standard deviation
    measurement_noise_std: float = 0.1  # Measurement noise standard deviation
    elasticity: float = 0.8  # Coefficient of restitution
    floor_y: float = 0.0  # Floor level


class PCRBComputer:
    """
    Computes Posterior Cramér-Rao Bound for point tracking
    
    State vector: x = [x, y, vx, vy]^T
    Measurement: y = [x_obs, y_obs]^T
    """
    
    def __init__(self, config: PCRBConfig):
        self.config = config
        self.dt = config.dt
        self.g = config.gravity
        self.alpha = config.air_resistance
        self.sigma_p = config.process_noise_std
        self.sigma_m = config.measurement_noise_std
        
        # State dimension
        self.nx = 4  # [x, y, vx, vy]
        self.ny = 2  # [x_obs, y_obs]
        
    def get_state_transition_matrix(self) -> np.ndarray:
        """
        Get state transition matrix F for constant velocity model with drag
        
        Returns:
            F: 4x4 state transition matrix
        """
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1 - self.alpha * self.dt, 0],
            [0, 0, 0, 1 - self.alpha * self.dt]
        ])
        return F
    
    def get_process_noise_covariance(self) -> np.ndarray:
        """
        Get process noise covariance matrix Q
        
        Returns:
            Q: 4x4 process noise covariance matrix
        """
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
    
    def get_measurement_matrix(self) -> np.ndarray:
        """
        Get measurement matrix H (observe position only)
        
        Returns:
            H: 2x4 measurement matrix
        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return H
    
    def get_measurement_noise_covariance(self) -> np.ndarray:
        """
        Get measurement noise covariance matrix R
        
        Returns:
            R: 2x2 measurement noise covariance matrix
        """
        R = self.sigma_m**2 * np.eye(2)
        return R
    
    def compute_fisher_matrices(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Fisher Information Matrix components D11, D12, D21, D22
        
        Args:
            k: Time step
            
        Returns:
            D11, D12, D21, D22: Fisher Information Matrix components
        """
        F = self.get_state_transition_matrix()
        Q = self.get_process_noise_covariance()
        H = self.get_measurement_matrix()
        R = self.get_measurement_noise_covariance()
        
        # Compute matrix inverses
        Q_inv = np.linalg.inv(Q)
        R_inv = np.linalg.inv(R)
        
        # D11: Information from state transition (previous state)
        D11 = F.T @ Q_inv @ F
        
        # D12: Cross-information between consecutive states
        D12 = -F.T @ Q_inv
        
        # D21: Transpose of D12
        D21 = D12.T
        
        # D22: Information from state transition (current state) + measurement
        D22 = Q_inv + H.T @ R_inv @ H
        
        return D11, D12, D21, D22
    
    def compute_pcrb_recursive(self, T: int, J0: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute PCRB recursively over time
        
        Args:
            T: Number of time steps
            J0: Initial information matrix (if None, uses default)
            
        Returns:
            J_history: List of information matrices J_k
            pcrb_history: List of PCRB covariance matrices J_k^{-1}
        """
        # Initialize information matrix
        if J0 is None:
            # Weak prior: large uncertainty
            J0 = 1e-6 * np.eye(self.nx)
        
        J_history = [J0]
        pcrb_history = [np.linalg.inv(J0)]
        
        J_k = J0.copy()
        
        for k in range(1, T + 1):
            # Compute Fisher Information Matrix components
            D11, D12, D21, D22 = self.compute_fisher_matrices(k)
            
            # PCRB recursion: J_{k+1} = D22 - D21 * (D11 + J_k)^{-1} * D12
            try:
                temp_inv = np.linalg.inv(D11 + J_k)
                J_k_plus_1 = D22 - D21 @ temp_inv @ D12
                
                # Ensure positive definiteness (numerical stability)
                eigenvals = np.linalg.eigvals(J_k_plus_1)
                if np.any(eigenvals <= 0):
                    print(f"Warning: Non-positive definite J at k={k}, min eigenvalue: {np.min(eigenvals)}")
                    # Add small regularization
                    J_k_plus_1 += 1e-12 * np.eye(self.nx)
                
                J_k = J_k_plus_1
                pcrb_k = np.linalg.inv(J_k)
                
                J_history.append(J_k.copy())
                pcrb_history.append(pcrb_k.copy())
                
            except np.linalg.LinAlgError as e:
                print(f"Numerical error at k={k}: {e}")
                # Use previous values
                J_history.append(J_history[-1].copy())
                pcrb_history.append(pcrb_history[-1].copy())
        
        return J_history, pcrb_history
    
    def validate_information_matrix(self, J: np.ndarray, k: int) -> bool:
        """
        Validate that J is a proper information matrix
        
        Args:
            J: Information matrix to validate
            k: Time step (for error reporting)
            
        Returns:
            True if valid, False otherwise
        """
        # Check dimensions
        if J.shape != (self.nx, self.nx):
            print(f"Error at k={k}: Wrong dimensions {J.shape}, expected ({self.nx}, {self.nx})")
            return False
        
        # Check symmetry
        if not np.allclose(J, J.T, rtol=1e-10):
            print(f"Error at k={k}: Matrix not symmetric")
            return False
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(J)
        if np.any(eigenvals <= 0):
            print(f"Error at k={k}: Not positive definite, min eigenvalue: {np.min(eigenvals)}")
            return False
        
        return True
    
    def compute_rmse_bounds(self, pcrb_history: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract RMSE bounds for each state component
        
        Args:
            pcrb_history: List of PCRB covariance matrices
            
        Returns:
            Dictionary with RMSE bounds for each state component
        """
        T = len(pcrb_history)
        
        bounds = {
            'x': np.zeros(T),
            'y': np.zeros(T),
            'vx': np.zeros(T),
            'vy': np.zeros(T),
            'position': np.zeros(T),  # Combined position RMSE
            'velocity': np.zeros(T)   # Combined velocity RMSE
        }
        
        for k, pcrb in enumerate(pcrb_history):
            # Individual component bounds (diagonal elements)
            bounds['x'][k] = np.sqrt(pcrb[0, 0])
            bounds['y'][k] = np.sqrt(pcrb[1, 1])
            bounds['vx'][k] = np.sqrt(pcrb[2, 2])
            bounds['vy'][k] = np.sqrt(pcrb[3, 3])
            
            # Combined bounds
            position_cov = pcrb[:2, :2]
            velocity_cov = pcrb[2:, 2:]
            bounds['position'][k] = np.sqrt(np.trace(position_cov))
            bounds['velocity'][k] = np.sqrt(np.trace(velocity_cov))
        
        return bounds


def plot_pcrb_evolution(time: np.ndarray, bounds: Dict[str, np.ndarray], 
                       scenario_name: str = "nominal") -> None:
    """
    Plot PCRB evolution over time
    
    Args:
        time: Time vector
        bounds: Dictionary of RMSE bounds
        scenario_name: Name of the scenario
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position bounds
    ax = axes[0, 0]
    ax.plot(time, bounds['x'], 'b-', label='x position', linewidth=2)
    ax.plot(time, bounds['y'], 'r-', label='y position', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position RMSE Bound (m)')
    ax.set_title('PCRB: Position Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity bounds
    ax = axes[0, 1]
    ax.plot(time, bounds['vx'], 'b-', label='x velocity', linewidth=2)
    ax.plot(time, bounds['vy'], 'r-', label='y velocity', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity RMSE Bound (m/s)')
    ax.set_title('PCRB: Velocity Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined position bound
    ax = axes[1, 0]
    ax.plot(time, bounds['position'], 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Combined Position RMSE Bound (m)')
    ax.set_title('PCRB: Combined Position')
    ax.grid(True, alpha=0.3)
    
    # Combined velocity bound
    ax = axes[1, 1]
    ax.plot(time, bounds['velocity'], 'm-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Combined Velocity RMSE Bound (m/s)')
    ax.set_title('PCRB: Combined Velocity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = f'outputs/pcrb_evolution_{scenario_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCRB evolution plot saved: {save_path}")
    plt.show()


def run_pcrb_analysis(scenario_name: str = "nominal") -> Dict:
    """
    Run complete PCRB analysis for a given scenario
    
    Args:
        scenario_name: Name of the scenario to analyze
        
    Returns:
        Dictionary containing all results
    """
    print(f"\n{'='*60}")
    print(f"PCRB Analysis: {scenario_name.upper()} Scenario")
    print(f"{'='*60}")
    
    # Load scenario configuration
    from synthetic_simulator import create_test_scenarios
    scenarios = create_test_scenarios()
    
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    sim_config = scenarios[scenario_name]
    
    # Create PCRB configuration
    pcrb_config = PCRBConfig(
        dt=sim_config.dt,
        gravity=sim_config.gravity,
        air_resistance=sim_config.air_resistance,
        process_noise_std=sim_config.process_noise_std,
        measurement_noise_std=sim_config.measurement_noise_std,
        elasticity=sim_config.elasticity,
        floor_y=sim_config.floor_y
    )
    
    # Initialize PCRB computer
    pcrb_computer = PCRBComputer(pcrb_config)
    
    # Compute PCRB over simulation duration
    T = sim_config.n_steps
    print(f"Computing PCRB for {T} time steps...")
    
    J_history, pcrb_history = pcrb_computer.compute_pcrb_recursive(T)
    
    # Validate all information matrices
    print("Validating information matrices...")
    all_valid = True
    for k, J in enumerate(J_history):
        if not pcrb_computer.validate_information_matrix(J, k):
            all_valid = False
    
    if all_valid:
        print("✓ All information matrices are valid")
    else:
        print("✗ Some information matrices failed validation")
    
    # Extract RMSE bounds
    bounds = pcrb_computer.compute_rmse_bounds(pcrb_history)
    
    # Create time vector
    time = np.arange(T + 1) * sim_config.dt
    
    # Plot results
    plot_pcrb_evolution(time, bounds, scenario_name)
    
    # Print summary statistics
    print(f"\nPCRB Summary Statistics:")
    print(f"  Final position bound: {bounds['position'][-1]:.4f} m")
    print(f"  Final velocity bound: {bounds['velocity'][-1]:.4f} m/s")
    print(f"  Max position bound: {np.max(bounds['position']):.4f} m")
    print(f"  Max velocity bound: {np.max(bounds['velocity']):.4f} m/s")
    
    return {
        'scenario_name': scenario_name,
        'config': pcrb_config,
        'J_history': J_history,
        'pcrb_history': pcrb_history,
        'bounds': bounds,
        'time': time,
        'valid': all_valid
    }


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("PCRB Computation for Point Tracking")
    print("=" * 50)
    
    # Run analysis for all scenarios
    scenarios = ['nominal', 'high_noise', 'fast_dynamics']
    results = {}
    
    for scenario in scenarios:
        try:
            results[scenario] = run_pcrb_analysis(scenario)
        except Exception as e:
            print(f"Error in scenario {scenario}: {e}")
    
    print(f"\n{'='*60}")
    print("PCRB Analysis Complete!")
    print(f"{'='*60}")
    print("Generated files:")
    for scenario in scenarios:
        if scenario in results:
            print(f"  - outputs/pcrb_evolution_{scenario}.png")
