#!/usr/bin/env python3
"""
Synthetic Data Simulator for Point Tracking with Kalman Filtering
EECE 7398: Bayesian Filtering - HW2

Simulates a bouncing ball with configurable physics and noise parameters.
Generates ground truth trajectories and noisy measurements for filter evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation"""
    # Physics parameters
    gravity: float = 9.81  # m/s^2
    elasticity: float = 0.8  # Coefficient of restitution (0-1)
    air_resistance: float = 0.01  # Air drag coefficient
    
    # Initial conditions
    initial_x: float = 0.0  # meters
    initial_y: float = 5.0  # meters (drop height)
    initial_vx: float = 2.0  # m/s (horizontal velocity)
    initial_vy: float = 0.0  # m/s (vertical velocity)
    
    # Simulation parameters
    dt: float = 0.033  # Time step (30 FPS)
    duration: float = 10.0  # Total simulation time (seconds)
    
    # Noise parameters
    process_noise_std: float = 0.1  # Process noise standard deviation
    measurement_noise_std: float = 0.2  # Measurement noise standard deviation
    
    # Environment
    floor_y: float = 0.0  # Floor position
    bounds_x: Tuple[float, float] = (0.0, 10.0)  # X boundaries
    
    def __post_init__(self):
        self.n_steps = int(self.duration / self.dt)


class BouncingBallSimulator:
    """
    Simulates a bouncing ball with realistic physics.
    
    State vector: [x, y, vx, vy]
    - x, y: position (meters)
    - vx, vy: velocity (m/s)
    
    Measurement vector: [x_obs, y_obs]
    - Noisy observations of position
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulator with configuration
        
        Args:
            config: SimulationConfig object with all parameters
        """
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset simulator to initial conditions"""
        self.time = 0.0
        self.state = np.array([
            self.config.initial_x,
            self.config.initial_y,
            self.config.initial_vx,
            self.config.initial_vy
        ])
        self.trajectory = []
        self.measurements = []
        self.bounce_times = []
        
    def state_transition(self, state: np.ndarray, dt: float, 
                        add_noise: bool = True) -> np.ndarray:
        """
        Propagate state forward by dt using physics model
        
        Args:
            state: Current state [x, y, vx, vy]
            dt: Time step
            add_noise: Whether to add process noise
            
        Returns:
            Next state [x, y, vx, vy]
        """
        x, y, vx, vy = state
        
        # Apply physics
        # Acceleration due to gravity and air resistance
        ax = -self.config.air_resistance * vx
        ay = -self.config.gravity - self.config.air_resistance * vy
        
        # Update velocity
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        
        # Update position
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        
        # Handle floor collision (bounce)
        if y_new <= self.config.floor_y:
            y_new = self.config.floor_y
            vy_new = -vy_new * self.config.elasticity  # Reverse and dampen
            self.bounce_times.append(self.time)
            
        # Handle wall collisions (optional - bounce off walls)
        if x_new <= self.config.bounds_x[0] or x_new >= self.config.bounds_x[1]:
            vx_new = -vx_new * self.config.elasticity
            x_new = np.clip(x_new, self.config.bounds_x[0], self.config.bounds_x[1])
        
        # Add process noise (uncertainty in physics model)
        if add_noise:
            process_noise = np.random.randn(4) * self.config.process_noise_std
            # Only add noise to velocity (position follows from velocity)
            process_noise[:2] = 0  # No direct position noise
            next_state = np.array([x_new, y_new, vx_new, vy_new]) + process_noise
        else:
            next_state = np.array([x_new, y_new, vx_new, vy_new])
            
        return next_state
    
    def generate_measurement(self, state: np.ndarray, 
                           add_noise: bool = True) -> np.ndarray:
        """
        Generate noisy measurement from true state
        
        Args:
            state: True state [x, y, vx, vy]
            add_noise: Whether to add measurement noise
            
        Returns:
            Measurement [x_obs, y_obs]
        """
        # Measurement model: observe position only
        true_measurement = state[:2]  # [x, y]
        
        if add_noise:
            measurement_noise = np.random.randn(2) * self.config.measurement_noise_std
            noisy_measurement = true_measurement + measurement_noise
        else:
            noisy_measurement = true_measurement
            
        return noisy_measurement
    
    def simulate(self, add_process_noise: bool = True,
                add_measurement_noise: bool = True) -> Dict:
        """
        Run complete simulation
        
        Args:
            add_process_noise: Add process noise to state evolution
            add_measurement_noise: Add noise to measurements
            
        Returns:
            Dictionary containing:
                - true_states: Ground truth state trajectory [n_steps, 4]
                - measurements: Noisy observations [n_steps, 2]
                - time: Time vector [n_steps]
                - config: Simulation configuration
        """
        self.reset()
        
        true_states = []
        measurements = []
        time_vec = []
        
        for step in range(self.config.n_steps):
            # Store current state
            true_states.append(self.state.copy())
            
            # Generate measurement
            measurement = self.generate_measurement(self.state, add_measurement_noise)
            measurements.append(measurement)
            
            # Store time
            time_vec.append(self.time)
            
            # Propagate state
            self.state = self.state_transition(self.state, self.config.dt, add_process_noise)
            self.time += self.config.dt
        
        return {
            'true_states': np.array(true_states),
            'measurements': np.array(measurements),
            'time': np.array(time_vec),
            'bounce_times': self.bounce_times,
            'config': asdict(self.config)
        }
    
    def run_monte_carlo(self, n_trials: int = 100,
                       add_process_noise: bool = True,
                       add_measurement_noise: bool = True) -> List[Dict]:
        """
        Run multiple simulation trials for Monte Carlo analysis
        
        Args:
            n_trials: Number of simulation runs
            add_process_noise: Add process noise
            add_measurement_noise: Add measurement noise
            
        Returns:
            List of simulation results
        """
        results = []
        for trial in range(n_trials):
            result = self.simulate(add_process_noise, add_measurement_noise)
            result['trial'] = trial
            results.append(result)
        return results
    
    def visualize_trajectory(self, simulation_result: Dict, 
                           save_path: Optional[str] = None):
        """
        Visualize simulation results
        
        Args:
            simulation_result: Output from simulate()
            save_path: Optional path to save figure
        """
        true_states = simulation_result['true_states']
        measurements = simulation_result['measurements']
        time = simulation_result['time']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 2D Trajectory
        ax = axes[0, 0]
        ax.plot(true_states[:, 0], true_states[:, 1], 'b-', 
                label='True Trajectory', linewidth=2)
        ax.scatter(measurements[:, 0], measurements[:, 1], 
                  c='r', s=10, alpha=0.3, label='Measurements')
        ax.axhline(y=self.config.floor_y, color='k', linestyle='--', 
                  label='Floor')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('2D Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # X position over time
        ax = axes[0, 1]
        ax.plot(time, true_states[:, 0], 'b-', label='True X', linewidth=2)
        ax.scatter(time, measurements[:, 0], c='r', s=10, alpha=0.3, 
                  label='Measured X')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('X Position (m)')
        ax.set_title('X Position vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Y position over time
        ax = axes[1, 0]
        ax.plot(time, true_states[:, 1], 'b-', label='True Y', linewidth=2)
        ax.scatter(time, measurements[:, 1], c='r', s=10, alpha=0.3, 
                  label='Measured Y')
        ax.axhline(y=self.config.floor_y, color='k', linestyle='--')
        # Mark bounces
        for bounce_time in simulation_result['bounce_times']:
            ax.axvline(x=bounce_time, color='orange', alpha=0.3, linestyle=':')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Y Position vs Time (vertical lines = bounces)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity magnitude
        ax = axes[1, 1]
        velocity_mag = np.sqrt(true_states[:, 2]**2 + true_states[:, 3]**2)
        ax.plot(time, velocity_mag, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Magnitude (m/s)')
        ax.set_title('Velocity Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def validate_noise_statistics(self, n_trials: int = 1000):
        """
        Validate that noise realizations match theoretical distributions
        
        Args:
            n_trials: Number of trials for validation
        """
        # Generate noise samples
        process_noise_samples = []
        measurement_noise_samples = []
        
        for _ in range(n_trials):
            # Process noise
            pn = np.random.randn(4) * self.config.process_noise_std
            process_noise_samples.append(pn)
            
            # Measurement noise
            mn = np.random.randn(2) * self.config.measurement_noise_std
            measurement_noise_samples.append(mn)
        
        process_noise_samples = np.array(process_noise_samples)
        measurement_noise_samples = np.array(measurement_noise_samples)
        
        # Plot histograms and statistics
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Process noise
        for i in range(4):
            ax = axes[0, i] if i < 3 else axes[1, 0]
            ax.hist(process_noise_samples[:, i], bins=50, density=True, 
                   alpha=0.7, edgecolor='black')
            
            # Theoretical Gaussian
            x = np.linspace(process_noise_samples[:, i].min(), 
                          process_noise_samples[:, i].max(), 100)
            theoretical = (1/np.sqrt(2*np.pi*self.config.process_noise_std**2)) * \
                         np.exp(-x**2/(2*self.config.process_noise_std**2))
            ax.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical')
            
            state_names = ['x', 'y', 'vx', 'vy']
            ax.set_title(f'Process Noise: {state_names[i]}')
            ax.set_xlabel('Noise Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Measurement noise
        for i in range(2):
            ax = axes[1, i+1]
            ax.hist(measurement_noise_samples[:, i], bins=50, density=True,
                   alpha=0.7, edgecolor='black')
            
            x = np.linspace(measurement_noise_samples[:, i].min(),
                          measurement_noise_samples[:, i].max(), 100)
            theoretical = (1/np.sqrt(2*np.pi*self.config.measurement_noise_std**2)) * \
                         np.exp(-x**2/(2*self.config.measurement_noise_std**2))
            ax.plot(x, theoretical, 'r-', linewidth=2, label='Theoretical')
            
            meas_names = ['x', 'y']
            ax.set_title(f'Measurement Noise: {meas_names[i]}')
            ax.set_xlabel('Noise Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/noise_validation.png', dpi=300, bbox_inches='tight')
        print("Noise validation plot saved to outputs/noise_validation.png")
        plt.show()
        
        # Print statistics
        print("\n=== Noise Statistics Validation ===")
        print(f"\nProcess Noise (theoretical std: {self.config.process_noise_std:.3f})")
        print(f"  Empirical std: {np.std(process_noise_samples, axis=0)}")
        print(f"  Empirical mean: {np.mean(process_noise_samples, axis=0)}")
        
        print(f"\nMeasurement Noise (theoretical std: {self.config.measurement_noise_std:.3f})")
        print(f"  Empirical std: {np.std(measurement_noise_samples, axis=0)}")
        print(f"  Empirical mean: {np.mean(measurement_noise_samples, axis=0)}")


def create_test_scenarios() -> Dict[str, SimulationConfig]:
    """
    Create predefined test scenarios for evaluation
    
    Returns:
        Dictionary of scenario name -> configuration
    """
    scenarios = {}
    
    # Scenario 1: Nominal - Simple bouncing with low noise
    scenarios['nominal'] = SimulationConfig(
        gravity=9.81,
        elasticity=0.8,
        air_resistance=0.01,
        initial_x=2.0,
        initial_y=5.0,
        initial_vx=1.5,
        initial_vy=0.0,
        dt=0.033,
        duration=10.0,
        process_noise_std=0.05,
        measurement_noise_std=0.1,
        floor_y=0.0,
        bounds_x=(0.0, 10.0)
    )
    
    # Scenario 2: High Noise - Challenging measurement conditions
    scenarios['high_noise'] = SimulationConfig(
        gravity=9.81,
        elasticity=0.8,
        air_resistance=0.01,
        initial_x=2.0,
        initial_y=5.0,
        initial_vx=1.5,
        initial_vy=0.0,
        dt=0.033,
        duration=10.0,
        process_noise_std=0.2,  # 4x higher
        measurement_noise_std=0.5,  # 5x higher
        floor_y=0.0,
        bounds_x=(0.0, 10.0)
    )
    
    # Scenario 3: Fast Dynamics - Higher drop, more elastic
    scenarios['fast_dynamics'] = SimulationConfig(
        gravity=9.81,
        elasticity=0.95,  # More elastic bounces
        air_resistance=0.005,  # Less drag
        initial_x=2.0,
        initial_y=10.0,  # Higher drop
        initial_vx=3.0,  # Faster horizontal
        initial_vy=0.0,
        dt=0.033,
        duration=15.0,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
        floor_y=0.0,
        bounds_x=(0.0, 15.0)
    )
    
    return scenarios


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("=== Bouncing Ball Simulator - HW2 ===\n")
    
    # Test nominal scenario
    print("Running nominal scenario...")
    config = create_test_scenarios()['nominal']
    simulator = BouncingBallSimulator(config)
    
    # Run single simulation
    result = simulator.simulate()
    
    print(f"Simulation complete:")
    print(f"  Duration: {config.duration}s")
    print(f"  Time steps: {config.n_steps}")
    print(f"  Number of bounces: {len(result['bounce_times'])}")
    print(f"  Bounce times: {[f'{t:.2f}s' for t in result['bounce_times'][:5]]}...")
    
    # Visualize
    simulator.visualize_trajectory(result, 'outputs/nominal_trajectory.png')
    
    # Validate noise
    print("\nValidating noise distributions...")
    simulator.validate_noise_statistics(n_trials=1000)
    
    # Save data
    print("\nSaving simulation data...")
    np.savez('outputs/nominal_simulation.npz',
             true_states=result['true_states'],
             measurements=result['measurements'],
             time=result['time'])
    print("Data saved to outputs/nominal_simulation.npz")
