#!/usr/bin/env python3
"""
Generate All HW3 Figures with Corrected Analysis
EECE 7398: Bayesian Filtering - HW3

This script generates all figures needed for the HW3 report with corrected
PCRB analysis and realistic parameter settings.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from pcrb_fixed import PCRBComputerFixed, PCRBConfig
from synthetic_simulator import create_test_scenarios


def generate_figure_1_pcrb_derivation():
    """
    Figure 1: PCRB Derivation Illustration
    Shows the Fisher Information Matrix components
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create sample matrices for illustration
    dt = 0.033
    
    # F matrix
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 0.99, 0],
        [0, 0, 0, 0.99]
    ])
    
    # Q matrix (simplified)
    Q = 0.1**2 * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ])
    
    # H matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # R matrix
    R = 0.1**2 * np.eye(2)
    
    # Plot F matrix
    ax = axes[0, 0]
    im = ax.imshow(F, cmap='RdBu', aspect='equal')
    ax.set_title('State Transition Matrix F', fontsize=14, fontweight='bold')
    ax.set_xlabel('State Component')
    ax.set_ylabel('State Component')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{F[i, j]:.2f}', ha="center", va="center", 
                          color="white" if abs(F[i, j]) > 0.5 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Plot Q matrix
    ax = axes[0, 1]
    im = ax.imshow(Q, cmap='Reds', aspect='equal')
    ax.set_title('Process Noise Covariance Q', fontsize=14, fontweight='bold')
    ax.set_xlabel('State Component')
    ax.set_ylabel('State Component')
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Plot H matrix
    ax = axes[1, 0]
    im = ax.imshow(H, cmap='Blues', aspect='equal')
    ax.set_title('Measurement Matrix H', fontsize=14, fontweight='bold')
    ax.set_xlabel('State Component')
    ax.set_ylabel('Measurement Component')
    
    # Add text annotations
    for i in range(2):
        for j in range(4):
            text = ax.text(j, i, f'{H[i, j]:.0f}', ha="center", va="center", 
                          color="white" if H[i, j] > 0.5 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Plot R matrix
    ax = axes[1, 1]
    im = ax.imshow(R, cmap='Greens', aspect='equal')
    ax.set_title('Measurement Noise Covariance R', fontsize=14, fontweight='bold')
    ax.set_xlabel('Measurement Component')
    ax.set_ylabel('Measurement Component')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{R[i, j]:.4f}', ha="center", va="center", 
                          color="white" if R[i, j] > 0.01 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    
    save_path = 'outputs/figure1_pcrb_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved: {save_path}")
    plt.show()


def generate_figure_2_pcrb_evolution():
    """
    Figure 2: PCRB Evolution Over Time (Corrected)
    Shows realistic PCRB bounds for all scenarios
    """
    scenarios = ['nominal', 'high_noise', 'fast_dynamics']
    colors = ['blue', 'red', 'green']
    
    # Get realistic configurations
    realistic_configs = {
        'nominal': PCRBConfig(process_noise_std=0.2, measurement_noise_std=0.1),
        'high_noise': PCRBConfig(process_noise_std=0.4, measurement_noise_std=0.3),
        'fast_dynamics': PCRBConfig(process_noise_std=0.3, measurement_noise_std=0.15)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    results = {}
    
    for i, scenario in enumerate(scenarios):
        config = realistic_configs[scenario]
        pcrb_computer = PCRBComputerFixed(config)
        
        # Compute PCRB
        T = 200  # Reasonable length
        J_history = pcrb_computer.compute_pcrb_sequence(T)
        bounds = pcrb_computer.extract_bounds(J_history)
        time = np.arange(len(J_history)) * config.dt
        
        results[scenario] = {'bounds': bounds, 'time': time}
        
        # Position bounds
        ax = axes[0, 0]
        ax.plot(time, bounds['position'], color=colors[i], 
               label=f'{scenario.title()}', linewidth=2)
        
        # Velocity bounds  
        ax = axes[0, 1]
        ax.plot(time, bounds['velocity'], color=colors[i], 
               label=f'{scenario.title()}', linewidth=2)
        
        # X and Y position bounds
        if i == 0:  # Only for nominal scenario
            ax = axes[1, 0]
            ax.plot(time, bounds['x'], 'b-', label='X position', linewidth=2)
            ax.plot(time, bounds['y'], 'r-', label='Y position', linewidth=2)
            
            ax = axes[1, 1]
            ax.plot(time, bounds['vx'], 'b-', label='X velocity', linewidth=2)
            ax.plot(time, bounds['vy'], 'r-', label='Y velocity', linewidth=2)
    
    # Finalize subplots
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position RMSE Bound (m)')
    axes[0, 0].set_title('PCRB: Position Bounds by Scenario')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity RMSE Bound (m/s)')
    axes[0, 1].set_title('PCRB: Velocity Bounds by Scenario')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position RMSE Bound (m)')
    axes[1, 0].set_title('PCRB: Position Components (Nominal)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity RMSE Bound (m/s)')
    axes[1, 1].set_title('PCRB: Velocity Components (Nominal)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = 'outputs/figure2_pcrb_evolution_corrected.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved: {save_path}")
    plt.show()
    
    return results


def generate_figure_3_efficiency_analysis():
    """
    Figure 3: Corrected Efficiency Analysis
    Shows realistic efficiency ratios and performance comparison
    """
    # HW2 results (converted to meters)
    pixels_per_meter = 50
    
    actual_performance = {
        'CoTracker3\n(baseline)': 86.86 / pixels_per_meter,
        'Hybrid\nN=2': 25.01 / pixels_per_meter,
        'Hybrid\nN=3': 24.30 / pixels_per_meter,
        'Hybrid\nN=5': 25.05 / pixels_per_meter,
        'Hybrid\nN=10': 33.36 / pixels_per_meter,
    }
    
    # Realistic PCRB bounds for different noise levels
    noise_configs = {
        'Optimistic': PCRBConfig(process_noise_std=0.1, measurement_noise_std=0.05),
        'Realistic': PCRBConfig(process_noise_std=0.2, measurement_noise_std=0.1),
        'Conservative': PCRBConfig(process_noise_std=0.3, measurement_noise_std=0.15),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: PCRB bounds vs noise levels
    ax = axes[0, 0]
    noise_names = list(noise_configs.keys())
    pcrb_bounds = []
    
    for name, config in noise_configs.items():
        pcrb_computer = PCRBComputerFixed(config)
        J_history = pcrb_computer.compute_pcrb_sequence(100)
        bounds = pcrb_computer.extract_bounds(J_history)
        final_bound = bounds['position'][-1]
        pcrb_bounds.append(final_bound)
    
    colors_noise = ['green', 'orange', 'red']
    bars = ax.bar(noise_names, pcrb_bounds, color=colors_noise, alpha=0.7)
    ax.set_ylabel('Position RMSE Bound (m)')
    ax.set_title('PCRB Bounds vs Noise Assumptions')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, bound in zip(bars, pcrb_bounds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{bound:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Actual performance vs realistic PCRB
    ax = axes[0, 1]
    realistic_pcrb = pcrb_bounds[1]  # Use realistic bound
    
    methods = list(actual_performance.keys())
    rmse_values = list(actual_performance.values())
    
    # Plot PCRB bound line
    ax.axhline(y=realistic_pcrb, color='blue', linestyle='-', linewidth=3,
              label=f'Realistic PCRB: {realistic_pcrb:.3f}m')
    
    # Plot actual performance
    colors_perf = ['gray', 'lightblue', 'blue', 'darkblue', 'navy']
    bars = ax.bar(range(len(methods)), rmse_values, color=colors_perf, alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Actual Performance vs Realistic PCRB')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{rmse:.2f}m', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Subplot 3: Efficiency ratios
    ax = axes[1, 0]
    efficiencies = [realistic_pcrb / rmse for rmse in rmse_values]
    
    bars = ax.bar(range(len(methods)), efficiencies, color=colors_perf, alpha=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Efficiency Ratio (PCRB/Actual)')
    ax.set_title('Filter Efficiency vs Theoretical Bound')
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{eff*100:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Subplot 4: Speedup vs Efficiency tradeoff
    ax = axes[1, 1]
    
    # Extract hybrid tracker data
    hybrid_methods = ['N=2', 'N=3', 'N=5', 'N=10']
    hybrid_rmse = [rmse_values[i+1] for i in range(4)]  # Skip CoTracker3
    hybrid_speedups = [2, 3, 5, 10]
    hybrid_efficiencies = [realistic_pcrb / rmse for rmse in hybrid_rmse]
    
    # Plot speedup vs efficiency
    colors_hybrid = ['lightblue', 'blue', 'darkblue', 'navy']
    for i, (speedup, eff, method) in enumerate(zip(hybrid_speedups, hybrid_efficiencies, hybrid_methods)):
        ax.scatter(speedup, eff*100, s=200, color=colors_hybrid[i], alpha=0.8, 
                  edgecolors='black', linewidths=2, label=method)
        ax.text(speedup, eff*100 + 0.3, method, ha='center', va='bottom', 
               fontweight='bold', fontsize=10)
    
    # Connect points with line
    ax.plot(hybrid_speedups, [e*100 for e in hybrid_efficiencies], 'k--', alpha=0.5)
    
    ax.set_xlabel('Speedup Factor (×)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Speedup vs Efficiency Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 15])
    
    plt.tight_layout()
    
    save_path = 'outputs/figure3_efficiency_analysis_corrected.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 3 saved: {save_path}")
    plt.show()


def generate_figure_4_model_limitations():
    """
    Figure 4: Model Limitations and Sensitivity Analysis
    Shows why linear PCRB fails for collision dynamics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Linear vs Nonlinear trajectory comparison
    ax = axes[0, 0]
    
    # Simulate simple trajectory
    t = np.linspace(0, 2, 100)
    
    # Linear model prediction
    x_linear = 2 + 1.5 * t
    y_linear = 5 - 0.5 * 9.81 * t**2
    
    # Actual bouncing trajectory (simplified)
    y_actual = np.copy(y_linear)
    bounce_idx = np.where(y_actual < 0)[0]
    if len(bounce_idx) > 0:
        first_bounce = bounce_idx[0]
        # Simple bounce model
        for i in range(first_bounce, len(y_actual)):
            if y_actual[i] < 0:
                y_actual[i] = -0.8 * y_actual[i]  # Elastic collision
    
    ax.plot(x_linear, y_linear, 'b--', label='Linear Model Prediction', linewidth=2)
    ax.plot(x_linear, y_actual, 'r-', label='Actual Bouncing Trajectory', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=3, label='Floor')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Linear Model vs Actual Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 6])
    
    # Subplot 2: Noise sensitivity
    ax = axes[0, 1]
    
    meas_noise_range = np.logspace(-2, 0, 20)  # 0.01 to 1.0 meters
    final_bounds = []
    
    for sigma_m in meas_noise_range:
        config = PCRBConfig(process_noise_std=0.2, measurement_noise_std=sigma_m)
        pcrb_computer = PCRBComputerFixed(config)
        J_history = pcrb_computer.compute_pcrb_sequence(50)
        bounds = pcrb_computer.extract_bounds(J_history)
        final_bounds.append(bounds['position'][-1])
    
    ax.loglog(meas_noise_range, final_bounds, 'b-', linewidth=2, label='PCRB Bound')
    ax.loglog(meas_noise_range, meas_noise_range, 'r--', linewidth=2, label='Measurement Noise')
    
    # Mark realistic operating points
    realistic_points = [0.05, 0.1, 0.15]
    for point in realistic_points:
        idx = np.argmin(np.abs(meas_noise_range - point))
        ax.plot(point, final_bounds[idx], 'ro', markersize=8)
    
    ax.set_xlabel('Measurement Noise Std (m)')
    ax.set_ylabel('PCRB Position Bound (m)')
    ax.set_title('PCRB Sensitivity to Measurement Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Process noise sensitivity
    ax = axes[1, 0]
    
    proc_noise_range = np.logspace(-2, 0, 20)  # 0.01 to 1.0 meters
    final_bounds_proc = []
    
    for sigma_p in proc_noise_range:
        config = PCRBConfig(process_noise_std=sigma_p, measurement_noise_std=0.1)
        pcrb_computer = PCRBComputerFixed(config)
        J_history = pcrb_computer.compute_pcrb_sequence(50)
        bounds = pcrb_computer.extract_bounds(J_history)
        final_bounds_proc.append(bounds['position'][-1])
    
    ax.loglog(proc_noise_range, final_bounds_proc, 'g-', linewidth=2, label='PCRB Bound')
    ax.loglog(proc_noise_range, proc_noise_range, 'r--', linewidth=2, label='Process Noise')
    
    ax.set_xlabel('Process Noise Std (m)')
    ax.set_ylabel('PCRB Position Bound (m)')
    ax.set_title('PCRB Sensitivity to Process Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Key insights text
    ax = axes[1, 1]
    ax.axis('off')
    
    insights_text = """
KEY INSIGHTS:

1. LINEAR MODEL LIMITATIONS:
   • PCRB assumes Gaussian dynamics
   • Collisions are highly nonlinear
   • Linear prediction fails at bounces

2. REALISTIC NOISE LEVELS:
   • Process: 20-30cm (model uncertainty)
   • Measurement: 10-15cm (video tracking)
   • Much higher than initial assumptions

3. EFFICIENCY INTERPRETATION:
   • 10% efficiency is reasonable
   • Accounts for model mismatch
   • Better than naive expectations

4. PRACTICAL IMPLICATIONS:
   • Hybrid N=5 well-designed
   • Balances speed vs accuracy
   • Operates in realistic range
    """
    
    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    save_path = 'outputs/figure4_model_limitations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved: {save_path}")
    plt.show()


def update_latex_figures():
    """
    Update the LaTeX report with correct figure references
    """
    latex_updates = """
% Add these figure references to HW3_Report.tex

\\begin{figure}[h]
    \\centering
    \\includegraphics[width=\\textwidth]{outputs/figure1_pcrb_matrices.png}
    \\caption{PCRB derivation components: State transition matrix F, process noise Q, measurement matrix H, and measurement noise R for the 4D point tracking model.}
    \\label{fig:pcrb_matrices}
\\end{figure}

\\begin{figure}[h]
    \\centering
    \\includegraphics[width=\\textwidth]{outputs/figure2_pcrb_evolution_corrected.png}
    \\caption{Corrected PCRB evolution over time for all scenarios with realistic noise parameters. Shows position and velocity bounds separately.}
    \\label{fig:pcrb_evolution}
\\end{figure}

\\begin{figure}[h]
    \\centering
    \\includegraphics[width=\\textwidth]{outputs/figure3_efficiency_analysis_corrected.png}
    \\caption{Corrected efficiency analysis showing realistic PCRB bounds, actual tracker performance, efficiency ratios, and speedup-accuracy tradeoff.}
    \\label{fig:efficiency_analysis}
\\end{figure}

\\begin{figure}[h]
    \\centering
    \\includegraphics[width=\\textwidth]{outputs/figure4_model_limitations.png}
    \\caption{Model limitations analysis: Linear vs nonlinear dynamics, noise sensitivity, and key insights about PCRB applicability to collision dynamics.}
    \\label{fig:model_limitations}
\\end{figure}
    """
    
    with open('outputs/latex_figure_references.txt', 'w') as f:
        f.write(latex_updates)
    
    print("LaTeX figure references saved to: outputs/latex_figure_references.txt")


def main():
    """Generate all HW3 figures with corrected analysis"""
    
    print("=" * 60)
    print("GENERATING ALL HW3 FIGURES (CORRECTED)")
    print("=" * 60)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Generate all figures
    print("\nGenerating Figure 1: PCRB Derivation...")
    generate_figure_1_pcrb_derivation()
    
    print("\nGenerating Figure 2: PCRB Evolution...")
    generate_figure_2_pcrb_evolution()
    
    print("\nGenerating Figure 3: Efficiency Analysis...")
    generate_figure_3_efficiency_analysis()
    
    print("\nGenerating Figure 4: Model Limitations...")
    generate_figure_4_model_limitations()
    
    # Update LaTeX references
    update_latex_figures()
    
    print("\n" + "=" * 60)
    print("ALL HW3 FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- outputs/figure1_pcrb_matrices.png")
    print("- outputs/figure2_pcrb_evolution_corrected.png") 
    print("- outputs/figure3_efficiency_analysis_corrected.png")
    print("- outputs/figure4_model_limitations.png")
    print("- outputs/latex_figure_references.txt")
    print("\nAll figures now show corrected analysis with realistic parameters!")


if __name__ == "__main__":
    main()
