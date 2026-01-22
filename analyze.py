#!/usr/bin/env python3
"""
Analyze MC pore filling parameter scan results.

Reads results.csv, averages replicates, and creates publication-style plots
showing final_filling vs. radius for different voltages, with separate plots
for each Na-defect energy and defect concentration combination.

Usage:
    python analyze_pore_filling.py [results.csv] [--output-dir ./]

Dependencies:
    pandas, numpy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import os
import sys
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Figure dimensions: portrait A4 width (8.27 inches) with 4:3 aspect ratio
FIG_WIDTH = 8.27  # inches (210 mm)
FIG_HEIGHT = FIG_WIDTH * 3 / 4  # 6.20 inches
DPI = 300

# Plot styling
PLOT_STYLE = 'seaborn-v0_8-whitegrid'  # publication style
COLORMAP = 'viridis'  # continuous colormap for voltage
MARKER_SIZE = 6
LINE_WIDTH = 1.5
ALPHA_CONFIDENCE = 0.2  # transparency for confidence bands
FONT_SIZE_TITLE = 12
FONT_SIZE_LABELS = 11
FONT_SIZE_TICKS = 10
FONT_SIZE_PARAMS = 7  # small font for parameter box

# ============================================================================
# Main analysis function
# ============================================================================

def analyze_pore_filling(
    input_file='results.csv',
    output_dir='.',
    show_plots=False,
    verbose=True
):
    """
    Main analysis pipeline.
    
    Parameters
    ----------
    input_file : str
        Path to results.csv file
    output_dir : str
        Directory to save PNG plots
    show_plots : bool
        If True, display plots interactively
    verbose : bool
        If True, print progress messages
    """
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. Load and preprocess data
    # ========================================================================
    
    if verbose:
        print(f"Loading data from {input_file}...")
    
    try:
        df = pd.read_csv(
            input_file,
            names=[
                "voltage", "radius",
                "defect_probability", "defect_placement",
                "energy_na_defect", "energy_na_na", "energy_na_c",
                "temperature", "steps", "seed",
                "final_filling", "equilibrium_reached", "mcs",
                "n_valid_sites", "n_surface_sites", "default_p_gcmc",
                "mu"
            ],
            )
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please ensure the file exists in the current directory or provide the correct path.")
        sys.exit(1)
    
    if verbose:
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        print("Columns:", df.columns.tolist())
    
    # Check for required columns
    required_cols = [
        'voltage', 'radius', 'defect_probability', 'energy_na_defect',
        'final_filling', 'equilibrium_reached', 'seed'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # ========================================================================
    # 2. Filter and validate data
    # ========================================================================
    
    # Count non-equilibrium simulations
    non_eq_mask = df['equilibrium_reached'] == False
    non_eq_count = non_eq_mask.sum()
    
    if non_eq_count > 0:
        warning_msg = (
            f"Warning: {non_eq_count:,} simulations ({non_eq_count/len(df)*100:.1f}%) "
            f"did not reach equilibrium. These will be excluded from analysis."
        )
        warnings.warn(warning_msg)
        if verbose:
            print(warning_msg)
        
        # Exclude non-equilibrium simulations
        df = df[~non_eq_mask].copy()
    
    if len(df) == 0:
        print("Error: No equilibrium-reached simulations found.")
        sys.exit(1)
    
    # ========================================================================
    # 3. Identify constant parameters
    # ========================================================================
    
    # Parameters that should be constant across the parameter scan
    constant_param_candidates = [
        'temperature', 'energy_na_na', 'energy_na_c', 'steps',
        'defect_placement'
    ]
    
    # Only check columns that exist
    constant_params_to_check = [
        col for col in constant_param_candidates if col in df.columns
    ]
    
    constant_params = {}
    varying_params = {}
    
    for col in constant_params_to_check:
        unique_vals = df[col].unique()
        if len(unique_vals) == 1:
            constant_params[col] = unique_vals[0]
        else:
            varying_params[col] = unique_vals
    
    # ========================================================================
    # 4. Group and aggregate data
    # ========================================================================
    
    if verbose:
        print("Grouping data by parameters and averaging replicates...")
    
    # Columns to group by (excluding seed and final_filling)
    group_cols = [
        'voltage', 'radius', 'defect_probability', 'energy_na_defect'
    ]
    
    # Add any other parameters that might vary (though they shouldn't for this scan)
    for col in varying_params:
        if col not in group_cols:
            group_cols.append(col)
    
    # Group and aggregate
    grouped = df.groupby(group_cols, as_index=False).agg({
        'final_filling': ['mean', 'std', 'count'],
        'n_valid_sites': 'first',  # should be constant for each radius
        'n_surface_sites': 'first'
    })
    
    # Flatten column names
    grouped.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0]
        for col in grouped.columns
    ]
    
    # Rename for convenience
    grouped = grouped.rename(columns={
        'final_filling_mean': 'final_filling_mean',
        'final_filling_std': 'final_filling_std',
        'final_filling_count': 'replicate_count'
    })
    
    # Check for missing replicates
    expected_replicates = df['seed'].nunique()  # Should be 15
    incomplete_groups = grouped[grouped['replicate_count'] < expected_replicates]
    
    if len(incomplete_groups) > 0:
        warning_msg = (
            f"Warning: {len(incomplete_groups):,} parameter combinations have "
            f"fewer than {expected_replicates} replicates "
            f"(min: {incomplete_groups['replicate_count'].min()})."
        )
        warnings.warn(warning_msg)
        if verbose:
            print(warning_msg)
    
    if verbose:
        print(f"Aggregated to {len(grouped):,} unique parameter combinations")
    
    # ========================================================================
    # 5. Create plots for each defect probability and Na-defect energy
    # ========================================================================
    
    # Get unique values for the parameters we'll iterate over
    defect_probs = sorted(df['defect_probability'].unique())
    na_defect_energies = sorted(df['energy_na_defect'].unique())
    
    if verbose:
        print(f"Creating {len(defect_probs) * len(na_defect_energies)} plots...")
        print(f"Defect probabilities: {defect_probs}")
        print(f"Na-defect energies: {na_defect_energies}")
    
    # Set matplotlib style
    plt.style.use(PLOT_STYLE)
    
    # Create a custom colorbar normalization based on voltage range
    voltage_min = df['voltage'].min()
    voltage_max = df['voltage'].max()
    norm = mpl.colors.Normalize(vmin=voltage_min, vmax=voltage_max)
    cmap = mpl.cm.get_cmap(COLORMAP)
    
    # Loop through each combination
    plots_created = 0
    
    for defect_prob in defect_probs:
        for energy_defect in na_defect_energies:
            # Filter data for this combination
            mask = (grouped['defect_probability'] == defect_prob) & \
                   (grouped['energy_na_defect'] == energy_defect)
            subset = grouped[mask].copy()
            
            if len(subset) == 0:
                warnings.warn(
                    f"No data for defect_probability={defect_prob}, "
                    f"energy_na_defect={energy_defect}. Skipping."
                )
                continue
            
            # Get unique voltages and radii for this subset
            voltages = sorted(subset['voltage'].unique())
            radii = sorted(subset['radius'].unique())
            
            if len(voltages) == 0 or len(radii) == 0:
                continue
            
            # ================================================================
            # Create figure
            # ================================================================
            
            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            
            # ================================================================
            # Plot each voltage as a separate line
            # ================================================================
            
            for voltage in voltages:
                # Filter for this voltage
                voltage_data = subset[subset['voltage'] == voltage].sort_values('radius')
                
                if len(voltage_data) == 0:
                    continue
                
                # Get color for this voltage
                color = cmap(norm(voltage))
                
                # Plot mean line with markers
                ax.plot(
                    voltage_data['radius'],
                    voltage_data['final_filling_mean'],
                    marker='o',
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                    color=color,
                    label=f'{voltage:.3f} V'
                )
                
                # Add confidence band (mean ± std)
                if 'final_filling_std' in voltage_data.columns:
                    ax.fill_between(
                        voltage_data['radius'],
                        voltage_data['final_filling_mean'] - voltage_data['final_filling_std'],
                        voltage_data['final_filling_mean'] + voltage_data['final_filling_std'],
                        alpha=ALPHA_CONFIDENCE,
                        color=color,
                        edgecolor='none'
                    )
            
            # ================================================================
            # Formatting
            # ================================================================
            
            # Axis labels
            ax.set_xlabel('Pore Radius (Å)', fontsize=FONT_SIZE_LABELS)
            ax.set_ylabel('Final Filling Fraction', fontsize=FONT_SIZE_LABELS)
            
            # Limits
            ax.set_xlim(min(radii) * 0.95, max(radii) * 1.05)
            ax.set_ylim(0, 1.05)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='-')
            
            # ================================================================
            # Add colorbar for voltage
            # ================================================================
            
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Voltage (V)', fontsize=FONT_SIZE_LABELS)
            cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
            
            # ================================================================
            # Add title and parameter information
            # ================================================================
            
            # Main title
            title = (
                f"Pore Filling vs Radius: "
                f"defect_probability={defect_prob:.4f}, "
                f"energy_na_defect={energy_defect:.2f} eV"
            )
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=15)
            
            # Parameter information box
            param_text = []
            
            # Constant parameters
            for param_name, param_value in constant_params.items():
                if isinstance(param_value, float):
                    param_text.append(f"{param_name} = {param_value:.2f}")
                else:
                    param_text.append(f"{param_name} = {param_value}")
            
            # Varying parameters (other than the ones we're iterating over)
            for param_name, param_values in varying_params.items():
                if param_name not in ['defect_probability', 'energy_na_defect']:
                    if len(param_values) <= 3:
                        values_str = ', '.join([f"{v:.2f}" if isinstance(v, float) else str(v) 
                                               for v in sorted(param_values)])
                        param_text.append(f"{param_name} = [{values_str}]")
                    else:
                        param_text.append(f"{param_name}: {len(param_values)} values")
            
            # Add text box with parameters
            if param_text:
                param_box = '\n'.join(param_text)
                ax.text(
                    0.02, 0.02, param_box,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE_PARAMS,
                    verticalalignment='bottom',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='wheat',
                        alpha=0.8,
                        edgecolor='gray'
                    )
                )
            
            # ================================================================
            # Save figure
            # ================================================================
            
            # Create filename
            filename = (
                f"filling_vs_radius_Edef_{energy_defect:.2f}_"
                f"pdef_{defect_prob:.4f}.png"
            )
            # Replace dots with 'p' for better filename compatibility
            filename = filename.replace('.', 'p')
            filepath = output_path / filename
            
            fig.tight_layout()
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            
            if verbose:
                print(f"  Saved: {filepath}")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
            
            plots_created += 1
    
    # ========================================================================
    # 6. Summary
    # ========================================================================
    
    if verbose:
        print(f"\nAnalysis complete!")
        print(f"  Created {plots_created} plots in '{output_dir}'")
        print(f"  Excluded {non_eq_count} non-equilibrium simulations")
        
        if 'constant_params' in locals():
            print("\nConstant parameters across scan:")
            for param, value in constant_params.items():
                print(f"  {param}: {value}")
        
        if varying_params:
            print("\nVarying parameters (other than defect_probability, energy_na_defect):")
            for param, values in varying_params.items():
                if param not in ['defect_probability', 'energy_na_defect']:
                    print(f"  {param}: {len(values)} unique values")
    
    return grouped, constant_params, varying_params


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze MC pore filling parameter scan results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Use default results.csv in current directory
  %(prog)s my_results.csv          # Specify input file
  %(prog)s --output-dir ./plots    # Save plots to ./plots directory
  %(prog)s --show-plots            # Display plots interactively
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default='results.csv',
        help='Path to results.csv file (default: results.csv)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Directory to save PNG plots (default: current directory)'
    )
    
    parser.add_argument(
        '--show-plots', '-s',
        action='store_true',
        help='Display plots interactively (default: save only)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    try:
        analyze_pore_filling(
            input_file=args.input_file,
            output_dir=args.output_dir,
            show_plots=args.show_plots,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

