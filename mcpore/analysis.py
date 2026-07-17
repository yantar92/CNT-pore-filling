"""Formation energy analysis and publication-quality plotting functions."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

from mcpore.core import HardCarbonPoreModel, save_model_svg, RESULTS_CSV_COLUMNS
from mcpore.plotting import setup_mpl_style


def get_formation_energies(radius, defect_probability=0.058*3, norm='Na', quiet=False):
    """Get formation energy vs. concentration for pore with RADIUS.

    Return (filling_ratios, energies).
    Unless QUIET is True, save snapshots of the pore for each concentration.
    """
    model = HardCarbonPoreModel(
        pore_radius_angstrom=radius,
        temperature_k=4000,
        defect_probability=defect_probability,
        voltage=0)
    filling_ratios = [0]
    energies = [model.formation_energy(norm)]
    for idx in range(len(model.valid_sites)):
        min_energy = 1E100
        min_loc = None
        for r, c in model.valid_sites:
            if model.grid[r, c] == model.EMPTY:
                model.grid[r, c] = model.NA
                new_energy = model.formation_energy(norm)
                model.grid[r, c] = model.EMPTY
                if new_energy < min_energy:
                    min_energy = new_energy
                    min_loc = (r, c)
        assert min_loc is not None
        model.grid[min_loc] = model.NA
        filling_ratios.append(model.get_filling_fraction())
        energies.append(model.formation_energy(norm))
        if not quiet:
            save_model_svg(model, f'test_{radius}_{defect_probability:.3f}_{idx:03d}.svg')
    return filling_ratios, energies


def plot_filled_pore_energy():
    """Plot formation energy of the pore vs. pore radius.

    Consider the pore to be fully filled.
    """
    energy_na_na = -0.35
    formation_energies = []
    formation_energies2 = []
    formation_energies3 = []
    formation_energies4 = []
    radiuses = []
    inv_radiuses = []
    for radius in np.arange(5, 100, 1, dtype=float):
        model = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=energy_na_na,
            energy_na_na=energy_na_na,
            voltage=0)
        model2 = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=-0.33,
            energy_na_na=energy_na_na,
            voltage=0)
        model3 = HardCarbonPoreModel(
            radius,
            defect_probability=0.174,
            energy_na_c=-0.33,
            energy_na_na=energy_na_na,
            voltage=0)
        model4 = HardCarbonPoreModel(
            radius,
            defect_probability=0,
            energy_na_c=0,
            energy_na_na=energy_na_na,
            voltage=0)
        # Fill the pore
        for m in [model, model2, model3, model4]:
            for r, c in m.valid_sites:
                m.grid[r, c] = model.NA
        formation_energies.append(model.formation_energy())
        formation_energies2.append(model2.formation_energy())
        formation_energies3.append(model3.formation_energy())
        formation_energies4.append(model4.formation_energy())
        radiuses.append(radius)
        inv_radiuses.append(1.0/radius)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(inv_radiuses, formation_energies, 'o-', label='Na in Na')
    ax.plot(inv_radiuses, formation_energies2, 'o-', label='Na in C')
    ax.plot(inv_radiuses, formation_energies3, 'o-', label='Na in C (with defects)')
    ax.plot(inv_radiuses, formation_energies4, 'o-', label='Na in C (Na-C = 0eV)')
    ax.set_xlabel('Reciprocal radius, 1/Å')
    ax.set_ylabel('Formation energy, eV/atom')
    ax.set_title('Formation energy of fully filled pore (Na in Na)')
    ax.legend()
    ax.grid()
    plt.show()


def plot_filling_barriers(
        defect_probabilities=None,
        radii=None):
    """Plot maximum formation barrier vs. pore diameter for a range of defect densities."""
    if defect_probabilities is None:
        defect_probabilities = [0, 0.012, 0.015, 0.02, 0.03, 0.05, 0.1, 0.058*3, 0.25]
    if radii is None:
        radii = np.arange(5, 31, 1)

    setup_mpl_style(width=4.13 * 2 * 1.12, ratio=0.45)

    fig, ax = plt.subplots(1, 1)

    for defect_probability in defect_probabilities:
        barriers = []
        barriers_q1 = []
        barriers_q3 = []
        min_barrier = 0
        radius_list = []
        seen = []
        print(defect_probability)
        for radius in radii:
            tem = HardCarbonPoreModel(pore_radius_angstrom=radius)
            n_sites = len(tem.valid_sites)
            if n_sites in seen:
                print(f"Skipping r={radius}")
                continue
            seen.append(n_sites)
            print(f"r={radius}")
            N_SAMPLES = 100
            with Pool() as pool:
                args = [(radius, defect_probability, None, True) for _ in range(N_SAMPLES)]
                results = pool.starmap(get_formation_energies, args)
            max_energies = []
            for filling_ratios, energies in results:
                energies = [e - x * energies[-1] - (1 - x) * energies[0]
                            for x, e in zip(filling_ratios, energies)]
                max_energies.append(max(energies))
            max_en = np.median(max_energies)
            max_en_q1 = np.quantile(max_energies, 0.25)
            max_en_q3 = np.quantile(max_energies, 0.75)
            max_en = max_en if max_en > min_barrier else 0
            max_en_q1 = max_en_q1 if max_en_q1 > min_barrier else 0
            max_en_q3 = max_en_q3 if max_en_q3 > min_barrier else 0
            barriers.append(max_en)
            barriers_q1.append(max_en_q1)
            barriers_q3.append(max_en_q3)
            actual_radius = tem.real_radius_angstrom
            radius_list.append(actual_radius*2/10)
        ax.plot(
            radius_list, barriers,
            'o-',
            label=f'{defect_probability:.3f}')
        ax.fill_between(radius_list, barriers_q1, barriers_q3, alpha=0.2)
    ax.set_xlabel('Diameter, nm')
    ax.set_ylabel('Maximum formation barrier, eV')
    ax.legend()
    name = "filling_barrier_vs_radius"
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')


def plot_formation_energies(
        radii=None,
        defect_probability=0.058*3,
        norm='pore',
        temperature=0):
    """Plot formation energies as a function of filling ratio for multiple radii."""
    if radii is None:
        radii = [5, 6, 10, 16, 20, 24, 30]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for radius in radii:
        filling_ratios, energies = get_formation_energies(
            radius, defect_probability, norm=norm, quiet=True)
        if temperature > 0:
            free_energies = []
            tmp_model = HardCarbonPoreModel()
            for c, en in zip(filling_ratios, energies):
                if np.isclose(c, 0) or np.isclose(c, 1):
                    free_energies.append(en)
                else:
                    free_energies.append(
                        en - tmp_model.kB * temperature * (
                            - c * np.log(c) - (1 - c) * np.log(1 - c)
                        ))
            energies = free_energies
        ax.plot(filling_ratios, energies, 'o-', label=f'{radius}Å')
    ax.set_xlabel('Filling ratio')
    name = 'Formation energy' if np.isclose(temperature, 0) else 'Free energy'
    if norm is None:
        ax.set_ylabel(f'{name}, eV')
    elif norm == 'pore':
        ax.set_ylabel(f'{name}, eV/site')
    else:
        ax.set_ylabel('Formation energy, eV/atom')
    ax.set_title(f'Gradually filled pore (defects: {defect_probability}, T={temperature}K)')
    ax.legend()
    # ax.set_xlim(0, 0.2)
    # ax.set_ylim(-1, 1)
    ax.grid()
    name = f"formation_energy_vs_filling_{defect_probability:.2f}_{temperature:.0f}K"
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')


def get_voltage_profile(radius, defect_probability=0, norm=None, quiet=True):
    """Compute voltage profile for a pore of given RADIUS.

    Calls get_formation_energies internally, then uses pymatgen to
    convert formation energies into insertion voltages.

    Parameters
    ----------
    radius : float
        Pore radius in angstroms.
    defect_probability : float
        Probability of defect sites.
    norm : str or None
        Normalization for formation energies (passed to get_formation_energies).
    quiet : bool
        If True, suppress per-step SVG output.

    Returns
    -------
    filling_fraction : list of float
        Filling fraction (0 to 1) at each voltage step.
    voltage : list of float
        Insertion voltage (V) at each filling step.
    """
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.apps.battery.insertion_battery import InsertionElectrode
    from pymatgen.apps.battery.plotter import VoltageProfilePlotter
    from pymatgen.core import Composition

    _, energies = get_formation_energies(
        radius, defect_probability, norm=norm, quiet=quiet)
    n_na = len(energies) - 1

    na_entry = ComputedEntry(Composition("Na"), 0)
    na_entry.data["volume"] = 1
    c_entry = ComputedEntry(Composition("C"), 0)
    c_entry.data["volume"] = 1
    entries = [c_entry]
    for n, e in enumerate(energies):
        if n == 0:
            continue
        entry = ComputedEntry(Composition(f"Na{n}C"), e)
        entry.data["volume"] = 1
        entries.append(entry)
    electrode = InsertionElectrode.from_entries(
        entries, working_ion_entry=na_entry, strip_structures=False)
    plotter = VoltageProfilePlotter(xaxis='x_form')
    x, voltage = plotter.get_plot_data(electrode, term_zero=False)
    filling_fraction = [xi / n_na for xi in x]
    return filling_fraction, voltage


def plot_filling_voltages(radii=None, defect_probability=0):
    """Plot filling voltage vs. pore diameter.

    Uses pymatgen to compute insertion voltages from formation energies.
    """
    if radii is None:
        radii = np.arange(5, 31)

    setup_mpl_style()

    fig, ax = plt.subplots(1, 1)

    ds = []
    vs = []

    for radius in radii:
        tem = HardCarbonPoreModel(pore_radius_angstrom=radius)
        print(tem.real_radius_angstrom)
        _, voltage = get_voltage_profile(radius, defect_probability)
        ds.append(tem.real_radius_angstrom * 2 / 10.0)
        vs.append(voltage[-1])
    ax.plot(ds, vs, 'o', color='black')
    # fit ds and vs as vs = A / ds + B and plot
    fit = np.polyfit(1/np.array(ds), vs, 1)
    x_vals = np.arange(min(ds), max(ds), 0.01)
    ax.plot(x_vals, fit[0] / x_vals + fit[1], '-', color='black',
            label=f'V $\\sim$ {fit[0]:.2f}/d')
    ax.set_xlabel('Diameter, nm')
    ax.set_ylabel('Filling voltage, V')
    ax.legend()
    name = f"filling_voltage_{defect_probability:.2f}"
    plt.tight_layout()
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')


def plot_voltages(radii=None, defect_probability=0.058*3):
    """Plot voltage profiles (voltage vs. filling ratio) for multiple radii.

    Uses pymatgen to compute insertion voltages from formation energies.
    """
    if radii is None:
        radii = [5, 7, 8, 10, 16, 20, 24, 30]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for radius in radii:
        tem = HardCarbonPoreModel(pore_radius_angstrom=radius)
        x, voltage = get_voltage_profile(radius, defect_probability)
        ax.plot(x, voltage, 'o-',
                label=f'{tem.real_radius_angstrom}Å')
    ax.set_xlabel('Filling ratio')
    ax.set_ylabel('Voltage, V')
    ax.set_title(f'Voltages for gradually filled pore (defects: {defect_probability})')
    ax.legend()
    ax.grid()
    name = f"voltage_{defect_probability:.2f}"
    plt.savefig(f'{name}.svg')
    plt.savefig(f'{name}.png')


# --- Parameter scan analysis (from original analyze.py) ---

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


def analyze_pore_filling(
    input_file='results.csv',
    output_dir='.',
    show_plots=False,
    verbose=True
):
    """Main analysis pipeline for parameter scan results.

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

    # 1. Load and preprocess data
    if verbose:
        print(f"Loading data from {input_file}...")

    try:
        df = pd.read_csv(
            input_file,
            names=RESULTS_CSV_COLUMNS,
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

    # 2. Filter and validate data
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

    # 3. Identify constant parameters
    constant_param_candidates = [
        'temperature', 'energy_na_na', 'energy_na_c', 'steps',
        'defect_placement'
    ]

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

    # 4. Group and aggregate data
    if verbose:
        print("Grouping data by parameters and averaging replicates...")

    group_cols = [
        'voltage', 'radius', 'defect_probability', 'energy_na_defect'
    ]

    for col in varying_params:
        if col not in group_cols:
            group_cols.append(col)

    grouped = df.groupby(group_cols, as_index=False).agg({
        'final_filling': ['mean', 'std', 'count', 'min', 'max'],
        'n_valid_sites': 'first',
        'n_surface_sites': 'first'
    })

    grouped.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0]
        for col in grouped.columns
    ]

    expected_replicates = df['seed'].nunique()
    incomplete_groups = grouped[grouped['final_filling_count'] < expected_replicates]

    if len(incomplete_groups) > 0:
        warning_msg = (
            f"Warning: {len(incomplete_groups):,} parameter combinations have "
            f"fewer than {expected_replicates} replicates "
            f"(min: {incomplete_groups['final_filling_count'].min()})."
        )
        warnings.warn(warning_msg)
        if verbose:
            print(warning_msg)

    if verbose:
        print(f"Aggregated to {len(grouped):,} unique parameter combinations")

    # 5. Create plots for each defect probability and Na-defect energy
    defect_probs = sorted(df['defect_probability'].unique())
    na_defect_energies = sorted(df['energy_na_defect'].unique())

    if verbose:
        print(f"Creating {len(defect_probs) * len(na_defect_energies)} plots...")
        print(f"Defect probabilities: {defect_probs}")
        print(f"Na-defect energies: {na_defect_energies}")

    plt.style.use(PLOT_STYLE)

    voltage_min = df['voltage'].min()
    voltage_max = df['voltage'].max()
    norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=0.5,
                                 vmin=voltage_min, vmax=voltage_max)
    cmap = mpl.colormaps.get_cmap(COLORMAP)

    plots_created = 0

    for defect_prob in defect_probs:
        for energy_defect in na_defect_energies:
            mask = (grouped['defect_probability'] == defect_prob) & \
                   (grouped['energy_na_defect'] == energy_defect)
            subset = grouped[mask].copy()

            if len(subset) == 0:
                warnings.warn(
                    f"No data for defect_probability={defect_prob}, "
                    f"energy_na_defect={energy_defect}. Skipping."
                )
                continue

            voltages = sorted(subset['voltage'].unique(), reverse=True)
            radii = sorted(subset['radius'].unique())

            surface_fraction = subset.groupby('radius').apply(
                lambda x: x['n_surface_sites_first'].iloc[0] / x['n_valid_sites_first'].iloc[0],
                include_groups=False
            ).sort_index()

            if len(voltages) == 0 or len(radii) == 0:
                continue

            fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            for voltage in voltages:
                voltage_data = subset[subset['voltage'] == voltage].sort_values('radius')

                if len(voltage_data) == 0:
                    continue

                color = cmap(norm(voltage))

                ax.plot(
                    voltage_data['radius'],
                    voltage_data['final_filling_mean'],
                    marker='o',
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                    color=color,
                    label=f'{voltage:.3f} V'
                )

                if 'final_filling_min' in voltage_data.columns and \
                   'final_filling_max' in voltage_data.columns:
                    ax.fill_between(
                        voltage_data['radius'],
                        voltage_data['final_filling_min'],
                        voltage_data['final_filling_max'],
                        alpha=ALPHA_CONFIDENCE,
                        color=color,
                        edgecolor='none'
                    )

            ax.plot(
                surface_fraction.index,
                surface_fraction.values,
                'o-',
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label='Surface site fraction',
                zorder=10, color='red')

            ax.set_xlabel('Pore Radius (Å)', fontsize=FONT_SIZE_LABELS)
            ax.set_ylabel('Final Filling Fraction', fontsize=FONT_SIZE_LABELS)

            ax.set_xlim(min(radii) * 0.95, max(radii) * 1.05)
            ax.set_ylim(0, 1.05)

            ax.grid(True, alpha=0.3, linestyle='-')
            ax.legend(
                ncol=3,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.00),
                fontsize='small',
                frameon=True,
                fancybox=False
            )

            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Voltage (V)', fontsize=FONT_SIZE_LABELS)
            cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)

            title = (
                f"Pore Filling vs Radius: "
                f"defect_probability={defect_prob:.4f}, "
                f"energy_na_defect={energy_defect:.2f} eV"
            )
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=15)

            param_text = []

            for param_name, param_value in constant_params.items():
                if isinstance(param_value, float):
                    param_text.append(f"{param_name} = {param_value:.2f}")
                else:
                    param_text.append(f"{param_name} = {param_value}")

            for param_name, param_values in varying_params.items():
                if param_name not in ['defect_probability', 'energy_na_defect']:
                    if len(param_values) <= 3:
                        values_str = ', '.join(
                            [f"{v:.2f}" if isinstance(v, float) else str(v)
                             for v in sorted(param_values)])
                        param_text.append(f"{param_name} = [{values_str}]")
                    else:
                        param_text.append(f"{param_name}: {len(param_values)} values")

            if param_text:
                param_box = '\n'.join(param_text)
                ax.text(
                    0.98, 0.98, param_box,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE_PARAMS,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='wheat',
                        alpha=0.8,
                        edgecolor='gray'
                    )
                )

            filename = (
                f"filling_vs_radius_Edef_{energy_defect:.2f}_"
                f"pdef_{defect_prob:.4f}.png"
            )
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

    # 6. Summary
    if verbose:
        print(f"\nAnalysis complete!")
        print(f"  Created {plots_created} plots in '{output_dir}'")
        print(f"  Excluded {non_eq_count} non-equilibrium simulations")

        if constant_params:
            print("\nConstant parameters across scan:")
            for param, value in constant_params.items():
                print(f"  {param}: {value}")

        if varying_params:
            print("\nVarying parameters (other than defect_probability, energy_na_defect):")
            for param, values in varying_params.items():
                if param not in ['defect_probability', 'energy_na_defect']:
                    print(f"  {param}: {len(values)} unique values")

    return grouped, constant_params, varying_params
