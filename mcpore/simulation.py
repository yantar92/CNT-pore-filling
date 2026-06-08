"""Simulation runners for hard-carbon pore-filling MC."""

import numpy as np
import random
import time
import pickle
import copy
import sys
import os
import pandas as pd

from mcpore.core import (
    HardCarbonPoreModel,
    save_model_svg,
    save_timeseries_csv,
    visualize_model,
    CSV_GZ_SUFFIX,
)


def run_simulation(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=298,
            voltage=0.1,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        visualize=True,
        snapshot_file: str | None = 'snapshots.pkl',
        csv_output=False,
        seed=None,
        anneal0K=False,
        quiet=False):
    """Run a Monte Carlo simulation of pore filling.

    Args:
        model: HardCarbonPoreModel
        steps: Number of normalized Monte Carlo steps (MCS)
        snapshot_file: If provided, save output to this file.
            If the filename ends with '.csv' or '.csv.gz', writes the time series
            (MCS, filling %, formation energy) as CSV (optionally gzip-compressed)
            instead of pickle snapshots.
        csv_output: If True, print a CSV line with results to stdout.
        seed: Random seed for reproducibility (None for random).
        quiet: Suppress progress output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine output mode from filename extension
    is_csv_output = (snapshot_file is not None
                     and (snapshot_file.lower().endswith('.csv')
                          or snapshot_file.lower().endswith(CSV_GZ_SUFFIX)))

    MC_STEPS = steps  # Total normalized steps (attempts per site)
    SNAPSHOT_INTERVAL = 400

    # Initialize Model
    model.quiet = quiet
    model.equilibrium_reached = False
    snapshots = []
    BEGIN_MC = model.mcs

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    if not quiet:
        print(f"Starting Simulation: {total_attempts} attempts ({MC_STEPS} MCS)...")
    start_time = time.time()

    # Visualization Setup
    if visualize:
        import matplotlib.pyplot as plt
        fig, ((ax_grid, ax_stats), (dE_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
        plt.show(block=False)

    for attempt in range(total_attempts):
        model.run_step()
        if model.equilibrium_reached and attempt < model.eq_min_mcs:
            model.equilibrium_reached = False

        if model.equilibrium_reached and anneal0K:
            model = run_0K_min(model, steps=steps)

        should_report = (model.equilibrium_reached
                         or attempt == total_attempts - 1
                         or attempt % (SNAPSHOT_INTERVAL * total_sites) == 0)

        if should_report:
            # Collect snapshots only when writing pickle output
            if snapshot_file is not None and not is_csv_output:
                snapshots.append(model.take_snapshot())
            if not quiet and model.equilibrium_reached:
                print(f"Equilibrium reached at MCS {model.mcs:.2f}")
            elif not quiet:
                print(f"  Step {int(model.mcs)}/{MC_STEPS + BEGIN_MC}:"
                      f" Filling = {model.filling_history[-1]:.2f}%")
            if not quiet and visualize:
                visualize_model(model, ax_grid, ax_stats, dE_axis, formation_axis)
                plt.draw()
                plt.pause(0.01)
        if model.equilibrium_reached:
            break

    elapsed = time.time() - start_time
    if not quiet:
        print(f"Simulation Complete in {elapsed:.2f}s")

    if snapshot_file is not None:
        if is_csv_output:
            # Write time series as CSV instead of pickle snapshots
            save_timeseries_csv(model, snapshot_file)
        else:
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshots, f)
            if not quiet:
                print(f"Saved {len(snapshots)} snapshots to {snapshot_file}")

    # CSV output
    if csv_output:
        final_filling = model.get_final_filling_percent()
        row = [
            f"{model.voltage:.6f}",
            f"{model.pore_radius:.1f}",
            f"{model.defect_probability:.6f}",
            model.defect_placement,
            f"{model.energies['Na_Defect']:.6f}",
            f"{model.energies['Na_Na']:.6f}",
            f"{model.energies['Na_C']:.6f}",
            f"{model.T:.1f}",
            f"{steps}",
            str(seed),
            f"{final_filling:.6f}",
            str(model.equilibrium_reached),
            f"{model.mcs:.2f}",
            f"{len(model.valid_sites)}",
            f"{len(model.surface_sites)}",
            f"{model.default_p_gcmc:.6f}",
            f"{model.mu:.6f}",
            f"{model.mcs_fill}",
            f"{model.real_radius_angstrom:.6f}",
        ]
        print(','.join(row))
    return model


def run_voltage_sweep_simulation(
        model=HardCarbonPoreModel(),
        voltages=np.arange(0.2, -1e-9, -0.01),
        steps=20000,
        visualize=True,
        converge=False,
        seed=None,
        anneal0K=False,
        quiet=True):
    """Run MODEL sweeping across VOLTAGES.

    For each voltage, hold up to STEPS or until MODEL stabilization.
    SEED is random seed.
    When VISUALIZE is True, visualize the model.
    When QUIET is True, avoid printing info.
    When CONVERGE is False, run simulation for each voltage once.
    Otherwise, CONVERGE should be a dict {'threshold': 0.01, 'max_runs': 50, 'min_runs': 3}
    When ANNEAL0K is True, anneal at 0K after each step.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if visualize:
        import matplotlib.pyplot as plt
        fig, ((ax_grid, ax_stats), (voltage_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
        plt.show(block=False)
    filling_data = []
    for voltage in voltages:
        model.voltage = voltage
        if converge:
            tem = copy.deepcopy(model)
            run_convergence_simulation(
                tem, steps=steps,
                convergence_threshold=converge['threshold'],
                min_replicates=converge['min_runs'],
                max_replicates=converge['max_runs'],
                quiet=quiet,
                anneal0K=anneal0K,
                snapshot_file=None,
            )
            model = tem
        else:
            run_simulation(
                model,
                steps=steps,
                visualize=False,
                snapshot_file=None,
                csv_output=True,
                anneal0K=anneal0K,
                quiet=quiet)
        save_model_svg(model, f"snapshot_{voltage}.svg")
        filling_data.append(model.get_final_filling_percent())
        if visualize:
            visualize_model(model, ax_grid, ax_stats, formation_axis=formation_axis)
            voltage_axis.clear()
            voltage_axis.set_title('CE profile')
            voltage_axis.set_ylabel('Voltage, V')
            voltage_axis.set_xlabel('Filling ratio, %')
            voltage_axis.plot(filling_data, voltages[:len(filling_data)])
            plt.draw()
            plt.pause(0.01)


def run_convergence_simulation(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=298,
            voltage=0.1,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        convergence_threshold=0.01,
        min_replicates=3,
        max_replicates=50,
        seed=None,
        anneal0K=False,
        snapshot_file=None,
        quiet=False):
    """Run multiple simulations until statistics converge.

    When SNAPSHOT_FILE is given, each replicate saves output using the file name
    as base with replicate index suffix (e.g. data_r0.csv, data_r1.csv).
    Returns list of (final_filling, mcs_fill) tuples.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    replicates = []
    fill_means = []
    fill_stds = []
    time_means = []
    time_stds = []

    for i in range(max_replicates):
        # Generate per-replicate filename if snapshot_file is given
        rep_snapshot = None
        if snapshot_file is not None:
            if snapshot_file.endswith(CSV_GZ_SUFFIX):
                base = snapshot_file[:-len(CSV_GZ_SUFFIX)]
                ext = CSV_GZ_SUFFIX
            else:
                base, ext = os.path.splitext(snapshot_file)
            rep_snapshot = f"{base}_r{i}{ext}"

        # Run simulation with csv_output=True (prints CSV line)
        model_tem = copy.deepcopy(model)
        model_tem.quiet = quiet
        model_tem = run_simulation(
            model=model_tem,
            steps=steps,
            visualize=False,
            snapshot_file=rep_snapshot,
            csv_output=True,
            seed=None,
            anneal0K=anneal0K,
            quiet=quiet)

        # Extract results
        final_filling = model_tem.get_final_filling_percent()
        mcs_fill = model_tem.mcs_fill if model_tem.mcs_fill is not None else 0.0

        replicates.append((final_filling, mcs_fill))

        # Check convergence after min_replicates
        if len(replicates) >= min_replicates:
            # Compute current statistics
            fill_array = np.array([r[0] for r in replicates])
            time_array = np.array([r[1] for r in replicates])
            fill_mean = fill_array.mean()
            fill_std = fill_array.std(ddof=1) if len(replicates) > 1 else 0.0
            time_mean = time_array.mean()
            time_std = time_array.std(ddof=1) if len(replicates) > 1 else 0.0

            if np.isclose(fill_mean, 0):
                fill_mean = 0
            if np.isclose(fill_std, 0):
                fill_std = 0
            if np.isclose(time_mean, 0):
                time_mean = 0
            if np.isclose(time_std, 0):
                time_std = 0

            # Compute relative changes (handle zero denominators)
            fill_mean_change = 0.0
            fill_std_change = 0.0
            time_mean_change = 0.0
            time_std_change = 0.0

            if fill_means:
                prev_fill_mean = fill_means[-1]
                if abs(prev_fill_mean) > 1e-12:
                    fill_mean_change = abs(fill_mean - prev_fill_mean) / prev_fill_mean
                else:
                    fill_mean_change = abs(fill_mean - prev_fill_mean)
                prev_fill_std = fill_stds[-1]
                if prev_fill_std > 1e-12:
                    fill_std_change = abs(fill_std - prev_fill_std) / prev_fill_std
                else:
                    fill_std_change = abs(fill_std - prev_fill_std)

            if time_means:
                prev_time_mean = time_means[-1]
                if abs(prev_time_mean) > 1e-12:
                    time_mean_change = abs(time_mean - prev_time_mean) / prev_time_mean
                else:
                    time_mean_change = abs(time_mean - prev_time_mean)
                prev_time_std = time_stds[-1]
                if prev_time_std > 1e-12:
                    time_std_change = abs(time_std - prev_time_std) / prev_time_std
                else:
                    time_std_change = abs(time_std - prev_time_std)

            fill_means.append(fill_mean)
            fill_stds.append(fill_std)
            time_means.append(time_mean)
            time_stds.append(time_std)

            if not quiet:
                print(f"fill_mean: Δ{fill_mean_change}, fill_std: Δ{fill_std_change}")
                print(f"time_mean: Δ{time_mean_change}, time_std: Δ{time_std_change}")
            # Check if all changes below threshold
            if (fill_mean_change <= convergence_threshold and
                fill_std_change <= convergence_threshold and
                time_mean_change <= convergence_threshold and
                time_std_change <= convergence_threshold):
                if not quiet:
                    print(f"Convergence reached after {i+1} replicates", file=sys.stderr)
                break
        else:
            # Not enough replicates yet, still store stats for future comparison
            fill_array = np.array([r[0] for r in replicates])
            time_array = np.array([r[1] for r in replicates])
            fill_mean = fill_array.mean()
            fill_std = fill_array.std(ddof=1) if len(replicates) > 1 else 0.0
            time_mean = time_array.mean()
            time_std = time_array.std(ddof=1) if len(replicates) > 1 else 0.0
            fill_means.append(fill_mean)
            fill_stds.append(fill_std)
            time_means.append(time_mean)
            time_stds.append(time_std)

    return replicates


def run_0K_min(
        model=HardCarbonPoreModel(
            pore_radius_angstrom=10.0,
            temperature_k=0,
            voltage=0.1,
            defect_probability=0.058 * 3,
            defect_placement='surface',
            energy_na_na=-0.35,
            energy_na_c=-0.32,
            energy_na_defect=-1.77,
            quiet=True,
        ),
        steps=20000,
        seed=None):
    """Minimize energy in MODEL at 0K, while keeping the number of Na constant.

    Args:
        model: HardCarbonPoreModel
        steps: Number of normalized Monte Carlo steps (MCS)
        seed: Random seed for reproducibility (None for random).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    MC_STEPS = steps  # Total normalized steps (attempts per site)

    # Initialize Model
    old_T = model.T
    old_quiet = model.quiet
    model.quiet = True

    total_sites = len(model.valid_sites)
    total_attempts = MC_STEPS * total_sites

    N_temps = 5

    # Down to 1K, we cannot use literally 0K.
    for new_T in np.logspace(np.log10(old_T), np.log10(1), N_temps):
        model.T = new_T

        for attempt in range(int(total_attempts/N_temps)):
            # Run step, while disallowing Na exiting or entering.
            model.run_step(p_gcmc=0)

    model.quiet = old_quiet
    model.T = old_T

    return model


def replay_simulation(snapshot_file, interval=0.01, every=1):
    """Load snapshots from SNAPSHOT_FILE and visualize them sequentially.

    INTERVAL is pause time between frames in seconds.
    EVERY X will only show every X's snapshot.
    """
    import matplotlib.pyplot as plt

    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)

    print(f"Loaded {len(snapshots)} snapshots")

    fig, ((ax_grid, ax_stats), (dE_axis, formation_axis)) = plt.subplots(2, 2, figsize=(10, 10))
    plt.show(block=False)

    for i, model in enumerate(snapshots):
        if i % every != 0:
            continue
        visualize_model(model, ax_grid, ax_stats, dE_axis, formation_axis)
        ax_grid.set_title(f"Pore State (MCS: {int(model.mcs)}) - Snapshot {i+1}/{len(snapshots)}")
        ax_stats.set_title(
            f"Filling Kinetics (P_GCMC={model.default_p_gcmc:.2f}) - Snapshot {i+1}/{len(snapshots)}")
        plt.draw()
        plt.pause(interval)

    plt.show()


def summarize_snapshots(pattern="*.pkl", output_csv="summary.csv"):
    """Process all .pkl files matching PATTERN, extract final snapshot data.

    Saves summary to OUTPUT_CSV.
    """
    import glob
    import pandas as pd
    import traceback

    files = glob.glob(pattern)
    if not files:
        print(f"No files matching pattern '{pattern}'")
        return

    print(f"Found {len(files)} files")

    data_rows = []

    for fpath in sorted(files):
        try:
            with open(fpath, 'rb') as f:
                snapshots = pickle.load(f)

            if not snapshots:
                print(f"Warning: {fpath} contains no snapshots")
                continue

            # Take the last snapshot
            model = snapshots[-1]

            # Extract parameters
            row = {
                'filename': fpath,
                'pore_radius_A': model.pore_radius,
                'voltage_V': model.voltage,
                'defect_probability': model.defect_probability,
                'defect_placement': model.defect_placement,
                'temperature_K': model.T,
                'energy_na_na_eV': model.energies['Na_Na'],
                'energy_na_c_eV': model.energies['Na_C'],
                'energy_na_defect_eV': model.energies['Na_Defect'],
                'final_filling': model.get_final_filling_percent(),
                'final_mcs': model.mcs,
                'n_snapshots': len(snapshots),
                'n_valid_sites': len(model.valid_sites),
                'n_surface_sites': len(model.surface_sites),
                'default_p_gcmc': model.default_p_gcmc,
                'mu_eV': model.mu,
                'fill_mcs': model.mcs_fill,
                'real_radius_A': model.real_radius_angstrom,
            }

            data_rows.append(row)
            print(f"Processed {fpath}: R={model.pore_radius:.1f}Å, "
                  f"V={model.voltage:.2f}V, filling={row['final_filling']:.3f}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            traceback.print_exc()
            continue

    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_csv(output_csv, index=False)
        print(f"Saved summary to {output_csv} with {len(df)} rows")
        return df
    else:
        print("No valid data extracted")
        return None
