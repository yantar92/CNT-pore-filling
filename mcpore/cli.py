"""Command-line interface for MC pore filling simulation.

Usage:
    python -m mcpore.cli [options]
    python mc-pore.py [options]   # backward-compatible wrapper
"""

import argparse
import sys

from mcpore.core import HardCarbonPoreModel
from mcpore.simulation import (
    run_simulation,
    run_convergence_simulation,
    run_voltage_sweep_simulation,
)


def main():
    parser = argparse.ArgumentParser(
        description='Metropolis Monte Carlo simulation of pore filling in hard carbon.'
    )
    parser.add_argument('--voltage', type=float, default=[0.1], nargs='*',
                        help='Voltage relative to bulk Na (V); single value or multiple for sweep')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Pore radius (Å)')
    parser.add_argument('--file', type=str, default='snapshots.pkl',
                        help='Output file. If the filename ends with .csv, writes time series'
                        ' CSV instead of pickle snapshots.'
                        ' In convergence mode, appends _rN suffix per replicate.')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='Number of normalized Monte Carlo steps (MCS)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable live visualization')
    parser.add_argument('--energy_na_defect', type=float, default=-1.53,
                        help='Na-defect interaction energy (eV)')
    parser.add_argument('--energy_na_na', type=float, default=-0.35,
                        help='Na-Na interaction energy (eV)')
    parser.add_argument('--energy_na_c', type=float, default=-0.26,
                        help='Na-C interaction energy (eV)')
    parser.add_argument('--temp', type=float, default=298.0,
                        help='Temperature (K)')
    parser.add_argument('--defect_placement', type=str, default='surface',
                        choices=['surface', 'random'],
                        help='Defect placement mode')
    parser.add_argument('--defect_probability', type=float, default=0.058*3,
                        help='Defect probability (fraction)')
    parser.add_argument('--converge', action='store_true',
                        help='Enable convergence loop: run replicates until statistics stabilize')
    parser.add_argument('--convergence_threshold', type=float, default=0.05,
                        help='Relative change threshold for mean and std (default: 0.05)')
    parser.add_argument('--min_replicates', type=int, default=3,
                        help='Minimum number of replicates before checking convergence (default: 3)')
    parser.add_argument('--max_replicates', type=int, default=50,
                        help='Maximum number of replicates (default: 50)')
    parser.add_argument('--csv', action='store_true',
                        help='Output a single CSV line with final results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress all progress output')
    parser.add_argument('--anneal', action='store_true',
                        help='Anneal the model at 0K after each step')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--p_swap', type=float, default=0.0,
                        help='Probability of non-local swap moves. '
                        'Use >0 (e.g. 0.15) for voltage sweeps/equilibration; '
                        'use 0 for kinetics (default: 0)')
    parser.add_argument('--initial-na-layers', type=int, default=0,
                        help='Number of wall-adjacent layers to pre-fill with Na '
                        'at time=0. 0 = empty pore (default), '
                        '1 = surface sites, 2+ = deeper layers.')
    args = parser.parse_args()

    # Normalize voltage to a list
    if isinstance(args.voltage, list):
        voltages = args.voltage
    else:
        voltages = [args.voltage]

    # Prepare converge dict if converge flag is set
    converge_dict = False
    if args.converge:
        converge_dict = {
            'threshold': args.convergence_threshold,
            'max_runs': args.max_replicates,
            'min_runs': args.min_replicates
        }

    model = HardCarbonPoreModel(
        voltage=voltages[0],
        temperature_k=args.temp,
        pore_radius_angstrom=args.radius,
        defect_placement=args.defect_placement,
        defect_probability=args.defect_probability,
        energy_na_na=args.energy_na_na,
        energy_na_c=args.energy_na_c,
        energy_na_defect=args.energy_na_defect,
        initial_na_layers=args.initial_na_layers,
        quiet=args.quiet,
        seed=args.seed
    )

    if len(voltages) == 1:
        # Single voltage mode
        if args.converge:
            run_convergence_simulation(
                model,
                steps=args.steps,
                convergence_threshold=args.convergence_threshold,
                min_replicates=args.min_replicates,
                max_replicates=args.max_replicates,
                seed=args.seed,
                anneal0K=args.anneal,
                p_swap=args.p_swap,
                snapshot_file=args.file,
                quiet=args.quiet)
        else:
            run_simulation(
                model,
                visualize=args.visualize,
                snapshot_file=args.file,
                steps=args.steps,
                csv_output=args.csv,
                quiet=args.quiet,
                anneal0K=args.anneal,
                p_swap=args.p_swap,
                seed=args.seed)
    else:
        # Multiple voltages: run voltage sweep
        run_voltage_sweep_simulation(
            model,
            voltages=voltages,
            steps=args.steps,
            visualize=args.visualize,
            converge=converge_dict,
            anneal0K=args.anneal,
            p_swap=args.p_swap,
            seed=args.seed,
            quiet=args.quiet)


if __name__ == "__main__":
    main()
