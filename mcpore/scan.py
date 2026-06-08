"""Parameter scan generation utilities."""

import numpy as np
import itertools
import argparse
import hashlib
import sys


def generate_voltage_points():
    """20 voltage points with higher density in 0-0.1V."""
    low = np.linspace(0, 0.1, 10, endpoint=True)      # 10 points
    mid = np.linspace(0.1, 1, 5, endpoint=False)      # 5 points (0.1 already in low)
    high = np.linspace(1, 4, 5, endpoint=True)        # 5 points
    voltages = np.unique(np.concatenate([low, mid, high]))
    return voltages


def generate_scan_commands(
        replicates=5,
        converge=False,
        convergence_threshold=0.01,
        min_replicates=3,
        max_replicates=100,
        steps=1000000,
        temperature=298.0,
        output='commands.txt',
        seed_base=12345):
    """Generate GNU Parallel commands for parameter scan.

    Returns nothing; writes commands to OUTPUT file.
    """
    voltages = generate_voltage_points()           # 20 points
    radii = np.arange(5, 31, 1)                    # 26 points: 5..30 Å
    defect_probs = [0.0, 0.174, 0.25]              # 3 defect densities
    na_defect_energies = np.linspace(-1.77, 0, 5) # 5 points

    print(f"Temperature: {temperature}", file=sys.stderr)
    print(f"Voltages: {len(voltages)} points", file=sys.stderr)
    print(f"Radii: {len(radii)} points", file=sys.stderr)
    print(f"Defect probabilities: {len(defect_probs)} points", file=sys.stderr)
    print(f"Na-defect energies: {len(na_defect_energies)} points", file=sys.stderr)
    print(f"Replicates: {replicates}", file=sys.stderr)
    if converge:
        total = len(voltages) * len(radii) * len(defect_probs) * len(na_defect_energies) * max_replicates
        print(f"Total simulations (max, converge mode): {total:,}", file=sys.stderr)
    else:
        total = len(voltages) * len(radii) * len(defect_probs) * len(na_defect_energies) * replicates
        print(f"Total simulations: {total:,}", file=sys.stderr)
    est_time = total * 15 / (96 * 3600)  # 15 seconds per run, 96 cores
    print(f"Estimated wall time: {est_time:.1f} hours", file=sys.stderr)

    commands = []
    for v, r, dp, e_nd in itertools.product(voltages, radii, defect_probs, na_defect_energies):
        param_str = f"{v:.6f}_{r:.1f}_{dp:.6f}_{e_nd:.6f}"
        if converge:
            seed = hashlib.md5((param_str + f"_{seed_base}").encode()).hexdigest()
            seed_int = int(seed[:8], 16)
            cmd = (f"python mc-pore.py --voltage {v:.6f} --radius {r:.1f} "
                   f"--defect_probability {dp:.6f} --energy_na_defect {e_nd:.6f} "
                   f"--steps {steps} --csv --quiet --seed {seed_int} --temp {temperature} "
                   f"--converge --convergence_threshold {convergence_threshold} "
                   f"--min_replicates {min_replicates} --max_replicates {max_replicates}")
            commands.append(cmd)
        else:
            for rep in range(replicates):
                seed = hashlib.md5((param_str + f"_{rep}_{seed_base}").encode()).hexdigest()
                seed_int = int(seed[:8], 16)
                cmd = (f"python mc-pore.py --voltage {v:.6f} --radius {r:.1f} "
                       f"--defect_probability {dp:.6f} --energy_na_defect {e_nd:.6f} "
                       f"--steps {steps} --csv --quiet --seed {seed_int} --temp {temperature}")
                commands.append(cmd)

    with open(output, 'w') as f:
        f.write('\n'.join(commands))

    print(f"Generated {len(commands)} commands in {output}", file=sys.stderr)
    # CSV header (same order as mc-pore-scan.py output)
    header = [
        'voltage', 'radius', 'defect_probability', 'defect_placement',
        'energy_na_defect', 'energy_na_na', 'energy_na_c', 'temperature',
        'steps', 'seed', 'final_filling', 'equilibrium_reached', 'mcs',
        'n_valid_sites', 'n_surface_sites', 'default_p_gcmc', 'mu'
    ]
    print('# ' + ','.join(header), file=sys.stderr)
    print(f"# To run: cat {output} | parallel -j 96 --line-buffer > results.csv", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Generate scan commands')
    parser.add_argument('--replicates', type=int, default=5,
                        help='Number of replicates per parameter set (default: 5)')
    parser.add_argument('--converge', action='store_true',
                        help='Enable convergence loop')
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help='Relative change threshold for mean and std (default: 0.01)')
    parser.add_argument('--min_replicates', type=int, default=3,
                        help='Minimum number of replicates before checking convergence (default: 3)')
    parser.add_argument('--max_replicates', type=int, default=100,
                        help='Maximum number of replicates (default: 100)')
    parser.add_argument('--steps', type=int, default=1000000,
                        help='MC steps (default: 1e6)')
    parser.add_argument('--temperature', type=float, default=298.0,
                        help='Temperature (default: 298K)')
    parser.add_argument('--output', type=str, default='commands.txt',
                        help='Output file for commands')
    parser.add_argument('--seed_base', type=int, default=12345,
                        help='Base seed for deterministic hashing')
    args = parser.parse_args()

    generate_scan_commands(
        replicates=args.replicates,
        converge=args.converge,
        convergence_threshold=args.convergence_threshold,
        min_replicates=args.min_replicates,
        max_replicates=args.max_replicates,
        steps=args.steps,
        temperature=args.temperature,
        output=args.output,
        seed_base=args.seed_base,
    )


if __name__ == '__main__':
    main()
