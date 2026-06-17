#!/usr/bin/env python
"""
Metropolis Monte Carlo Simulation for Hard Carbon Pore Filling.

Backward-compatible wrapper -- delegates to mcpore package.
Install the package first:  pip install -e .

Command line usage:
    python mc-pore.py [--voltage 0.1 [0.1 ...]] [--radius 10.0] [--file snapshots.pkl]
        [--steps 20000] [--visualize] [--energy_na_defect -1.53]
        [--temp 298] [--defect_placement surface] [--defect_probability 0.174]
        [--csv] [--quiet] [--seed INT] [--converge] [--convergence_threshold 0.05]
        [--min_replicates 3] [--max_replicates 50]
"""

from mcpore.cli import main

if __name__ == "__main__":
    main()
