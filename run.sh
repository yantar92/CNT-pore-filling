#!/usr/bin/env bash
# 1. Install the package in development mode:
#    pip install -e .

# 2. Generate the command list (takes a few seconds):
python generate_scan.py --replicates 15 --steps 1000000 --output commands.txt

# 3. Run with GNU Parallel on 96 cores:
cat commands.txt | parallel -j 96 --line-buffer > results.csv

# 4. After all jobs finish, you will have a single CSV file `results.csv`
#    with one row per simulation. Analyze with:
#    python analyze.py results.csv --output-dir ./plots
