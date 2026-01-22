#!/usr/bin/env bash
# 1. Save the two scripts as mc-pore-scan.py and generate_scan.py
# 2. Install dependencies (if not already installed):
#    pip install numpy matplotlib pandas

# 3. Generate the command list (takes a few seconds):
python generate_scan.py --replicates 15 --steps 1000000 --output commands.txt

# 4. Run with GNU Parallel on 96 cores:
cat commands.txt | parallel -j 96 --line-buffer > results.csv

# 5. After all jobs finish, you will have a single CSV file `results.csv`
#    with one row per simulation. You can load it with pandas for analysis.
