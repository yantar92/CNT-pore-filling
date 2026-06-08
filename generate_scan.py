#!/usr/bin/env python
"""
Generate GNU Parallel commands for parameter scan.

Backward-compatible wrapper -- delegates to mcpore package.
Install the package first:  pip install -e .

Usage:
    python generate_scan.py > commands.txt
    cat commands.txt | parallel -j 96 --line-buffer > results.csv
"""

from mcpore.scan import main

if __name__ == '__main__':
    main()
