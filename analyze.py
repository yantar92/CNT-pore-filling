#!/usr/bin/env python3
"""
Analyze MC pore filling parameter scan results.

Backward-compatible wrapper -- delegates to mcpore package.
Install the package first:  pip install -e .

Usage:
    python analyze.py [results.csv] [--output-dir ./] [--show-plots] [--quiet]
"""

from mcpore.analysis import analyze_pore_filling
import argparse
import sys


def main():
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

