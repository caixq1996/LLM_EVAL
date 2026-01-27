
import argparse
from plot_algos_curves import parse_args

# Simulate CLI arguments
import sys

def test_args(argv_list):
    # Mock sys.argv
    sys.argv = ['plot_algos_curves.py'] + argv_list
    args = parse_args()
    print(f"Args: {argv_list} => auto_y_range: {args.auto_y_range}")

print("Testing argparse for auto_y_range...")
test_args([])
test_args(['--auto_y_range'])
# test_args(['--auto_y_range', 'True']) # ArgumentError expected if store_true
