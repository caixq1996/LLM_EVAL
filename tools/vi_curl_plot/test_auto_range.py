
import pandas as pd
import numpy as np
from plot_algos_curves import compute_dataset_y_ranges, METRIC_DATASET_Y_RANGES

# Mock data
df = pd.DataFrame({
    'base': ['base1'] * 2,
    'metric': ['pass@1'] * 2,
    'dataset': ['aime24x8'] * 2,
    'value': [50.0, 60.0],
    'std': [1.0, 1.0]
})

print(f"METRIC_DATASET_Y_RANGES['pass@1']['aime24x8'] = {METRIC_DATASET_Y_RANGES['pass@1']['aime24x8']}")

# Test 1: Default (manual override expected since aime24x8 has entry)
ranges_default = compute_dataset_y_ranges(df, 'base1', 'pass@1', auto_y_range_flag=False)
print(f"Default (flag=False): {ranges_default['aime24x8']}")

# Test 2: Auto Y Range (should ignore manual and use data)
ranges_auto = compute_dataset_y_ranges(df, 'base1', 'pass@1', auto_y_range_flag=True)
print(f"Auto (flag=True): {ranges_auto['aime24x8']}")

# Check if they differ
if ranges_default['aime24x8'] == (0.0, 20.0) and ranges_auto['aime24x8'] != (0.0, 20.0):
    print("SUCCESS: Logic works as expected.")
else:
    print("FAILURE: Logic did not produce expected difference.")
