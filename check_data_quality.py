#!/usr/bin/env python3
"""
Simple wrapper for data quality checking inside Docker container
"""

import sys
import os

# Add paths for the container environment
sys.path.insert(0, '/app/metric_depth')

# Import the data quality checker
from data_quality_checker import DataQualityChecker

def main():
    print("🔍 DIODE Dataset Quality Checker")
    print("=" * 50)
    
    # Use container paths
    data_root = "/data"
    splits_dir = "/app/dataset_splits"
    output_dir = "/app/data_quality_reports"
    
    print(f"Data root: {data_root}")
    print(f"Splits directory: {splits_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if paths exist
    if not os.path.exists(data_root):
        print(f"❌ Error: Data root not found: {data_root}")
        print("Make sure your data is mounted at /data")
        return 1
    
    if not os.path.exists(splits_dir):
        print(f"⚠️  Warning: Splits directory not found: {splits_dir}")
        print("Will scan entire data directory instead")
        splits_dir = None
    
    # Create checker
    checker = DataQualityChecker(data_root, splits_dir, output_dir)
    
    # Run quality check
    try:
        checker.run_quality_check()
        return 0
    except Exception as e:
        print(f"❌ Error during quality check: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
