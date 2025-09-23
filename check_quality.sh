#!/bin/bash

# Simple wrapper script to run data quality check on host

echo "🔍 DIODE Dataset Quality Checker"
echo "Running on host system (no Docker required)"
echo "================================================="

# Activate virtual environment
echo "🔧 Activating virtual environment 'env'..."
if [ -f "./env/bin/activate" ]; then
    source ./env/bin/activate
    echo "✅ Virtual environment activated"
elif [ -f "./env/Scripts/activate" ]; then
    source ./env/Scripts/activate
    echo "✅ Virtual environment activated (Windows)"
else
    echo "❌ Virtual environment 'env' not found in current directory"
    echo "Please ensure you have a virtual environment named 'env' with numpy, Pillow, tqdm installed"
    exit 1
fi

# Check if Python and required packages are available
python3 -c "import numpy, PIL, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required Python packages in virtual environment."
    echo "Please ensure you have: numpy, Pillow, tqdm"
    echo "Install with: pip install numpy Pillow tqdm"
    exit 1
fi

echo "✅ All required packages found in virtual environment"

# Run the quality checker
echo "🚀 Starting data quality check..."
python3 check_data_quality_host.py

echo ""
echo "📊 Data quality check completed!"
echo "📁 Check the reports in: ./data_quality_reports/"

# Deactivate virtual environment
deactivate
