# 📊 Dataset Splitting Guide for DIODE Training

## 🎯 Overview

This guide shows you how to properly split your DIODE dataset into **train**, **validation**, and **test** sets, and use them for training and evaluation.

## 🚀 Quick Start

### 1. Create Dataset Splits
```bash
# Create splits with default ratios (70% train, 15% val, 15% test)
./dataset_manager.sh create-splits

# Custom ratios (edit script to modify)
# TRAIN_RATIO=0.8, VAL_RATIO=0.1, TEST_RATIO=0.1
```

### 2. View Split Information
```bash
./dataset_manager.sh show-splits
```

### 3. Train with Splits
```bash
# Train using the split dataset
./dataset_manager.sh train 10 8 my_experiment

# Format: train <epochs> <batch_size> <experiment_name> <workers>
./dataset_manager.sh train 20 16 production_run 0
```

### 4. Evaluate on Specific Split
```bash
# Evaluate on test split (default)
./dataset_manager.sh evaluate /outputs/my_experiment/my_experiment_best.pt

# Evaluate on validation split
./dataset_manager.sh evaluate /outputs/my_experiment/my_experiment_best.pt val
```

## 📋 Detailed Workflow

### Step 1: Understanding Dataset Splitting

**Why Split by Scenes?**
- **Prevents data leakage**: Images from the same scene don't appear in different splits
- **Better generalization**: Model learns to work on unseen environments
- **Realistic evaluation**: Tests true performance on new scenes

**Split Options:**
- **By scenes** (recommended): `deployment_1`, `deployment_2`, etc. go to different splits
- **By samples**: Individual images randomly distributed (may cause leakage)

### Step 2: Creating Splits

```bash
# Create splits (runs inside Docker container)
./dataset_manager.sh create-splits
```

**What this does:**
- Analyzes your data in `/data` (mounted from `/mnt/fast-data/monocular/data/diode-tnc`)
- Groups samples by scene (deployment folders)
- Randomly assigns scenes to train/val/test splits
- Creates text files with image and depth paths
- Saves split information as JSON

**Generated Files:**
```
dataset_splits/
├── diode_train_files.txt    # Training samples
├── diode_val_files.txt      # Validation samples  
├── diode_test_files.txt     # Test samples
└── split_info.json          # Split statistics
```

### Step 3: Training with Splits

```bash
# Basic training
./dataset_manager.sh train 5 8 test_run

# Production training
./dataset_manager.sh train 20 16 production_$(date +%Y%m%d)

# Custom parameters
./dataset_manager.sh train 15 12 custom_experiment 0
```

**What happens:**
- Uses `train_with_splits.py` instead of regular training script
- Loads training data from `diode_train_files.txt`
- Loads validation data from `diode_val_files.txt`
- Validates during training using the validation split
- Saves best model based on validation performance

### Step 4: Evaluation

```bash
# Test final performance
./dataset_manager.sh evaluate /outputs/my_experiment/my_experiment_best.pt test

# Check validation performance
./dataset_manager.sh evaluate /outputs/my_experiment/my_experiment_best.pt val

# Compare across splits
./dataset_manager.sh compare /outputs/my_experiment/my_experiment_best.pt
```

## ⚙️ Configuration Options

### Split Ratios
Edit `dataset_manager.sh` to modify:
```bash
TRAIN_RATIO=0.7    # 70% for training
VAL_RATIO=0.15     # 15% for validation  
TEST_RATIO=0.15    # 15% for testing
```

### Split Method
```bash
SPLIT_METHOD="scenes"    # Split by scenes (recommended)
SPLIT_METHOD="samples"   # Split by individual samples
```

### Random Seed
```bash
SEED=42    # For reproducible splits
```

## 📊 Understanding Your Splits

### View Split Statistics
```bash
./dataset_manager.sh show-splits
```

**Example Output:**
```json
{
  "total_samples": 69588,
  "train_samples": 48711,
  "val_samples": 10439,
  "test_samples": 10438,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15
}
```

### Validate Split Integrity
```bash
./dataset_manager.sh validate-splits
```

**Checks:**
- All split files exist
- Referenced image, depth, and mask files exist
- No corrupted file paths

## 🎯 Training Commands Comparison

### Regular Training (Original)
```bash
# Uses entire dataset for training
docker compose run --rm depth-anything train -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 --epochs=5
```

### Split-Based Training (New)
```bash
# Uses only training split, validates on validation split
./dataset_manager.sh train 5 8 my_experiment
```

## 📈 Evaluation Commands Comparison

### Regular Evaluation (Original)
```bash
# Evaluates on entire dataset
docker compose run --rm depth-anything evaluate -m zoedepth \
  --pretrained_resource="local::./checkpoints/model.pt" -d diode_outdoor
```

### Split-Based Evaluation (New)
```bash
# Evaluates on specific split (test/val)
./dataset_manager.sh evaluate /outputs/experiment/model_best.pt test
```

## 🔧 Advanced Usage

### Custom Split Creation
```bash
# Run splitter directly with custom parameters
docker compose run --rm depth-anything python /app/dataset_splitter.py \
  --data_root="/data" \
  --output_dir="/app/custom_splits" \
  --train_ratio=0.8 \
  --val_ratio=0.1 \
  --test_ratio=0.1 \
  --split_by=scenes \
  --seed=123
```

### Direct Training Script
```bash
# Use the split training script directly
docker compose run --rm depth-anything python /app/train_with_splits.py \
  -m zoedepth \
  -d diode_outdoor \
  --splits_dir="/app/dataset_splits" \
  --epochs=10 \
  --batch_size=8 \
  --workers=0
```

### Direct Evaluation Script
```bash
# Use the split evaluation script directly
docker compose run --rm depth-anything python /app/evaluate_with_splits.py \
  -m zoedepth \
  --pretrained_resource="local::/outputs/exp/model_best.pt" \
  -d diode_outdoor \
  --splits_dir="/app/dataset_splits" \
  --split=test
```

## 📋 File Structure

```
depth-anything-V1/
├── dataset_splitter.py           # Creates train/val/test splits
├── dataset_manager.sh            # Main management script
├── train_with_splits.py          # Training with split support
├── evaluate_with_splits.py       # Evaluation with split support
├── dataset_splits/               # Generated split files
│   ├── diode_train_files.txt
│   ├── diode_val_files.txt
│   ├── diode_test_files.txt
│   └── split_info.json
└── metric_depth/zoedepth/data/
    └── diode_splits.py           # Split-aware data loader
```

## 💡 Best Practices

### 1. **Always Split by Scenes**
```bash
SPLIT_METHOD="scenes"  # Prevents data leakage
```

### 2. **Reasonable Split Ratios**
```bash
# Good for large datasets (60K+ samples)
TRAIN_RATIO=0.7, VAL_RATIO=0.15, TEST_RATIO=0.15

# Good for smaller datasets 
TRAIN_RATIO=0.8, VAL_RATIO=0.1, TEST_RATIO=0.1
```

### 3. **Validate Before Training**
```bash
./dataset_manager.sh validate-splits
```

### 4. **Monitor Validation Performance**
```bash
# Use validation split during training for early stopping
./dataset_manager.sh train 20 8 experiment
```

### 5. **Test on Unseen Data**
```bash
# Final evaluation on test split only
./dataset_manager.sh evaluate /outputs/exp/model_best.pt test
```

### 6. **Reproducible Splits**
```bash
SEED=42  # Use same seed for consistent splits
```

## 🚨 Common Issues

### Split Files Not Found
```bash
# Solution: Create splits first
./dataset_manager.sh create-splits
```

### Data Leakage Warning
```bash
# Use scene-based splitting
SPLIT_METHOD="scenes"
```

### Validation Loss Not Improving
```bash
# Check if validation split is too small
./dataset_manager.sh show-splits
# Consider adjusting VAL_RATIO
```

### Memory Issues
```bash
# Reduce batch size or workers
./dataset_manager.sh train 10 4 experiment 0
```

## 🎯 Complete Example Workflow

```bash
# 1. Create splits
docker compose run --rm depth-anything /app/dataset_manager.sh create-splits

# 2. View split info
docker compose run --rm depth-anything /app/dataset_manager.sh show-splits

# 3. Validate splits
docker compose run --rm depth-anything /app/dataset_manager.sh validate-splits

# 4. Train model
docker compose run --rm depth-anything /app/dataset_manager.sh train 15 8 production_run

# 5. Evaluate on validation split
docker compose run --rm depth-anything /app/dataset_manager.sh evaluate /outputs/production_run/production_run_best.pt val

# 6. Final test evaluation
docker compose run --rm depth-anything /app/dataset_manager.sh evaluate /outputs/production_run/production_run_best.pt test

# 7. Compare performance
docker compose run --rm depth-anything /app/dataset_manager.sh compare /outputs/production_run/production_run_best.pt
```

Your dataset is now properly split for robust machine learning training and evaluation! 🚀

## 🔧 Troubleshooting

```bash
# Debug split creation
docker compose run --rm depth-anything python /app/dataset_splitter.py --help

# Check data access
docker compose run --rm depth-anything ls -la /data

# Verify split files
docker compose run --rm depth-anything head /app/dataset_splits/diode_train_files.txt
```
