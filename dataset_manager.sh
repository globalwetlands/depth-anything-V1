#!/bin/bash
# Dataset Splitting and Training Manager

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_section() { echo -e "${BLUE}==== $1 ====${NC}"; }

# Default configuration
DATA_ROOT="/data"
SPLITS_DIR="./dataset_splits"
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15
SPLIT_METHOD="scenes"
SEED=42

# Create dataset splits
create_splits() {
    print_section "Creating Dataset Splits"
    
    print_info "Parameters:"
    print_info "  Data root: $DATA_ROOT"
    print_info "  Output directory: $SPLITS_DIR"
    print_info "  Train ratio: $TRAIN_RATIO"
    print_info "  Validation ratio: $VAL_RATIO"
    print_info "  Test ratio: $TEST_RATIO"
    print_info "  Split method: $SPLIT_METHOD"
    print_info "  Random seed: $SEED"
    
    # Run inside container to access data
    docker compose run --rm depth-anything python /app/dataset_splitter.py \
        --data_root="$DATA_ROOT" \
        --output_dir="/app/$SPLITS_DIR" \
        --train_ratio="$TRAIN_RATIO" \
        --val_ratio="$VAL_RATIO" \
        --test_ratio="$TEST_RATIO" \
        --split_by="$SPLIT_METHOD" \
        --seed="$SEED"
    
    print_info "✅ Dataset splits created successfully!"
}

# Show split information
show_splits() {
    print_section "Dataset Split Information"
    
    # Check if splits info exists inside container
    if ! docker compose run --rm depth-anything test -f "/app/$SPLITS_DIR/split_info.json" 2>/dev/null; then
        print_error "Split info not found inside container. Run 'create-splits' first."
        return 1
    fi
    
    # Show split info from inside container
    echo "Split Information:"
    docker compose run --rm depth-anything cat "/app/$SPLITS_DIR/split_info.json"
    
    # Show file counts from inside container
    echo ""
    echo "File Counts:"
    for split in train val test; do
        count=$(docker compose run --rm depth-anything wc -l "/app/$SPLITS_DIR/diode_${split}_files.txt" 2>/dev/null | awk '{print $1}' | tr -d '\r')
        if [ ! -z "$count" ] && [ "$count" != "0" ]; then
            echo "  $split: $count samples"
        fi
    done
}

# Train with splits
train_with_splits() {
    local epochs=${1:-5}
    local batch_size=${2:-8}
    local experiment_name=${3:-"diode_split_$(date +%Y%m%d_%H%M%S)"}
    local workers=${4:-0}
    local resume_from=${5:-""}
    
    # Handle case where workers might contain resume_from path
    if [[ "$workers" == --resume_from=* ]]; then
        resume_from="$workers"
        workers=0
    fi
    
    print_section "Training with Dataset Splits"
    
    # Check if splits exist inside container (not on host)
    if ! docker compose run --rm depth-anything test -d "/app/$SPLITS_DIR" 2>/dev/null; then
        print_error "Splits directory not found inside container. Run 'create-splits' first."
        return 1
    fi
    
    print_info "Training parameters:"
    print_info "  Epochs: $epochs"
    print_info "  Batch size: $batch_size"
    print_info "  Experiment: $experiment_name"
    print_info "  Workers: $workers"
    print_info "  Resume from: ${resume_from:-'none'}"
    print_info "  Splits directory: $SPLITS_DIR"
    
    # Create experiment directory
    mkdir -p "training_logs/$experiment_name"
    
    # Build command with optional resume parameter
    local cmd="WANDB_MODE=disabled docker compose run --rm depth-anything python /app/train_with_splits.py \
        -m zoedepth \
        -d diode_outdoor \
        --splits_dir=\"/app/$SPLITS_DIR\" \
        --pretrained_resource=\"local::/app/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt\" \
        --workers=\"$workers\" \
        --epochs=\"$epochs\" \
        --bs=\"$batch_size\" \
        --save_dir=\"/outputs/$experiment_name\" \
        --validate_every=0.2 \
        --log_images_every=0.1 \
        --experiment_id=\"$experiment_name\""
    
    # Add resume parameter if provided
    if [[ -n "$resume_from" ]]; then
        cmd="$cmd $resume_from"
    fi
    
    # Execute command with logging
    eval "$cmd 2>&1 | tee \"training_logs/$experiment_name/training.log\""
    
    print_info "✅ Training completed!"
    print_info "Logs saved to: training_logs/$experiment_name/"
}

# Evaluate with splits
evaluate_with_splits() {
    local model_path=${1:-""}
    local split=${2:-"test"}
    
    print_section "Evaluating with Dataset Splits"
    
    # Check if splits exist inside container (not on host)
    if ! docker compose run --rm depth-anything test -d "/app/$SPLITS_DIR" 2>/dev/null; then
        print_error "Splits directory not found inside container. Run 'create-splits' first."
        return 1
    fi
    
    if [ -z "$model_path" ]; then
        print_error "Model path required. Usage: evaluate <model_path> [split]"
        print_info "Example: evaluate /outputs/my_experiment/my_experiment_best.pt test"
        return 1
    fi
    
    print_info "Evaluation parameters:"
    print_info "  Model: $model_path"
    print_info "  Split: $split"
    print_info "  Splits directory: $SPLITS_DIR"
    
    # Run evaluation
    docker compose run --rm depth-anything python /app/evaluate_with_splits.py \
        -m zoedepth \
        --pretrained_resource="local::$model_path" \
        -d diode_outdoor \
        --splits_dir="/app/$SPLITS_DIR" \
        --split="$split"
    
    print_info "✅ Evaluation completed!"
}

# Check data quality
check_data_quality() {
    print_section "Checking Data Quality"
    
    print_info "Analyzing DIODE dataset for quality issues..."
    print_info "This may take several minutes for large datasets."
    
    # Run data quality checker
    docker compose run --rm depth-anything python /app/check_data_quality.py
    
    print_info "✅ Data quality check completed!"
    print_info "Reports saved to: data_quality_reports/"
    
    # Show reports location on host
    if [ -d "./data_quality_reports" ]; then
        echo ""
        echo "📊 Quality Reports:"
        ls -la ./data_quality_reports/ | tail -n +2
    fi
}

# Compare splits (train on one, test on different splits)
compare_splits() {
    local model_path=${1:-""}
    
    print_section "Comparing Performance Across Splits"
    
    if [ -z "$model_path" ]; then
        print_error "Model path required."
        return 1
    fi
    
    print_info "Testing model: $model_path"
    
    for split in val test; do
        print_info "Evaluating on $split split..."
        evaluate_with_splits "$model_path" "$split"
        echo ""
    done
}

# Validate splits (check data integrity)
validate_splits() {
    print_section "Validating Dataset Splits"
    
    # Check if splits exist inside container (not on host)
    if ! docker compose run --rm depth-anything test -d "/app/$SPLITS_DIR" 2>/dev/null; then
        print_error "Splits directory not found inside container. Run 'create-splits' first."
        return 1
    fi
    
    # Check inside container
    docker compose run --rm depth-anything python -c "
import os
import sys
sys.path.insert(0, 'metric_depth')

splits_dir = '/app/$SPLITS_DIR'
print('Validating splits in:', splits_dir)

for split in ['train', 'val', 'test']:
    split_file = os.path.join(splits_dir, f'diode_{split}_files.txt')
    if not os.path.exists(split_file):
        print(f'❌ Missing: {split_file}')
        continue
    
    print(f'✅ Found: {split_file}')
    
    # Check first few samples
    missing_files = []
    with open(split_file, 'r') as f:
        lines = f.readlines()[:5]  # Check first 5 samples
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_path, depth_path = parts[0], parts[1]
            mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')
            
            for path in [img_path, depth_path, mask_path]:
                if not os.path.exists(path):
                    missing_files.append(path)
    
    if missing_files:
        print(f'❌ Missing files in {split} split:')
        for f in missing_files:
            print(f'   {f}')
    else:
        print(f'✅ {split} split validation passed')
        
print('Validation completed!')
"
}

# Show usage
show_help() {
    echo "Dataset Splitting and Training Manager"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  create-splits                    - Create train/val/test splits"
    echo "  show-splits                      - Show split information"
    echo "  validate-splits                  - Validate split file integrity"
    echo "  check-data-quality              - Check dataset for quality issues"
    echo "  train [epochs] [batch] [name] [workers|resume] - Train with splits"
    echo "  evaluate <model_path> [split]    - Evaluate on specific split"
    echo "  compare <model_path>             - Compare performance across splits"
    echo "  help                            - Show this help"
    echo ""
    echo "Configuration (modify script to change):"
    echo "  DATA_ROOT=$DATA_ROOT"
    echo "  SPLITS_DIR=$SPLITS_DIR"
    echo "  TRAIN_RATIO=$TRAIN_RATIO"
    echo "  VAL_RATIO=$VAL_RATIO"
    echo "  TEST_RATIO=$TEST_RATIO"
    echo "  SPLIT_METHOD=$SPLIT_METHOD"
    echo ""
    echo "Examples:"
    echo "  $0 create-splits                                    # Create dataset splits"
    echo "  $0 show-splits                                      # Show split statistics"
    echo "  $0 check-data-quality                              # Check data quality"
    echo "  $0 train 10 8 my_experiment 0                      # Train for 10 epochs"
    echo "  $0 train 120 2 resume_exp --resume_from=/outputs/exp/exp_latest.pt  # Resume training"
    echo "  $0 evaluate /outputs/exp/model_best.pt             # Evaluate on test split"
    echo "  $0 compare /outputs/exp/model_best.pt              # Test on val and test"
    echo ""
}

# Main script logic
case "${1:-help}" in
    create-splits)
        create_splits
        ;;
    show-splits)
        show_splits
        ;;
    validate-splits)
        validate_splits
        ;;
    check-data-quality)
        check_data_quality
        ;;
    train)
        train_with_splits "$2" "$3" "$4" "$5"
        ;;
    evaluate)
        evaluate_with_splits "$2" "$3"
        ;;
    compare)
        compare_splits "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
