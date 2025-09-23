#!/bin/bash
# Training Monitoring and Checkpoint Management Script

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

# Create necessary directories
setup_monitoring() {
    print_section "Setting up monitoring directories"
    mkdir -p tensorboard_logs training_logs checkpoints
    print_info "Created monitoring directories"
}

# Start TensorBoard
start_tensorboard() {
    print_section "Starting TensorBoard"
    
    if docker compose ps | grep -q tensorboard; then
        print_warning "TensorBoard is already running"
        print_info "Access TensorBoard at: http://localhost:6006"
        return
    fi
    
    print_info "Starting TensorBoard service..."
    docker compose --profile monitoring up -d tensorboard
    
    # Wait a moment for TensorBoard to start
    sleep 3
    
    if docker compose ps | grep -q tensorboard; then
        print_info "✅ TensorBoard started successfully!"
        print_info "🌐 Access TensorBoard at: http://localhost:6006"
    else
        print_error "Failed to start TensorBoard"
    fi
}

# Stop TensorBoard
stop_tensorboard() {
    print_section "Stopping TensorBoard"
    docker compose --profile monitoring down
    print_info "TensorBoard stopped"
}

# Start training with optimized checkpoint settings
start_training() {
    local epochs=${1:-5}
    local batch_size=${2:-8}
    local experiment_name=${3:-"diode_outdoor_$(date +%Y%m%d_%H%M%S)"}
    
    print_section "Starting Training"
    print_info "Epochs: $epochs"
    print_info "Batch size: $batch_size" 
    print_info "Experiment: $experiment_name"
    
    # Create experiment-specific directories
    mkdir -p "training_logs/$experiment_name"
    
    print_info "Starting training with checkpoint monitoring..."
    
    # Start training with comprehensive logging
    WANDB_MODE=disabled docker compose run --rm depth-anything train \
        -m zoedepth \
        -d diode_outdoor \
        --pretrained_resource="" \
        --workers=0 \
        --epochs="$epochs" \
        --batch_size="$batch_size" \
        --save_dir="/outputs/$experiment_name" \
        --validate_every=0.2 \
        --log_images_every=0.1 \
        --experiment_id="$experiment_name" \
        2>&1 | tee "training_logs/$experiment_name/training.log"
    
    print_info "Training completed. Logs saved to: training_logs/$experiment_name/"
}

# Monitor training progress
monitor_training() {
    print_section "Training Progress Monitor"
    
    # Find latest training log
    latest_log=$(find training_logs -name "training.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$latest_log" ]; then
        print_warning "No training logs found"
        return
    fi
    
    print_info "Monitoring: $latest_log"
    print_info "Press Ctrl+C to stop monitoring"
    
    tail -f "$latest_log"
}

# Check checkpoint status
check_checkpoints() {
    print_section "Checkpoint Status"
    
    # Check local checkpoints
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        print_info "Local checkpoints:"
        ls -la checkpoints/
    else
        print_warning "No local checkpoints found"
    fi
    
    # Check output directory checkpoints
    if [ -d "/mnt/fast-data/monocular/outputs" ]; then
        print_info "Output directory experiments:"
        ls -la /mnt/fast-data/monocular/outputs/
    fi
    
    # Check for latest experiments
    if find /mnt/fast-data/monocular/outputs -name "*.pt" -type f 2>/dev/null | head -5; then
        print_info "Recent checkpoint files:"
        find /mnt/fast-data/monocular/outputs -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -5 | while read timestamp file; do
            echo "  $file"
        done
    fi
}

# Backup checkpoints
backup_checkpoints() {
    local backup_dir="checkpoint_backups/$(date +%Y%m%d_%H%M%S)"
    
    print_section "Backing up Checkpoints"
    mkdir -p "$backup_dir"
    
    # Backup from outputs
    if [ -d "/mnt/fast-data/monocular/outputs" ]; then
        cp -r /mnt/fast-data/monocular/outputs/* "$backup_dir/" 2>/dev/null || true
    fi
    
    # Backup local checkpoints
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        cp -r checkpoints/* "$backup_dir/" 2>/dev/null || true
    fi
    
    print_info "Checkpoints backed up to: $backup_dir"
}

# Show help
show_help() {
    echo "Training Monitor and Checkpoint Manager"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                    - Setup monitoring directories"
    echo "  start-tensorboard        - Start TensorBoard service"
    echo "  stop-tensorboard         - Stop TensorBoard service"
    echo "  train [epochs] [batch] [name] - Start training with monitoring"
    echo "  monitor                  - Monitor latest training progress"
    echo "  checkpoints              - Check checkpoint status"
    echo "  backup                   - Backup all checkpoints"
    echo "  help                     - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup                           # Setup directories"
    echo "  $0 start-tensorboard               # Start TensorBoard"
    echo "  $0 train 10 8 my_experiment        # Train for 10 epochs, batch size 8"
    echo "  $0 monitor                         # Monitor training progress"
    echo "  $0 checkpoints                     # Check saved models"
    echo ""
    echo "TensorBoard URL: http://localhost:6006"
}

# Main script logic
case "${1:-help}" in
    setup)
        setup_monitoring
        ;;
    start-tensorboard)
        start_tensorboard
        ;;
    stop-tensorboard)
        stop_tensorboard
        ;;
    train)
        setup_monitoring
        start_training "$2" "$3" "$4"
        ;;
    monitor)
        monitor_training
        ;;
    checkpoints)
        check_checkpoints
        ;;
    backup)
        backup_checkpoints
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
