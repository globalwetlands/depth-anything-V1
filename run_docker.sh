#!/bin/bash

# Helper script for running Depth Anything Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose is not installed. You can still use Docker directly."
fi

# Function to build the image
build_image() {
    print_info "Building Docker image..."
    if command -v docker-compose &> /dev/null; then
        docker-compose build
    else
        docker build -t depth-anything .
    fi
    print_info "Image built successfully!"
}

# Function to run training
run_training() {
    local dataset=${1:-nyu}
    local model=${2:-zoedepth}
    local pretrained=${3:-""}
    
    print_info "Starting training with dataset: $dataset, model: $model"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose run --rm depth-anything train -m $model -d $dataset --pretrained_resource="$pretrained"
    else
        docker run --rm --gpus all \
            -v /mnt/fast-data/monocular/data/diode-tnc:/data:ro \
            -v /mnt/fast-data/monocular/outputs:/outputs \
            -v $(pwd)/checkpoints:/app/metric_depth/checkpoints \
            depth-anything train -m $model -d $dataset --pretrained_resource="$pretrained"
    fi
}

# Function to run evaluation
run_evaluation() {
    local dataset=${1:-nyu}
    local model=${2:-zoedepth}
    local checkpoint=${3:-"local::./checkpoints/depth_anything_metric_depth_indoor.pt"}
    
    print_info "Starting evaluation with dataset: $dataset, model: $model"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose run --rm depth-anything evaluate -m $model --pretrained_resource="$checkpoint" -d $dataset
    else
        docker run --rm --gpus all \
            -v /mnt/fast-data/monocular/data/diode-tnc:/data:ro \
            -v /mnt/fast-data/monocular/outputs:/outputs \
            -v $(pwd)/checkpoints:/app/metric_depth/checkpoints \
            depth-anything evaluate -m $model --pretrained_resource="$checkpoint" -d $dataset
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                    Build the Docker image"
    echo "  train [dataset] [model]  Run training (default: nyu, zoedepth)"
    echo "  evaluate [dataset]       Run evaluation (default: nyu)"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 train nyu"
    echo "  $0 train kitti"
    echo "  $0 evaluate nyu"
    echo "  $0 evaluate kitti"
    echo ""
    echo "Available datasets:"
    echo "  Training: nyu, kitti"
    echo "  Evaluation: nyu, sunrgbd, ibims, hypersim_test, kitti, vkitti2, diode_outdoor"
}

# Main script logic
case "${1:-help}" in
    "build")
        build_image
        ;;
    "train")
        run_training "${2:-nyu}" "${3:-zoedepth}" "${4:-""}"
        ;;
    "evaluate")
        run_evaluation "${2:-nyu}" "${3:-zoedepth}" "${4:-"local::./checkpoints/depth_anything_metric_depth_indoor.pt"}"
        ;;
    "help"|*)
        show_usage
        ;;
esac
