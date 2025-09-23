#!/bin/bash

# Script to download required checkpoint files for Depth Anything

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

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# URLs for the checkpoint files
DEPTH_ANYTHING_URL="https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
INDOOR_MODEL_URL="https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt"
OUTDOOR_MODEL_URL="https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt"

# Function to download file
download_file() {
    local url=$1
    local filename=$2
    local description=$3
    
    if [ -f "checkpoints/$filename" ]; then
        print_info "$description already exists, skipping download"
        return 0
    fi
    
    print_info "Downloading $description..."
    if curl -L -o "checkpoints/$filename" "$url"; then
        print_info "Successfully downloaded $description"
    else
        print_error "Failed to download $description"
        return 1
    fi
}

# Download the main Depth Anything pre-trained model (required for training)
download_file "$DEPTH_ANYTHING_URL" "depth_anything_vitl14.pth" "Depth Anything pre-trained model"

# Download metric depth models (for evaluation)
print_info "Downloading metric depth models for evaluation..."
download_file "$INDOOR_MODEL_URL" "depth_anything_metric_depth_indoor.pt" "Indoor metric depth model"
download_file "$OUTDOOR_MODEL_URL" "depth_anything_metric_depth_outdoor.pt" "Outdoor metric depth model"

print_info "All checkpoint files downloaded successfully!"
print_info "You can now run training and evaluation commands."

echo ""
echo "Example commands:"
echo "  # Training"
echo "  docker compose run --rm depth-anything train -m zoedepth -d nyu --pretrained_resource=\"\""
echo "  docker compose run --rm depth-anything train -m zoedepth -d kitti --pretrained_resource=\"\""
echo ""
echo "  # Evaluation"
echo "  docker compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource=\"local::./checkpoints/depth_anything_metric_depth_indoor.pt\" -d nyu"
echo "  docker compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource=\"local::./checkpoints/depth_anything_metric_depth_outdoor.pt\" -d kitti"
