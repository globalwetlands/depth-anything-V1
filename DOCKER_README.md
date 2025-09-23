# Docker Setup for Depth Anything Metric Depth Training

This Docker setup allows you to run the Depth Anything metric depth training and evaluation scripts in a containerized environment with GPU support.

## Prerequisites

1. **Docker** with NVIDIA Container Runtime support
2. **Docker Compose** (optional, for easier management)
3. **NVIDIA GPU** with proper drivers installed

## Quick Start

### Using Docker Compose (Recommended)

1. **Build the image:**
   ```bash
   docker-compose build
   ```

2. **Run training:**
   ```bash
   docker-compose run --rm depth-anything train -m zoedepth -d nyu --pretrained_resource=""
   ```

3. **Run evaluation:**
   ```bash
   docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d nyu
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t depth-anything .
   ```

2. **Run training:**
   ```bash
   docker run --rm --gpus all \
     -v /mnt/fast-data/monocular/data/diode-tnc:/data:ro \
     -v /mnt/fast-data/monocular/outputs:/outputs \
     -v $(pwd)/checkpoints:/app/metric_depth/checkpoints \
     depth-anything train -m zoedepth -d nyu --pretrained_resource=""
   ```

3. **Run evaluation:**
   ```bash
   docker run --rm --gpus all \
     -v /mnt/fast-data/monocular/data/diode-tnc:/data:ro \
     -v /mnt/fast-data/monocular/outputs:/outputs \
     -v $(pwd)/checkpoints:/app/metric_depth/checkpoints \
     depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d nyu
   ```

## Volume Mounts

The Docker setup includes the following volume mounts:

- **Data directory**: `/mnt/fast-data/monocular/data/diode-tnc` → `/data` (read-only)
- **Outputs directory**: `/mnt/fast-data/monocular/outputs` → `/outputs` (read-write)
- **Checkpoints directory**: `./checkpoints` → `/app/metric_depth/checkpoints` (read-write)

## Training Examples

### Indoor Training (NYUv2)
```bash
docker-compose run --rm depth-anything train -m zoedepth -d nyu --pretrained_resource=""
```

### Outdoor Training (KITTI)
```bash
docker-compose run --rm depth-anything train -m zoedepth -d kitti --pretrained_resource=""
```

## Evaluation Examples

### Indoor Evaluation
```bash
# NYUv2
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d nyu

# SUN RGB-D
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d sunrgbd

# iBims-1
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d ibims

# HyperSim
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d hypersim_test
```

### Outdoor Evaluation
```bash
# KITTI
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d kitti

# Virtual KITTI 2
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d vkitti2

# DIODE Outdoor
docker-compose run --rm depth-anything evaluate -m zoedepth --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_outdoor.pt" -d diode_outdoor
```

## Data Preparation

Before running training or evaluation, ensure your data is properly organized in the mounted data directory. The expected structure should follow the ZoeDepth data format.

## Checkpoints

Download the required checkpoints and place them in the `./checkpoints` directory:

1. **Depth Anything pre-trained model** (for training initialization):
   - Download from: https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth

2. **Metric depth models** (for evaluation):
   - Indoor model: https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
   - Outdoor model: https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt

## Troubleshooting

### GPU Issues
- Ensure NVIDIA Container Runtime is properly installed
- Verify GPU is visible: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`

### Permission Issues
- Ensure the mounted directories have proper permissions
- You may need to run Docker with `--user` flag if there are permission conflicts

### Memory Issues
- Monitor GPU memory usage during training
- Consider reducing batch size if running out of memory

## Customization

You can modify the `docker-compose.yml` file to:
- Change volume mount paths
- Add additional environment variables
- Modify resource limits
- Add additional services

## Environment Variables

The container sets the following environment variables:
- `PYTHONPATH=/app`: Ensures Python can find the project modules
- `PYOPENGL_PLATFORM=egl`: Required for OpenGL operations
- `WANDB_START_METHOD=thread`: Prevents issues with Weights & Biases
- `DEBIAN_FRONTEND=noninteractive`: Prevents interactive prompts during package installation
