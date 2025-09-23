# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Set environment variables
ENV PYTHONPATH=/app
ENV PYOPENGL_PLATFORM=egl
ENV WANDB_START_METHOD=thread
ENV WANDB_MODE=disabled
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY metric_depth/environment.yml .

# Check Python version and install compatible packages
RUN python --version

# Install Python dependencies
# First install the base requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies needed for metric depth training/evaluation
RUN pip install --no-cache-dir \
    "numpy<2" \
    h5py \
    matplotlib \
    scipy \
    timm \
    tqdm \
    wandb \
    easydict \
    open3d \
    plyfile \
    imageio \
    imageio-ffmpeg \
    tensorboard \
    opencv-python-headless

# Copy the entire project
COPY . .

# Create directories for data and outputs
RUN mkdir -p /data /outputs

# Create entrypoint script
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
if [ "$1" = "train" ]; then
    shift
    cd /app/metric_depth
    python train_mono.py "$@"
elif [ "$1" = "evaluate" ]; then
    shift
    cd /app/metric_depth
    python evaluate.py "$@"
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    shift
    exec "$@"
elif [ "$1" = "ls" ] || [ "$1" = "cat" ] || [ "$1" = "find" ] || [ "$1" = "head" ] || [ "$1" = "tail" ] || [ "$1" = "test" ] || [ "$1" = "wc" ] || [ "$1" = "mkdir" ]; then
    exec "$@"
elif [ "$1" = "python" ]; then
    shift
    exec python "$@"
else
    echo "Usage: docker run <image> train <args> or docker run <image> evaluate <args>"
    echo "Example: docker run <image> train -m zoedepth -d nyu --pretrained_resource=\"\""
    echo "Example: docker run <image> evaluate -m zoedepth --pretrained_resource=\"local::./checkpoints/depth_anything_metric_depth_indoor.pt\" -d nyu"
fi
EOF
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["train"]