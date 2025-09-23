#!/usr/bin/env python3

import sys
sys.path.insert(0, '/app/metric_depth')

import numpy as np
import torch
from zoedepth.data.diode_splits import ToTensorSplits

# Test the transform pipeline step by step
splits_dir = "/app/dataset_splits"
split_file = f"{splits_dir}/diode_train_files.txt"

# Read first sample from split file
with open(split_file, 'r') as f:
    line = f.readline().strip()
    parts = line.split()
    img_path, depth_path = parts[0], parts[1]
    mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')

print(f"Paths:")
print(f"  Image: {img_path}")
print(f"  Depth: {depth_path}")
print(f"  Mask: {mask_path}")

# Load raw data
from PIL import Image
image = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
depth = np.load(depth_path)
valid = np.load(mask_path)

print(f"\nRaw data shapes:")
print(f"  Image: {image.shape}")
print(f"  Depth: {depth.shape} (min: {depth.min():.3f}, max: {depth.max():.3f})")
print(f"  Valid: {valid.shape}")

# Only add channel dimension to mask (depth already has it)
valid = valid[..., None]

print(f"\nAfter adding channel dimension:")
print(f"  Image: {image.shape}")
print(f"  Depth: {depth.shape}")
print(f"  Valid: {valid.shape}")

# Apply transforms
transform = ToTensorSplits()

sample = dict(image=image, depth=depth, mask=valid)
print(f"\nBefore transform:")
for key, val in sample.items():
    print(f"  {key}: {val.shape} ({type(val).__name__})")

# Apply to_tensor to each
image_tensor = transform.to_tensor(image)
depth_tensor = transform.to_tensor(depth)
mask_tensor = transform.to_tensor(valid)

print(f"\nAfter to_tensor:")
print(f"  Image: {image_tensor.shape}")
print(f"  Depth: {depth_tensor.shape}")
print(f"  Mask: {mask_tensor.shape}")

# Apply full transform
sample_transformed = transform(sample)
print(f"\nAfter full transform:")
for key, val in sample_transformed.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: {val.shape}")
    else:
        print(f"  {key}: {val}")
