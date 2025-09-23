#!/usr/bin/env python3
"""
Debug script to check model output shapes
"""

import sys
sys.path.insert(0, '/app/metric_depth')

import torch
import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.data.diode_splits import get_diode_train_loader

def debug_model_and_data():
    print("=== Debugging Model and Data Shapes ===")
    
    # Load config and model
    config = get_config("zoedepth", "train", "diode_outdoor", 
                       pretrained_resource="local::/app/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt")
    config.splits_dir = "/app/dataset_splits"
    
    print(f"Config input image size: {config.img_size}")
    print(f"Config input height: {config.input_height}")
    print(f"Config input width: {config.input_width}")
    
    # Build model
    model = build_model(config)
    model.cuda()
    model.eval()
    
    print(f"Model built successfully")
    
    # Load some data
    train_loader = get_diode_train_loader(config.splits_dir, batch_size=1, num_workers=0)
    
    print(f"Data loader created")
    
    # Get a sample
    for batch in train_loader:
        print(f"\n=== Sample Data ===")
        image = batch['image']
        depth = batch['depth'] 
        mask = batch['mask']
        
        print(f"Image shape: {image.shape}")
        print(f"Depth shape: {depth.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image min/max: {image.min():.3f}/{image.max():.3f}")
        print(f"Depth min/max: {depth.min():.3f}/{depth.max():.3f}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        # Test model forward pass
        print(f"\n=== Model Forward Pass ===")
        image_cuda = image.cuda()
        
        with torch.no_grad():
            try:
                output = model(image_cuda)
                print(f"Model output keys: {list(output.keys())}")
                
                if 'metric_depth' in output:
                    pred_depth = output['metric_depth']
                    print(f"Predicted depth shape: {pred_depth.shape}")
                    print(f"Predicted depth min/max: {pred_depth.min():.3f}/{pred_depth.max():.3f}")
                else:
                    print("No 'metric_depth' in output!")
                    
            except Exception as e:
                print(f"Model forward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        break  # Only test first sample
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_model_and_data()
