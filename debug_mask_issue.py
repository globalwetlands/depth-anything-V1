#!/usr/bin/env python3
"""
Debug script to investigate the empty mask issue
"""

import sys
sys.path.insert(0, '/app/metric_depth')

import torch
import numpy as np
from zoedepth.data.diode_splits import get_diode_train_loader

def debug_mask_issue():
    print("=== Debugging Empty Mask Issue ===")
    
    # Load data loader
    splits_dir = "/app/dataset_splits"
    train_loader = get_diode_train_loader(splits_dir, batch_size=2, num_workers=0)
    
    print(f"Data loader created with batch size 2")
    
    # Check several batches
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Check first 5 batches
            break
            
        print(f"\n=== Batch {batch_idx + 1} ===")
        image = batch['image']
        depth = batch['depth'] 
        mask = batch['mask']
        
        print(f"Batch shapes:")
        print(f"  Image: {image.shape}")
        print(f"  Depth: {depth.shape}")
        print(f"  Mask: {mask.shape}")
        
        # Check mask statistics
        for i in range(image.shape[0]):  # For each sample in batch
            sample_mask = mask[i]
            sample_depth = depth[i]
            
            valid_pixels = torch.sum(sample_mask > 0).item()
            total_pixels = sample_mask.numel()
            valid_ratio = valid_pixels / total_pixels
            
            print(f"  Sample {i+1}:")
            print(f"    Valid pixels: {valid_pixels}/{total_pixels} ({valid_ratio:.1%})")
            print(f"    Mask range: {sample_mask.min().item():.3f} to {sample_mask.max().item():.3f}")
            print(f"    Mask unique values: {torch.unique(sample_mask)}")
            print(f"    Depth range: {sample_depth.min().item():.3f} to {sample_depth.max().item():.3f}")
            
            if valid_pixels == 0:
                print(f"    ⚠️  WARNING: Sample {i+1} has NO valid pixels!")
            elif valid_pixels < 1000:
                print(f"    ⚠️  WARNING: Sample {i+1} has very few valid pixels!")
        
        # Test what happens during loss computation simulation
        print(f"\n  Simulating loss computation...")
        
        # Simulate the model output (same size as resized image)
        model_output_shape = (image.shape[0], 1, image.shape[2], image.shape[3])  # [B, 1, H, W]
        fake_model_output = torch.randn(model_output_shape) + 2.0  # Positive depths
        
        print(f"  Fake model output shape: {fake_model_output.shape}")
        
        # Simulate the interpolation that happens in loss function
        if fake_model_output.shape[-1] != depth.shape[-1]:
            print(f"  Interpolating model output from {fake_model_output.shape[-2:]} to {depth.shape[-2:]}")
            interpolated_output = torch.nn.functional.interpolate(
                fake_model_output, size=depth.shape[-2:], mode='bilinear', align_corners=True)
            print(f"  Interpolated shape: {interpolated_output.shape}")
        else:
            interpolated_output = fake_model_output
        
        # Apply mask as done in loss function
        mask_bool = mask.to(torch.bool)
        for i in range(image.shape[0]):
            masked_pred = interpolated_output[i][mask_bool[i]]
            masked_target = depth[i][mask_bool[i]]
            
            print(f"  Sample {i+1} after masking:")
            print(f"    Masked prediction size: {masked_pred.shape}")
            print(f"    Masked target size: {masked_target.shape}")
            
            if masked_pred.numel() == 0:
                print(f"    ❌ PROBLEM: Empty tensor after masking!")
            else:
                print(f"    ✅ OK: {masked_pred.numel()} valid pixels for loss computation")
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_mask_issue()
