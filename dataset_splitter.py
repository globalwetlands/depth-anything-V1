#!/usr/bin/env python3
"""
DIODE Dataset Splitter for Train/Validation/Test splits
"""

import os
import glob
import random
import argparse
from pathlib import Path
import json

def get_all_samples(data_root):
    """Get all image samples from DIODE dataset"""
    # Find all PNG images in the dataset
    image_files = glob.glob(os.path.join(data_root, '*', '*', '*.png'))
    
    samples = []
    for img_path in image_files:
        # Check if corresponding depth and mask files exist
        depth_path = img_path.replace('.png', '_depth.npy')
        mask_path = img_path.replace('.png', '_depth_mask.npy')
        
        if os.path.exists(depth_path) and os.path.exists(mask_path):
            # Store relative paths from data_root
            rel_img_path = os.path.relpath(img_path, data_root)
            samples.append(rel_img_path)
    
    return sorted(samples)

def split_by_scenes(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset by scenes to avoid data leakage"""
    # Group samples by scene (first directory level)
    scenes = {}
    for sample in samples:
        scene = sample.split(os.sep)[0]  # deployment_X
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(sample)
    
    print(f"Found {len(scenes)} scenes with {len(samples)} total samples")
    
    # Get scene names and shuffle them
    scene_names = list(scenes.keys())
    random.shuffle(scene_names)
    
    # Calculate split indices
    n_scenes = len(scene_names)
    train_end = int(n_scenes * train_ratio)
    val_end = train_end + int(n_scenes * val_ratio)
    
    # Split scenes
    train_scenes = scene_names[:train_end]
    val_scenes = scene_names[train_end:val_end]
    test_scenes = scene_names[val_end:]
    
    # Collect samples for each split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for scene in train_scenes:
        train_samples.extend(scenes[scene])
    
    for scene in val_scenes:
        val_samples.extend(scenes[scene])
    
    for scene in test_scenes:
        test_samples.extend(scenes[scene])
    
    return train_samples, val_samples, test_samples

def split_by_samples(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset by individual samples (may cause data leakage)"""
    random.shuffle(samples)
    
    n_samples = len(samples)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    return train_samples, val_samples, test_samples

def save_split_files(train_samples, val_samples, test_samples, output_dir, data_root):
    """Save split files in the format expected by the data loader"""
    os.makedirs(output_dir, exist_ok=True)
    
    def save_split(samples, split_name):
        filepath = os.path.join(output_dir, f'diode_{split_name}_files.txt')
        with open(filepath, 'w') as f:
            for sample in samples:
                # Convert to format expected by data loader
                # Format: rgb_path depth_path
                img_path = os.path.join(data_root, sample)
                depth_path = img_path.replace('.png', '_depth.npy')
                f.write(f"{img_path} {depth_path}\n")
        
        print(f"Saved {len(samples)} samples to {filepath}")
        return filepath
    
    train_file = save_split(train_samples, 'train')
    val_file = save_split(val_samples, 'val')
    test_file = save_split(test_samples, 'test')
    
    # Save split info
    split_info = {
        'data_root': data_root,
        'total_samples': len(train_samples) + len(val_samples) + len(test_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_file': train_file,
        'val_file': val_file,
        'test_file': test_file,
        'train_ratio': len(train_samples) / (len(train_samples) + len(val_samples) + len(test_samples)),
        'val_ratio': len(val_samples) / (len(train_samples) + len(val_samples) + len(test_samples)),
        'test_ratio': len(test_samples) / (len(train_samples) + len(val_samples) + len(test_samples))
    }
    
    info_file = os.path.join(output_dir, 'split_info.json')
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split information saved to {info_file}")
    return split_info

def create_diode_splits(data_root, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                       split_by='scenes', seed=42):
    """Create train/val/test splits for DIODE dataset"""
    
    print(f"Creating DIODE dataset splits...")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    print(f"Split method: {split_by}")
    print(f"Random seed: {seed}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all samples
    samples = get_all_samples(data_root)
    if not samples:
        raise ValueError(f"No valid samples found in {data_root}")
    
    print(f"Found {len(samples)} valid samples")
    
    # Split the dataset
    if split_by == 'scenes':
        train_samples, val_samples, test_samples = split_by_scenes(
            samples, train_ratio, val_ratio, test_ratio)
    else:
        train_samples, val_samples, test_samples = split_by_samples(
            samples, train_ratio, val_ratio, test_ratio)
    
    # Save split files
    split_info = save_split_files(train_samples, val_samples, test_samples, output_dir, data_root)
    
    print("\n✅ Dataset splitting completed!")
    print(f"Train: {len(train_samples)} samples ({split_info['train_ratio']:.1%})")
    print(f"Val:   {len(val_samples)} samples ({split_info['val_ratio']:.1%})")
    print(f"Test:  {len(test_samples)} samples ({split_info['test_ratio']:.1%})")
    
    return split_info

def main():
    parser = argparse.ArgumentParser(description='Split DIODE dataset into train/val/test')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of DIODE dataset')
    parser.add_argument('--output_dir', type=str, default='./dataset_splits',
                       help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--split_by', choices=['scenes', 'samples'], default='scenes',
                       help='Split by scenes (recommended) or individual samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create splits
    create_diode_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by=args.split_by,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
