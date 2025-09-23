#!/usr/bin/env python3
"""
Comprehensive Data Quality Checker for DIODE Dataset

This script verifies the quality of the DIODE dataset and flags problematic samples.
Issues checked:
- Negative depth values
- Empty or invalid masks
- Corrupted files
- Extreme depth values
- Missing files
- Dimension mismatches
- Data type issues
"""

import os
import sys
import numpy as np
import json
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
import traceback

class DataQualityChecker:
    def __init__(self, data_root, splits_dir=None, output_dir="./data_quality_reports"):
        self.data_root = data_root
        self.splits_dir = splits_dir
        self.output_dir = output_dir
        self.issues = {
            'negative_depths': [],
            'zero_depths': [],
            'extreme_depths': [],
            'empty_masks': [],
            'sparse_masks': [],
            'corrupted_files': [],
            'missing_files': [],
            'dimension_mismatches': [],
            'dtype_issues': [],
            'file_read_errors': []
        }
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'problematic_samples': 0,
            'depth_stats': {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0},
            'mask_stats': {'min_valid_ratio': 1.0, 'max_valid_ratio': 0.0, 'mean_valid_ratio': 0.0}
        }
        
        # Thresholds for quality checks
        self.thresholds = {
            'max_depth': 100.0,  # Maximum reasonable depth in meters
            'min_depth': 0.001,  # Minimum reasonable depth in meters
            'min_mask_ratio': 0.01,  # Minimum 1% valid pixels required
            'extreme_depth_ratio': 0.05,  # Flag if >5% of pixels have extreme depths
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_sample_paths(self):
        """Load all sample paths from splits or scan directory"""
        samples = []
        
        if self.splits_dir and os.path.exists(self.splits_dir):
            # Load from split files
            for split in ['train', 'val', 'test']:
                split_file = os.path.join(self.splits_dir, f'diode_{split}_files.txt')
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                img_path = parts[0]
                                depth_path = parts[1]
                                mask_path = depth_path.replace('_depth.npy', '_depth_mask.npy')
                                samples.append({
                                    'image': img_path,
                                    'depth': depth_path,
                                    'mask': mask_path,
                                    'split': split
                                })
        else:
            # Scan directory for all samples
            import glob
            image_files = glob.glob(os.path.join(self.data_root, '*', '*', '*.png'))
            for img_path in image_files:
                depth_path = img_path.replace('.png', '_depth.npy')
                mask_path = img_path.replace('.png', '_depth_mask.npy')
                if os.path.exists(depth_path) and os.path.exists(mask_path):
                    samples.append({
                        'image': img_path,
                        'depth': depth_path,
                        'mask': mask_path,
                        'split': 'unknown'
                    })
        
        return samples
    
    def check_file_existence(self, sample):
        """Check if all required files exist"""
        issues = []
        for key, path in sample.items():
            if key == 'split':
                continue
            if not os.path.exists(path):
                issues.append(f"Missing {key}: {path}")
        return issues
    
    def check_image_quality(self, image_path):
        """Check image file quality"""
        issues = []
        try:
            with Image.open(image_path) as img:
                # Check if image can be loaded
                img.verify()
                
            # Reload to get actual data
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Check dimensions
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                issues.append(f"Invalid image dimensions: {img_array.shape}")
            
            # Check data range
            if img_array.max() > 255 or img_array.min() < 0:
                issues.append(f"Invalid pixel values: [{img_array.min()}, {img_array.max()}]")
                
        except Exception as e:
            issues.append(f"Image read error: {str(e)}")
        
        return issues
    
    def check_depth_quality(self, depth_path):
        """Check depth file quality"""
        issues = []
        stats = {}
        
        try:
            depth = np.load(depth_path)
            
            # Check shape
            if len(depth.shape) not in [2, 3]:
                issues.append(f"Invalid depth shape: {depth.shape}")
                return issues, stats
            
            # Flatten depth for analysis
            if len(depth.shape) == 3:
                depth_flat = depth.squeeze()
            else:
                depth_flat = depth
            
            # Basic statistics
            stats['shape'] = depth.shape
            stats['min'] = float(depth_flat.min())
            stats['max'] = float(depth_flat.max())
            stats['mean'] = float(depth_flat.mean())
            stats['std'] = float(depth_flat.std())
            
            # Check for negative values
            negative_count = np.sum(depth_flat < 0)
            if negative_count > 0:
                negative_ratio = negative_count / depth_flat.size
                issues.append(f"Negative depths: {negative_count} pixels ({negative_ratio:.2%})")
                stats['negative_ratio'] = negative_ratio
            
            # Check for zero values
            zero_count = np.sum(depth_flat == 0)
            if zero_count > 0:
                zero_ratio = zero_count / depth_flat.size
                issues.append(f"Zero depths: {zero_count} pixels ({zero_ratio:.2%})")
                stats['zero_ratio'] = zero_ratio
            
            # Check for extreme values
            extreme_large = np.sum(depth_flat > self.thresholds['max_depth'])
            extreme_small = np.sum((depth_flat > 0) & (depth_flat < self.thresholds['min_depth']))
            
            if extreme_large > 0:
                extreme_ratio = extreme_large / depth_flat.size
                if extreme_ratio > self.thresholds['extreme_depth_ratio']:
                    issues.append(f"Extreme large depths: {extreme_large} pixels ({extreme_ratio:.2%}) > {self.thresholds['max_depth']}m")
            
            if extreme_small > 0:
                extreme_ratio = extreme_small / depth_flat.size
                if extreme_ratio > self.thresholds['extreme_depth_ratio']:
                    issues.append(f"Extreme small depths: {extreme_small} pixels ({extreme_ratio:.2%}) < {self.thresholds['min_depth']}m")
            
            # Check data type
            if depth.dtype not in [np.float32, np.float64]:
                issues.append(f"Unexpected depth dtype: {depth.dtype}")
            
        except Exception as e:
            issues.append(f"Depth read error: {str(e)}")
            stats['error'] = str(e)
        
        return issues, stats
    
    def check_mask_quality(self, mask_path):
        """Check mask file quality"""
        issues = []
        stats = {}
        
        try:
            mask = np.load(mask_path)
            
            # Check shape
            if len(mask.shape) != 2:
                issues.append(f"Invalid mask shape: {mask.shape}")
                return issues, stats
            
            # Basic statistics
            stats['shape'] = mask.shape
            unique_vals = np.unique(mask)
            stats['unique_values'] = unique_vals.tolist()
            
            # Check if binary
            if not np.all(np.isin(unique_vals, [0, 1])):
                issues.append(f"Non-binary mask values: {unique_vals}")
            
            # Check valid pixel ratio
            valid_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            valid_ratio = valid_pixels / total_pixels
            stats['valid_ratio'] = valid_ratio
            
            if valid_ratio == 0:
                issues.append("Empty mask: no valid pixels")
            elif valid_ratio < self.thresholds['min_mask_ratio']:
                issues.append(f"Sparse mask: only {valid_ratio:.2%} valid pixels")
            
            # Check data type
            if mask.dtype not in [np.uint8, np.bool_, np.int8, np.int16, np.int32]:
                issues.append(f"Unexpected mask dtype: {mask.dtype}")
            
        except Exception as e:
            issues.append(f"Mask read error: {str(e)}")
            stats['error'] = str(e)
        
        return issues, stats
    
    def check_dimension_consistency(self, sample, image_shape=None, depth_shape=None, mask_shape=None):
        """Check if image, depth, and mask dimensions are consistent"""
        issues = []
        
        if image_shape and depth_shape and mask_shape:
            # Extract height and width
            img_h, img_w = image_shape[:2]
            
            if len(depth_shape) == 3:
                depth_h, depth_w = depth_shape[:2]
            else:
                depth_h, depth_w = depth_shape
            
            mask_h, mask_w = mask_shape
            
            # Check consistency
            if not (img_h == depth_h == mask_h and img_w == depth_w == mask_w):
                issues.append(f"Dimension mismatch - Image: {img_h}x{img_w}, Depth: {depth_h}x{depth_w}, Mask: {mask_h}x{mask_w}")
        
        return issues
    
    def check_sample(self, sample):
        """Check a single sample for all quality issues"""
        sample_issues = []
        sample_stats = {}
        
        # Check file existence
        file_issues = self.check_file_existence(sample)
        if file_issues:
            sample_issues.extend(file_issues)
            return sample_issues, sample_stats  # Can't proceed without files
        
        # Check image quality
        try:
            img_issues = self.check_image_quality(sample['image'])
            sample_issues.extend([f"Image: {issue}" for issue in img_issues])
            
            img = np.array(Image.open(sample['image']))
            sample_stats['image_shape'] = img.shape
        except:
            sample_issues.append("Image: Failed to load")
            sample_stats['image_shape'] = None
        
        # Check depth quality
        try:
            depth_issues, depth_stats = self.check_depth_quality(sample['depth'])
            sample_issues.extend([f"Depth: {issue}" for issue in depth_issues])
            sample_stats['depth'] = depth_stats
        except:
            sample_issues.append("Depth: Failed to load")
            sample_stats['depth'] = {'error': 'Failed to load'}
        
        # Check mask quality
        try:
            mask_issues, mask_stats = self.check_mask_quality(sample['mask'])
            sample_issues.extend([f"Mask: {issue}" for issue in mask_issues])
            sample_stats['mask'] = mask_stats
        except:
            sample_issues.append("Mask: Failed to load")
            sample_stats['mask'] = {'error': 'Failed to load'}
        
        # Check dimension consistency
        img_shape = sample_stats.get('image_shape')
        depth_shape = sample_stats.get('depth', {}).get('shape')
        mask_shape = sample_stats.get('mask', {}).get('shape')
        
        dim_issues = self.check_dimension_consistency(sample, img_shape, depth_shape, mask_shape)
        sample_issues.extend(dim_issues)
        
        return sample_issues, sample_stats
    
    def categorize_issues(self, sample, issues, stats):
        """Categorize issues for the sample"""
        sample_path = sample['image']
        
        for issue in issues:
            if 'negative depths' in issue.lower():
                self.issues['negative_depths'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'zero depths' in issue.lower():
                self.issues['zero_depths'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'extreme' in issue.lower():
                self.issues['extreme_depths'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'empty mask' in issue.lower():
                self.issues['empty_masks'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'sparse mask' in issue.lower():
                self.issues['sparse_masks'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'missing' in issue.lower():
                self.issues['missing_files'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'dimension mismatch' in issue.lower():
                self.issues['dimension_mismatches'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'dtype' in issue.lower():
                self.issues['dtype_issues'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            elif 'read error' in issue.lower() or 'failed to load' in issue.lower():
                self.issues['file_read_errors'].append({'path': sample_path, 'issue': issue, 'stats': stats})
            else:
                self.issues['corrupted_files'].append({'path': sample_path, 'issue': issue, 'stats': stats})
    
    def update_global_stats(self, sample_stats):
        """Update global statistics"""
        self.stats['total_samples'] += 1
        
        # Update depth statistics
        if 'depth' in sample_stats and 'min' in sample_stats['depth']:
            depth_stats = sample_stats['depth']
            self.stats['depth_stats']['min'] = min(self.stats['depth_stats']['min'], depth_stats['min'])
            self.stats['depth_stats']['max'] = max(self.stats['depth_stats']['max'], depth_stats['max'])
            
        # Update mask statistics
        if 'mask' in sample_stats and 'valid_ratio' in sample_stats['mask']:
            valid_ratio = sample_stats['mask']['valid_ratio']
            self.stats['mask_stats']['min_valid_ratio'] = min(self.stats['mask_stats']['min_valid_ratio'], valid_ratio)
            self.stats['mask_stats']['max_valid_ratio'] = max(self.stats['mask_stats']['max_valid_ratio'], valid_ratio)
    
    def generate_reports(self):
        """Generate comprehensive quality reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary report
        summary_file = os.path.join(self.output_dir, f"data_quality_summary_{timestamp}.json")
        
        # Calculate final statistics
        self.stats['problematic_samples'] = sum(len(issues) for issues in self.issues.values())
        self.stats['valid_samples'] = self.stats['total_samples'] - len(set(
            item['path'] for category in self.issues.values() for item in category
        ))
        
        # Calculate average depth and mask statistics
        total_depth_samples = 0
        total_depth_sum = 0
        total_mask_ratio_sum = 0
        total_mask_samples = 0
        
        summary_data = {
            'timestamp': timestamp,
            'data_root': self.data_root,
            'thresholds': self.thresholds,
            'statistics': self.stats,
            'issue_counts': {category: len(issues) for category, issues in self.issues.items()},
            'total_issues': sum(len(issues) for issues in self.issues.values())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Detailed issues report
        detailed_file = os.path.join(self.output_dir, f"data_quality_detailed_{timestamp}.json")
        with open(detailed_file, 'w') as f:
            json.dump(self.issues, f, indent=2)
        
        # Problematic files list (for easy filtering)
        problematic_files = set(item['path'] for category in self.issues.values() for item in category)
        
        problematic_list_file = os.path.join(self.output_dir, f"problematic_files_{timestamp}.txt")
        with open(problematic_list_file, 'w') as f:
            f.write(f"# Problematic files found on {timestamp}\n")
            f.write(f"# Total problematic files: {len(problematic_files)}\n")
            f.write(f"# Data root: {self.data_root}\n\n")
            for path in sorted(problematic_files):
                f.write(f"{path}\n")
        
        # Clean files list (for training with good data only)
        all_files = set()
        samples = self.load_sample_paths()
        for sample in samples:
            all_files.add(sample['image'])
        
        clean_files = all_files - problematic_files
        clean_list_file = os.path.join(self.output_dir, f"clean_files_{timestamp}.txt")
        with open(clean_list_file, 'w') as f:
            f.write(f"# Clean files found on {timestamp}\n")
            f.write(f"# Total clean files: {len(clean_files)}\n")
            f.write(f"# Data root: {self.data_root}\n\n")
            for path in sorted(clean_files):
                f.write(f"{path}\n")
        
        return summary_file, detailed_file, problematic_list_file, clean_list_file
    
    def run_quality_check(self):
        """Run the complete data quality check"""
        print(f"🔍 Starting data quality check for: {self.data_root}")
        print(f"📊 Output directory: {self.output_dir}")
        
        samples = self.load_sample_paths()
        print(f"📁 Found {len(samples)} samples to check")
        
        if not samples:
            print("❌ No samples found! Check data_root and splits_dir paths.")
            return
        
        # Check each sample
        for sample in tqdm(samples, desc="Checking data quality"):
            try:
                issues, stats = self.check_sample(sample)
                
                if issues:
                    self.categorize_issues(sample, issues, stats)
                
                self.update_global_stats(stats)
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.issues['file_read_errors'].append({
                    'path': sample.get('image', 'unknown'),
                    'issue': error_msg,
                    'stats': {'error': str(e), 'traceback': traceback.format_exc()}
                })
                self.stats['total_samples'] += 1
        
        # Generate reports
        print("\n📝 Generating quality reports...")
        summary_file, detailed_file, problematic_file, clean_file = self.generate_reports()
        
        # Print summary
        print(f"\n✅ Data quality check completed!")
        print(f"📊 Total samples checked: {self.stats['total_samples']}")
        print(f"✅ Valid samples: {self.stats['valid_samples']}")
        print(f"⚠️  Problematic samples: {self.stats['problematic_samples']}")
        
        print(f"\n📄 Reports generated:")
        print(f"  📋 Summary: {summary_file}")
        print(f"  📝 Detailed: {detailed_file}")
        print(f"  🚫 Problematic files: {problematic_file}")
        print(f"  ✅ Clean files: {clean_file}")
        
        # Print issue breakdown
        print(f"\n🔍 Issue breakdown:")
        for category, issues in self.issues.items():
            if issues:
                print(f"  {category.replace('_', ' ').title()}: {len(issues)}")
        
        return summary_file, detailed_file, problematic_file, clean_file


def main():
    parser = argparse.ArgumentParser(description='Check DIODE dataset quality')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of DIODE dataset')
    parser.add_argument('--splits_dir', type=str, default=None,
                       help='Directory containing dataset split files (optional)')
    parser.add_argument('--output_dir', type=str, default='./data_quality_reports',
                       help='Output directory for reports')
    parser.add_argument('--max_depth', type=float, default=100.0,
                       help='Maximum reasonable depth in meters')
    parser.add_argument('--min_depth', type=float, default=0.001,
                       help='Minimum reasonable depth in meters')
    parser.add_argument('--min_mask_ratio', type=float, default=0.01,
                       help='Minimum valid mask ratio (0.01 = 1%)')
    
    args = parser.parse_args()
    
    # Create checker with custom thresholds
    checker = DataQualityChecker(args.data_root, args.splits_dir, args.output_dir)
    checker.thresholds.update({
        'max_depth': args.max_depth,
        'min_depth': args.min_depth,
        'min_mask_ratio': args.min_mask_ratio
    })
    
    # Run quality check
    checker.run_quality_check()


if __name__ == "__main__":
    main()
