#!/usr/bin/env python3
"""
Evaluation script with support for dataset splits
"""

import argparse
import os
import sys

# Add the metric_depth directory to Python path
sys.path.insert(0, '/app/metric_depth')

import numpy as np
import torch
import torch.nn as nn
from pprint import pprint
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.data.diode_splits import get_diode_test_loader, get_diode_val_loader
from zoedepth.utils.misc import compute_errors
import cv2
from tqdm import tqdm

# Global settings
FL = 715.0873
FY = 715.0873
CX = 1024 // 2
CY = 768 // 2
FINAL_HEIGHT = 480
FINAL_WIDTH = 640

def compute_scale_and_shift(prediction, target, mask):
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


@torch.no_grad()
def infer(model, image, dataset="diode", focal=None):

    if isinstance(image, (list, tuple)):
        image = image[0]

    if isinstance(focal, (list, tuple)):
        focal = focal[0]

    image_input = image.unsqueeze(0)
    pred1 = model(image_input, dataset=dataset, focal=focal)
    pred2 = model(torch.flip(image_input, [3]), dataset=dataset, focal=focal)
    pred = 0.5 * (pred1 + torch.flip(pred2, [3]))
    pred = pred.squeeze().cpu().numpy()
    pred[pred < 0] = 0.25
    return pred


@torch.no_grad()
def evaluate(model, test_loader, config):
    model.eval()
    metrics = RunningAverageDict()
    test_loader = tqdm(test_loader, desc="Evaluating")
    
    for i, sample in enumerate(test_loader):
        if isinstance(sample, dict):
            image = sample['image'].to(config.gpu, non_blocking=True)
            gt_depth = sample['depth'].to(config.gpu, non_blocking=True)
            mask = sample['mask'].to(config.gpu, non_blocking=True).to(torch.bool)
            focal = [FL]
        else:
            image, gt_depth, mask = sample
            image = image.to(config.gpu, non_blocking=True)
            gt_depth = gt_depth.to(config.gpu, non_blocking=True)
            mask = mask.to(config.gpu, non_blocking=True).to(torch.bool)
            focal = [FL]

        bs = image.shape[0]
        
        # Inference
        pred = infer(model, image, dataset="diode", focal=focal)
        pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).cuda()

        # Resize prediction to match ground truth
        pred = nn.functional.interpolate(pred, size=gt_depth.shape[-2:], mode='bilinear', align_corners=True)
        
        # Apply mask and compute metrics
        pred = pred.squeeze().unsqueeze(0)
        
        # Compute scale and shift (for evaluation metric)
        scale, shift = compute_scale_and_shift(pred, gt_depth.squeeze(), mask.squeeze())
        pred_aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
        
        # Compute errors
        pred_aligned = pred_aligned.squeeze()
        gt_depth = gt_depth.squeeze()
        mask = mask.squeeze()
        
        # Apply depth range constraints
        pred_aligned[pred_aligned < config.min_depth_eval] = config.min_depth_eval
        pred_aligned[pred_aligned > config.max_depth_eval] = config.max_depth_eval
        
        pred_aligned = pred_aligned[mask]
        gt_depth = gt_depth[mask]
        
        if len(pred_aligned) > 0:
            errors = compute_errors(gt_depth.cpu().numpy(), pred_aligned.cpu().numpy())
            for key, val in errors.items():
                metrics.update(key, val)

    return metrics.get_value()


class RunningAverageDict:
    def __init__(self):
        self._dict = {}

    def update(self, key, value):
        if key not in self._dict:
            self._dict[key] = RunningAverage()
        self._dict[key].append(value)

    def get_value(self):
        return {k: v.get_value() for k, v in self._dict.items()}


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str, required=False, default="", 
                       help="Pretrained resource to use for fetching weights")
    parser.add_argument("-d", "--dataset", type=str, required=False, default='diode_outdoor', 
                       help="Dataset to evaluate on")
    parser.add_argument("--splits_dir", type=str, required=True,
                       help="Directory containing dataset split files")
    parser.add_argument("--split", type=str, choices=['val', 'test'], default='test',
                       help="Which split to evaluate on")
    
    args = parser.parse_args()

    config = get_config(args.model, "eval", args.dataset)
    config.pretrained_resource = args.pretrained_resource
    config.splits_dir = args.splits_dir
    config.gpu = 0

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.cuda.current_device()
    else:
        device = torch.device('cpu')
        config.gpu = None

    print("Evaluating model:", args.model)
    print("Pretrained resource:", args.pretrained_resource)
    print("Dataset:", args.dataset) 
    print("Split:", args.split)
    print("Device:", device)

    # Build model
    model = build_model(config)
    if config.gpu is not None:
        model = model.cuda(config.gpu)

    # Load test data
    if args.split == 'test':
        test_loader = get_diode_test_loader(
            config.splits_dir,
            batch_size=1,
            num_workers=1
        )
    else:
        test_loader = get_diode_val_loader(
            config.splits_dir,
            batch_size=1,
            num_workers=1
        )

    print(f"Loaded {len(test_loader)} samples for evaluation")

    # Evaluate
    print("Starting evaluation...")
    metrics = evaluate(model, test_loader, config)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
