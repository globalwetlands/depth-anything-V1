#!/usr/bin/env python3
import numpy as np
import os

# Check depth file
depth_path = '/data/deployment_1/scan_000001/1_gp1_L_edt_001292_depth.npy'
if os.path.exists(depth_path):
    depth = np.load(depth_path)
    print('Depth shape:', depth.shape)
    print('Depth dtype:', depth.dtype)
    print('Depth min/max:', depth.min(), depth.max())
else:
    print('Depth file not found')

# Check image file
image_path = '/data/deployment_1/scan_000001/1_gp1_L_edt_001292.png'
if os.path.exists(image_path):
    from PIL import Image
    img = Image.open(image_path)
    print('Image size:', img.size)
    print('Image mode:', img.mode)
else:
    print('Image file not found')
