#!/usr/bin/env python3
import numpy as np
import os

# Check mask file
mask_path = '/data/deployment_1/scan_000001/1_gp1_L_edt_001292_depth_mask.npy'
if os.path.exists(mask_path):
    mask = np.load(mask_path)
    print('Mask shape:', mask.shape)
    print('Mask dtype:', mask.dtype)
    print('Mask unique values:', np.unique(mask))
    print('Mask min/max:', mask.min(), mask.max())
else:
    print('Mask file not found')
