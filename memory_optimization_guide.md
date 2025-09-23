# 🚀 Memory Optimization Guide for ZoeDepth Training

## 🎯 Current Issue
CUDA out of memory with 31.37 GiB GPU using batch size 8. The model is using 30.18 GiB, leaving insufficient memory for training.

## 💡 Solutions (Try in Order)

### 1. **Reduce Batch Size Further**
```bash
# Try batch size 4
./dataset_manager.sh train 120 4 production_$(date +%Y%m%d)

# If still fails, try batch size 2
./dataset_manager.sh train 120 2 production_$(date +%Y%m%d)

# If still fails, try batch size 1
./dataset_manager.sh train 120 1 production_$(date +%Y%m%d)
```

### 2. **Enable PyTorch Memory Optimization**
Set the memory allocation strategy suggested in the error:

```bash
# Set environment variable for memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 4 production_$(date +%Y%m%d)
```

### 3. **Enable Gradient Checkpointing (Recommended)**
Modify the configuration to use gradient checkpointing, which trades compute for memory.

### 4. **Use Mixed Precision Training**
Enable automatic mixed precision to reduce memory usage.

### 5. **Reduce Image Resolution**
Use smaller input images during training.

## 🔧 Quick Fix Commands

### Option A: Memory-Optimized Training
```bash
# Use memory optimization environment variable
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 4 production_$(date +%Y%m%d)
```

### Option B: Ultra-Conservative Training
```bash
# Smallest possible batch size with memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 1 production_$(date +%Y%m%d)
```

### Option C: Check GPU Memory First
```bash
# Clear any existing GPU memory
docker compose run --rm depth-anything python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB')
    print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB')
"
```

## 🎛️ Advanced Memory Optimization

I can also create modified training scripts with:
- Gradient checkpointing enabled
- Mixed precision training
- Smaller model variants
- Dynamic batch sizing

Would you like me to implement any of these advanced optimizations?
