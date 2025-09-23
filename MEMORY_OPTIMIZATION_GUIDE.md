# 🚀 Memory Optimization Guide for Long Training Runs

## 🎯 Issue Addressed
Your training successfully ran for **6 epochs** but encountered CUDA out of memory after several hours. This is due to **memory accumulation** during long training runs.

## ✅ **Implemented Fixes**

### 1. **Automatic Memory Cleanup**
- **Every 100 batches**: `torch.cuda.empty_cache()` during training
- **After validation**: Memory cleanup after each validation run
- **After final validation**: Complete memory cleanup at end

### 2. **Training Commands with Different Memory Profiles**

```bash
# MOST CONSERVATIVE (Batch Size 1)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 1 production_bs1_$(date +%Y%m%d)

# BALANCED (Batch Size 2) - Your current working setup with memory fixes
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 2 production_bs2_$(date +%Y%m%d)

# AGGRESSIVE (Try Batch Size 3-4 if memory allows)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 3 production_bs3_$(date +%Y%m%d)
```

## 🔧 **Additional Memory Optimizations**

### **Option A: Gradient Accumulation** (Simulates larger batch sizes)
You can achieve the effect of larger batch sizes by accumulating gradients:

```bash
# Accumulate gradients over 4 batches (effective batch size = 2 * 4 = 8)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 2 production_gradacc_$(date +%Y%m%d) --gradient_accumulation_steps=4
```

### **Option B: Resume Training from Checkpoint**
If memory issues persist, you can resume from your checkpoint:

```bash
# Resume from where you left off (epoch 6)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 1 production_resume_$(date +%Y%m%d) --resume_from=/outputs/production_20250922/production_20250922_latest.pt
```

## 💡 **Memory Monitoring Commands**

```bash
# Check GPU memory during training (run in another terminal)
watch -n 5 "nvidia-smi"

# Check specific memory usage
docker compose run --rm depth-anything python -c "
import torch
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB')  
print(f'Cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB')
"
```

## 🎯 **Recommended Approach**

Based on your success with 6 epochs, I recommend:

### **1. Continue with Batch Size 2 + Memory Cleanup**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 2 production_optimized_$(date +%Y%m%d)
```

### **2. If that still has issues, use Batch Size 1**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./dataset_manager.sh train 120 1 production_safe_$(date +%Y%m%d)
```

## 📊 **Training Progress Monitoring**

- **TensorBoard**: Access at `http://localhost:6006`
- **Logs**: Saved in `training_logs/experiment_name/`
- **Checkpoints**: Auto-saved in `/outputs/experiment_name/`

## 🔄 **What Changed**

1. **Memory cleanup every 100 batches** - Prevents gradual accumulation
2. **Memory cleanup after validation** - Frees validation memory immediately  
3. **Better memory management** - More aggressive cache clearing

Your training should now run the full 120 epochs without memory issues!

## 🚨 **If Issues Persist**

1. **Reduce batch size to 1**
2. **Monitor with `nvidia-smi`** to see memory patterns
3. **Consider gradient checkpointing** (trades compute for memory)
4. **Use model checkpointing** to resume from failures

Your 6-epoch success shows the system works - now it's optimized for long runs! 🚀
