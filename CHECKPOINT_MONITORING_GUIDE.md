# 📊 Checkpoint Saving & TensorBoard Monitoring Guide

## 🚀 Quick Start

### 1. Setup Monitoring
```bash
./training_monitor.sh setup
```

### 2. Start TensorBoard (Optional)
```bash
./training_monitor.sh start-tensorboard
# Access at: http://localhost:6006
```

### 3. Start Training with Monitoring
```bash
# Quick test (1 epoch)
./training_monitor.sh train 1 8 test_run

# Full training (20 epochs) 
./training_monitor.sh train 20 8 production_run

# Custom training
./training_monitor.sh train 10 16 custom_experiment
```

## 📁 Checkpoint System

### ✅ **Automatic Checkpoint Saving**

The system **automatically saves** checkpoints:

1. **`{experiment_id}_latest.pt`** - Saved every validation interval
2. **`{experiment_id}_best.pt`** - Saved when validation improves
3. **Final checkpoint** - Saved at training end

### 📍 **Checkpoint Locations**

```bash
# Output directory (persistent)
/mnt/fast-data/monocular/outputs/{experiment_name}/

# Local directory (temporary)
./checkpoints/
```

### ⚙️ **Checkpoint Frequency Control**

```bash
# More frequent checkpoints (every 10% of epoch)
--validate_every=0.1

# Default frequency (every 25% of epoch)  
--validate_every=0.25

# Less frequent (every 50% of epoch)
--validate_every=0.5
```

## 📈 Monitoring Options

### 🔥 **Option 1: TensorBoard (Recommended)**

```bash
# Start TensorBoard service
./training_monitor.sh start-tensorboard

# Access dashboard
open http://localhost:6006

# Stop TensorBoard
./training_monitor.sh stop-tensorboard
```

**Features:**
- Real-time loss tracking
- Learning rate monitoring  
- Model architecture visualization
- Image logging (every 10% of epoch)

### 📊 **Option 2: Weights & Biases**

```bash
# Enable wandb (remove WANDB_MODE=disabled)
docker compose run --rm depth-anything train \
  -m zoedepth -d diode_outdoor \
  --pretrained_resource="" \
  -e WANDB_API_KEY="your_key_here"
```

### 📝 **Option 3: Log File Monitoring**

```bash
# Monitor training progress in real-time
./training_monitor.sh monitor

# Or manually:
tail -f training_logs/{experiment_name}/training.log
```

## 🛠️ Manual Checkpoint Management

### Check Checkpoint Status
```bash
./training_monitor.sh checkpoints
```

### Backup Checkpoints
```bash
./training_monitor.sh backup
```

### Load Checkpoint for Evaluation
```bash
# Use your trained model
docker compose run --rm depth-anything evaluate \
  -m zoedepth \
  --pretrained_resource="local::/outputs/my_experiment/my_experiment_best.pt" \
  -d diode_outdoor
```

## 📋 Training Command Examples

### 🧪 **Testing Setup**
```bash
# Quick validation (1 epoch, frequent checkpoints)
docker compose run --rm depth-anything train \
  -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 \
  --epochs=1 --batch_size=4 \
  --validate_every=0.1 --log_images_every=0.05 \
  --save_dir="/outputs/test_$(date +%Y%m%d_%H%M%S)"
```

### 🎯 **Production Training**
```bash
# Full training with optimal monitoring
docker compose run --rm depth-anything train \
  -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 \
  --epochs=20 --batch_size=8 \
  --validate_every=0.2 --log_images_every=0.1 \
  --save_dir="/outputs/production_$(date +%Y%m%d_%H%M%S)" \
  --experiment_id="diode_outdoor_production"
```

### ⚡ **Fast Training**
```bash
# Minimal checkpointing for speed
docker compose run --rm depth-anything train \
  -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 \
  --epochs=10 --batch_size=16 \
  --validate_every=0.5 --log_images_every=0.25 \
  --save_dir="/outputs/fast_training"
```

## 📊 What Gets Monitored

### 📈 **Training Metrics**
- Loss per iteration
- Learning rate changes
- Gradient norms
- Training time per epoch

### 📉 **Validation Metrics**
- Validation loss
- Depth estimation accuracy
- RMSE, MAE metrics
- Best model tracking

### 🖼️ **Visual Monitoring**
- Input images
- Ground truth depth
- Predicted depth
- Error visualizations

## 🔧 Troubleshooting

### Checkpoint Issues
```bash
# Check disk space
df -h /mnt/fast-data/monocular/outputs

# Check permissions
ls -la /mnt/fast-data/monocular/outputs

# Manual cleanup
find /mnt/fast-data/monocular/outputs -name "*.pt" -mtime +7 -delete
```

### TensorBoard Issues
```bash
# Restart TensorBoard
./training_monitor.sh stop-tensorboard
./training_monitor.sh start-tensorboard

# Check logs
docker compose --profile monitoring logs tensorboard
```

### Monitoring Issues
```bash
# Check if training is running
docker compose ps

# Monitor resource usage
docker stats
```

## 💡 Best Practices

1. **🔄 Regular Backups**: Use `./training_monitor.sh backup`
2. **📝 Descriptive Names**: Use meaningful experiment IDs
3. **💾 Storage Management**: Clean old checkpoints periodically
4. **📊 Monitor Early**: Start TensorBoard before training
5. **🚨 Validation Frequency**: Balance monitoring vs. speed
6. **🔍 Early Stopping**: Stop if validation loss plateaus

## 🎯 Quick Commands Summary

```bash
# Setup and start monitoring
./training_monitor.sh setup
./training_monitor.sh start-tensorboard

# Start training (epochs, batch_size, name)
./training_monitor.sh train 5 8 my_experiment

# Monitor progress
./training_monitor.sh monitor

# Check saved models
./training_monitor.sh checkpoints

# Backup everything
./training_monitor.sh backup
```

**🌐 TensorBoard Dashboard**: http://localhost:6006

Your training setup is now fully equipped with comprehensive checkpoint saving and monitoring! 🚀
