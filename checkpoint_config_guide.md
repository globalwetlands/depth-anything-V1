# Checkpoint Saving & Monitoring Guide

## 📁 Checkpoint Configuration

### Default Checkpoint Settings
```bash
# Checkpoints are saved to:
--save_dir="./depth_anything_finetune"

# Validation frequency (controls checkpoint frequency):
--validate_every=0.25  # Every 25% of epoch (4 times per epoch)

# Image logging frequency:
--log_images_every=0.1  # Every 10% of epoch
```

### Custom Checkpoint Settings
```bash
# Save more frequently
docker compose run --rm depth-anything train -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 --epochs=5 \
  --save_dir="/outputs/my_training_run" \
  --validate_every=0.1 \
  --log_images_every=0.05

# Save less frequently (for faster training)
docker compose run --rm depth-anything train -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 --epochs=5 \
  --validate_every=0.5 \
  --log_images_every=0.2
```

## 📊 Current Monitoring (Wandb)

The system uses **Weights & Biases (wandb)** for monitoring by default, but we've disabled it.

### Enable Wandb Monitoring
```bash
# Remove WANDB_MODE=disabled and add your API key
docker compose run --rm depth-anything train -m zoedepth -d diode_outdoor \
  --pretrained_resource="" --workers=0 --epochs=5 \
  -e WANDB_API_KEY="your_wandb_api_key_here"
```

## 📈 TensorBoard Setup

TensorBoard support needs to be added. Here's how:

### 1. Add TensorBoard to Requirements
Already included: `tensorboard` is in the Dockerfile pip install

### 2. Mount TensorBoard Logs
```yaml
# Add to docker-compose.yml volumes:
- ./tensorboard_logs:/app/tensorboard_logs
```

### 3. Run TensorBoard
```bash
# In a separate terminal:
docker run --rm -p 6006:6006 -v $(pwd)/tensorboard_logs:/logs tensorflow/tensorflow:latest tensorboard --logdir=/logs --host=0.0.0.0

# Then open: http://localhost:6006
```

## 📋 Checkpoint Files Explained

### Automatic Saves:
- `{experiment_id}_latest.pt` - Most recent model
- `{experiment_id}_best.pt` - Best validation performance
- Contains: model weights, epoch number (optimizer excluded for size)

### Manual Model Saving:
```python
# In your code, save additional checkpoints:
self.save_checkpoint(f"epoch_{epoch}_checkpoint.pt")
```

## 🔍 Monitoring Progress

### Option 1: Enable Wandb (Recommended)
```bash
# Set your wandb API key
echo "WANDB_API_KEY=your_key_here" >> .env

# Modify docker-compose.yml to include:
env_file: .env

# Run without WANDB_MODE=disabled
docker compose run --rm depth-anything train ...
```

### Option 2: Add TensorBoard Support
Would require modifying the trainer to add TensorBoard logging alongside wandb.

### Option 3: Log File Monitoring
```bash
# Monitor training progress via logs
docker compose run --rm depth-anything train ... | tee training.log

# In another terminal:
tail -f training.log
```

## 💡 Best Practices

1. **Regular Backups**: Mount `/outputs` to persistent storage
2. **Experiment Naming**: Use descriptive experiment IDs
3. **Validation Frequency**: Balance between monitoring and training speed
4. **Storage Management**: Clean up old checkpoints periodically
5. **Resume Training**: Use latest checkpoint if training stops

## 🚀 Recommended Training Command

```bash
# Production training with good monitoring
docker compose run --rm depth-anything train \
  -m zoedepth -d diode_outdoor \
  --pretrained_resource="" \
  --workers=0 \
  --epochs=20 \
  --batch_size=8 \
  --save_dir="/outputs/diode_outdoor_$(date +%Y%m%d_%H%M%S)" \
  --validate_every=0.2 \
  --log_images_every=0.1 \
  --experiment_id="diode_outdoor_training"
```
