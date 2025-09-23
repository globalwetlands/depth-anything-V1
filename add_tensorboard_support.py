#!/usr/bin/env python3
"""
Script to add TensorBoard support to the existing trainer.
This modifies the base_trainer.py to include TensorBoard logging alongside wandb.
"""

import os

# Content to add TensorBoard support
tensorboard_imports = """
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
"""

tensorboard_init = """
        # Initialize TensorBoard writer
        self.tb_writer = None
        if TENSORBOARD_AVAILABLE and self.should_log:
            tb_log_dir = os.path.join(self.config.root, 'tensorboard_logs', self.config.experiment_id)
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_log_dir)
            print(f"TensorBoard logging to: {tb_log_dir}")
"""

def add_tensorboard_to_trainer():
    """Add TensorBoard support to the base trainer"""
    trainer_file = "metric_depth/zoedepth/trainers/base_trainer.py"
    
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Add TensorBoard imports after existing imports
    if "from torch.utils.tensorboard import SummaryWriter" not in content:
        # Find the last import line
        lines = content.split('\n')
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                last_import_idx = i
        
        if last_import_idx != -1:
            lines.insert(last_import_idx + 1, '')
            lines.insert(last_import_idx + 2, tensorboard_imports.strip())
            content = '\n'.join(lines)
    
    # Add TensorBoard initialization in __init__
    if "self.tb_writer = None" not in content:
        # Find the wandb.init call and add TensorBoard init after it
        content = content.replace(
            'wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,\n                       tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))',
            'wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,\n                       tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))\n' + tensorboard_init.strip()
        )
    
    # Add TensorBoard logging methods
    tb_logging_methods = '''
    def tb_log_scalar(self, tag, value, step):
        """Log scalar to TensorBoard"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
    
    def tb_log_image(self, tag, image, step):
        """Log image to TensorBoard"""
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, image, step)
    
    def tb_log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars to TensorBoard"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
'''
    
    if "def tb_log_scalar" not in content:
        # Add before the last method
        content = content.replace(
            'wandb.log({title: wandb.plot.bar(table, "label",\n                  "value", title=title)}, step=self.step)',
            'wandb.log({title: wandb.plot.bar(table, "label",\n                  "value", title=title)}, step=self.step)' + tb_logging_methods
        )
    
    # Add TensorBoard logging calls
    # Log training losses
    content = content.replace(
        'wandb.log({f"Train/{name}": loss.item()\n                              for name, loss in losses.items()}, step=self.step)',
        'wandb.log({f"Train/{name}": loss.item()\n                              for name, loss in losses.items()}, step=self.step)\n                # TensorBoard logging\n                for name, loss in losses.items():\n                    self.tb_log_scalar(f"Train/{name}", loss.item(), self.step)'
    )
    
    # Log validation losses and metrics
    content = content.replace(
        'wandb.log(\n                                {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)\n\n                            wandb.log({f"Metrics/{k}": v for k,\n                                      v in metrics.items()}, step=self.step)',
        'wandb.log(\n                                {f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)\n\n                            wandb.log({f"Metrics/{k}": v for k,\n                                      v in metrics.items()}, step=self.step)\n                            # TensorBoard logging\n                            for name, tloss in test_losses.items():\n                                self.tb_log_scalar(f"Test/{name}", tloss, self.step)\n                            for k, v in metrics.items():\n                                self.tb_log_scalar(f"Metrics/{k}", v, self.step)'
    )
    
    with open(trainer_file, 'w') as f:
        f.write(content)
    
    print("✅ TensorBoard support added to base_trainer.py")

if __name__ == "__main__":
    add_tensorboard_to_trainer()
