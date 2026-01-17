"""
Logging utilities for training and evaluation.

This module provides:
- Progress bar utilities
- Training logger
- Formatted output functions
"""

import time
from typing import Dict, Optional
from tqdm import tqdm


class TrainingLogger:
    """
    Logger for tracking training progress and metrics.

    Args:
        log_interval: Number of batches between log messages
    """

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.history = {
            'train_loss': [],
            'dev_loss': [],
            'dev_f1': [],
            'dev_precision': [],
            'dev_recall': []
        }

        self.best_f1 = 0.0
        self.best_epoch = 0

    def log_batch(self, epoch: int, batch_idx: int, num_batches: int,
                  loss: float, time_elapsed: float) -> None:
        """
        Log batch-level information.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            num_batches: Total number of batches
            loss: Current loss value
            time_elapsed: Time elapsed for this batch
        """
        if (batch_idx + 1) % self.log_interval == 0:
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | "
                  f"Loss: {loss:.4f} | Time: {time_elapsed:.2f}s")

    def log_epoch(self, epoch: int, train_loss: float, dev_loss: float,
                  dev_metrics: Dict[str, float], time_elapsed: float) -> None:
        """
        Log epoch-level information.

        Args:
            epoch: Current epoch number
            train_loss: Average training loss
            dev_loss: Average development loss
            dev_metrics: Development set metrics
            time_elapsed: Time elapsed for this epoch
        """
        self.history['train_loss'].append(train_loss)
        self.history['dev_loss'].append(dev_loss)
        self.history['dev_f1'].append(dev_metrics['f1'])
        self.history['dev_precision'].append(dev_metrics['precision'])
        self.history['dev_recall'].append(dev_metrics['recall'])

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} Summary")
        print("=" * 80)
        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Dev Loss:        {dev_loss:.4f}")
        print(f"Dev Precision:   {dev_metrics['precision']:.4f}")
        print(f"Dev Recall:      {dev_metrics['recall']:.4f}")
        print(f"Dev F1:          {dev_metrics['f1']:.4f}")
        print(f"Time:            {time_elapsed:.2f}s")

        # Check if best
        if dev_metrics['f1'] > self.best_f1:
            self.best_f1 = dev_metrics['f1']
            self.best_epoch = epoch
            print(f"\n*** New best F1 score! ({self.best_f1:.4f}) ***")

        print(f"Best F1 so far:  {self.best_f1:.4f} (Epoch {self.best_epoch})")
        print("=" * 80 + "\n")

    def log_final(self) -> None:
        """Log final training summary."""
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best Dev F1:     {self.best_f1:.4f}")
        print(f"Best Epoch:      {self.best_epoch}")
        print(f"Total Epochs:    {len(self.history['train_loss'])}")
        print("=" * 80 + "\n")


def progress_bar(iterable, desc: str = "", total: Optional[int] = None):
    """
    Create a progress bar for iterations.

    Args:
        iterable: Iterable to track
        desc: Description to display
        total: Total number of items (if not inferrable from iterable)

    Returns:
        tqdm progress bar
    """
    return tqdm(iterable, desc=desc, total=total, ncols=80)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_model_summary(model, vocab_size: int, num_labels: int) -> None:
    """
    Print a summary of the model.

    Args:
        model: PyTorch model
        vocab_size: Vocabulary size
        num_labels: Number of labels
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Model type:          {model.__class__.__name__}")
    print(f"Vocabulary size:     {vocab_size}")
    print(f"Number of labels:    {num_labels}")
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60 + "\n")


def print_config(config: Dict) -> None:
    """
    Print configuration in a readable format.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(config)
    print("=" * 60 + "\n")


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} took {format_time(self.elapsed)}")
