"""
Training script for NER models.

This is the main training pipeline that:
1. Loads configuration and data
2. Builds vocabularies (word, label, character)
3. Loads pre-trained embeddings (GloVe)
4. Creates dataloaders with character features
5. Initializes model with all enhancements
6. Trains with warmup, class weights, and TensorBoard logging
7. Evaluates on dev set after each epoch
8. Saves comprehensive checkpoints

Built from scratch for NLP Course Project.

Author: NLP Course Project
"""

import os
import argparse
import yaml
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import random
import numpy as np
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from src.utils.vocab import build_vocabularies_from_file, CharVocabulary
from src.data.dataset import create_dataloader, print_data_statistics, compute_class_weights
from src.models.baseline_tagger import BaselineBiLSTMTagger
from src.models.bilstm_crf import BiLSTMCRF, BaselineBiLSTM
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logging_utils import TrainingLogger, print_model_summary, print_config
from src.utils.embeddings import initialize_embeddings, load_glove_embeddings


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """Get device for training."""
    device_config = config.get('device', 'auto')

    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and linear decay.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Linear decay to min_lr_ratio
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.

    Falls back to no-op if tensorboard is not available.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer:
            self.writer.add_text(tag, text, step)

    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None,
    gradient_clip: Optional[float] = None,
    use_char_features: bool = False,
    tb_logger: Optional[TensorBoardLogger] = None,
    global_step: int = 0
) -> Tuple[float, int]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        scheduler: Learning rate scheduler (per-step)
        gradient_clip: Gradient clipping value
        use_char_features: Whether to use character features
        tb_logger: TensorBoard logger
        global_step: Global step counter

    Returns:
        Tuple of (average_loss, new_global_step)
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        # Move batch to device
        token_ids = batch['token_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        mask = batch['mask'].to(device)

        # Get character IDs if available
        char_ids = None
        if use_char_features and 'char_ids' in batch:
            char_ids = batch['char_ids'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = model.loss(token_ids, label_ids, mask, char_ids)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        else:
            grad_norm = 0.0

        # Update weights
        optimizer.step()

        # Update learning rate (per-step scheduler)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Log to TensorBoard
        if tb_logger is not None and global_step % 10 == 0:
            tb_logger.log_scalar('train/loss_step', loss.item(), global_step)
            tb_logger.log_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            if gradient_clip is not None:
                tb_logger.log_scalar('train/grad_norm', grad_norm, global_step)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label_vocab,
    use_char_features: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device
        label_vocab: Label vocabulary for decoding
        use_char_features: Whether to use character features

    Returns:
        (avg_loss, metrics) tuple
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            token_ids = batch['token_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)

            # Get character IDs if available
            char_ids = None
            if use_char_features and 'char_ids' in batch:
                char_ids = batch['char_ids'].to(device)

            # Compute loss
            loss = model.loss(token_ids, label_ids, mask, char_ids)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            pred_tag_ids = model.predict(token_ids, mask, char_ids)

            # Convert to tag strings
            for i, (pred_ids, true_ids, m) in enumerate(zip(pred_tag_ids, label_ids, mask)):
                # Get actual length
                length = m.sum().item()

                # Decode predictions
                pred_tags = label_vocab.decode(pred_ids)

                # Decode true tags
                true_tags = label_vocab.decode(true_ids[:length].tolist())

                all_pred_tags.append(pred_tags)
                all_true_tags.append(true_tags)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Compute metrics
    metrics = compute_metrics(all_true_tags, all_pred_tags)

    return avg_loss, metrics


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    best_f1: float,
    config: Dict,
    history: Dict
) -> None:
    """
    Save a comprehensive checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        global_step: Global step counter
        best_f1: Best F1 score achieved
        config: Configuration dictionary
        history: Training history
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_f1': best_f1,
        'config': config,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict:
    """
    Load a checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def train(config_path: str, model_type: str = 'bilstm_crf', resume_from: Optional[str] = None):
    """
    Main training function.

    Args:
        config_path: Path to configuration file
        model_type: Type of model ('baseline', 'baseline_bilstm', or 'bilstm_crf')
        resume_from: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print_config(config)

    # Set random seed
    set_seed(config['random_seed'])

    # Create artifacts directory
    save_dir = config['artifacts']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Get device
    device = get_device(config)

    # Print data statistics
    print_data_statistics(
        config['data']['train_file'],
        config['data']['dev_file'],
        config['data']['test_file']
    )

    # ========== Build Vocabularies ==========
    print("\nBuilding vocabularies...")

    # Check if we should use character features
    use_char_features = config.get('model', {}).get('use_char_features', False)

    word_vocab, label_vocab, char_vocab = build_vocabularies_from_file(
        config['data']['train_file'],
        min_word_freq=config['preprocessing']['min_word_freq'],
        pad_token=config['special_tokens']['pad_token'],
        unk_token=config['special_tokens']['unk_token'],
        build_char_vocab=use_char_features
    )

    # Save vocabularies
    vocab_path = config['artifacts']['vocab_file']
    word_vocab.save(vocab_path.replace('.pkl', '_word.pkl'))
    label_vocab.save(vocab_path.replace('.pkl', '_label.pkl'))
    if char_vocab:
        char_vocab.save(vocab_path.replace('.pkl', '_char.pkl'))

    print(f"Word vocabulary size: {len(word_vocab)}")
    print(f"Label vocabulary size: {len(label_vocab)}")
    if char_vocab:
        print(f"Character vocabulary size: {len(char_vocab)}")

    # ========== Load Pre-trained Embeddings ==========
    pretrained_embeddings = None
    embedding_dim = config['model']['embedding_dim']

    use_pretrained = config.get('model', {}).get('use_pretrained_embeddings', False)
    pretrained_path = config.get('model', {}).get('pretrained_embedding_path', None)
    freeze_embeddings = config.get('model', {}).get('freeze_embeddings', False)

    if use_pretrained and pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pre-trained embeddings from {pretrained_path}...")
        pretrained_embeddings, stats = load_glove_embeddings(
            pretrained_path,
            word_vocab,
            embedding_dim
        )
        print(f"Embedding coverage: {stats['coverage_percent']:.1f}%")
    elif use_pretrained:
        print("\nPre-trained embeddings requested but not found. Using random initialization.")

    # ========== Compute Class Weights ==========
    use_class_weights = config.get('training', {}).get('use_class_weights', False)
    class_weights = None

    if use_class_weights:
        print("\nComputing class weights for imbalanced labels...")
        class_weights = compute_class_weights(
            config['data']['train_file'],
            label_vocab
        )
        print(f"Class weights: {class_weights.tolist()}")

    # ========== Create DataLoaders ==========
    print("\nCreating dataloaders...")

    max_word_length = config.get('model', {}).get('max_word_length', 20)

    train_loader = create_dataloader(
        config['data']['train_file'],
        word_vocab,
        label_vocab,
        char_vocab=char_vocab,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        max_seq_length=config['preprocessing']['max_seq_length'],
        max_word_length=max_word_length,
        use_char_features=use_char_features
    )

    dev_loader = create_dataloader(
        config['data']['dev_file'],
        word_vocab,
        label_vocab,
        char_vocab=char_vocab,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        max_seq_length=config['preprocessing']['max_seq_length'],
        max_word_length=max_word_length,
        use_char_features=use_char_features
    )

    # ========== Initialize Model ==========
    print("\nInitializing model...")

    if model_type == 'baseline':
        # Original baseline (from baseline_tagger.py)
        model = BaselineBiLSTMTagger(
            vocab_size=len(word_vocab),
            num_tags=len(label_vocab),
            embedding_dim=embedding_dim,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=word_vocab.pad_idx
        )
    elif model_type == 'baseline_bilstm':
        # BiLSTM without CRF (for ablation)
        model = BaselineBiLSTM(
            vocab_size=len(word_vocab),
            num_tags=len(label_vocab),
            embedding_dim=embedding_dim,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=word_vocab.pad_idx,
            pretrained_embeddings=pretrained_embeddings
        )
    else:  # bilstm_crf (full model)
        # Get model configuration
        use_attention = config.get('model', {}).get('use_attention', False)
        attention_heads = config.get('model', {}).get('attention_heads', 4)
        attention_dropout = config.get('model', {}).get('attention_dropout', 0.1)
        char_embedding_dim = config.get('model', {}).get('char_embedding_dim', 30)
        char_hidden_size = config.get('model', {}).get('char_hidden_size', 50)
        char_kernel_sizes = config.get('model', {}).get('char_kernel_sizes', [2, 3, 4])
        use_highway = config.get('model', {}).get('use_highway', True)

        model = BiLSTMCRF(
            vocab_size=len(word_vocab),
            num_tags=len(label_vocab),
            embedding_dim=embedding_dim,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=word_vocab.pad_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
            use_char_features=use_char_features,
            num_chars=len(char_vocab) if char_vocab else 0,
            char_embedding_dim=char_embedding_dim,
            char_hidden_size=char_hidden_size,
            char_kernel_sizes=char_kernel_sizes,
            use_highway=use_highway,
            use_attention=use_attention,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout
        )

    model = model.to(device)

    print_model_summary(model, len(word_vocab), len(label_vocab))

    # Print model configuration
    if hasattr(model, 'get_model_summary'):
        summary = model.get_model_summary()
        print("\nModel Configuration:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    # ========== Initialize Optimizer ==========
    optimizer_type = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

    # ========== Learning Rate Scheduler ==========
    scheduler = None
    num_epochs = config['training']['num_epochs']
    num_training_steps = len(train_loader) * num_epochs

    use_warmup = config.get('training', {}).get('use_warmup', False)
    warmup_epochs = config.get('training', {}).get('warmup_epochs', 2)
    num_warmup_steps = len(train_loader) * warmup_epochs

    if use_warmup:
        print(f"\nUsing warmup scheduler: {warmup_epochs} warmup epochs ({num_warmup_steps} steps)")
        scheduler = get_warmup_scheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=0.01
        )
    elif config['training'].get('use_lr_scheduler', False):
        lr_scheduler_type = config['training'].get('lr_scheduler', 'step')
        if lr_scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['lr_step_size'],
                gamma=config['training']['lr_gamma']
            )
        elif lr_scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config['training']['lr_gamma'],
                patience=3,
                verbose=True
            )

    # ========== TensorBoard Logger ==========
    use_tensorboard = config.get('logging', {}).get('use_tensorboard', False)
    tb_log_dir = os.path.join(save_dir, 'tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_logger = TensorBoardLogger(tb_log_dir, enabled=use_tensorboard)

    # ========== Resume from Checkpoint ==========
    start_epoch = 1
    global_step = 0
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'dev_loss': [],
        'dev_f1': [],
        'dev_precision': [],
        'dev_recall': [],
        'learning_rate': []
    }

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        best_f1 = checkpoint['best_f1']
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ========== Training Logger ==========
    logger = TrainingLogger(log_interval=config['logging']['log_interval'])
    logger.best_f1 = best_f1

    # ========== Training Loop ==========
    print("\n" + "=" * 80)
    print(f"Training {model_type.upper()} model")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Character features: {use_char_features}")
    print(f"  Pre-trained embeddings: {use_pretrained}")
    print(f"  Warmup: {use_warmup}")
    print(f"  Class weights: {use_class_weights}")
    print("=" * 80 + "\n")

    patience_counter = 0

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler if use_warmup else None,
            gradient_clip=config['training']['gradient_clip'],
            use_char_features=use_char_features,
            tb_logger=tb_logger,
            global_step=global_step
        )

        # Evaluate on dev set
        dev_loss, dev_metrics = evaluate(
            model, dev_loader, device, label_vocab,
            use_char_features=use_char_features
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        history['dev_f1'].append(dev_metrics['f1'])
        history['dev_precision'].append(dev_metrics['precision'])
        history['dev_recall'].append(dev_metrics['recall'])
        history['learning_rate'].append(current_lr)

        # Log to console
        logger.log_epoch(epoch, train_loss, dev_loss, dev_metrics, epoch_time)
        print(f"Learning rate: {current_lr:.6f}")

        # Log to TensorBoard
        tb_logger.log_scalar('train/loss_epoch', train_loss, epoch)
        tb_logger.log_scalar('dev/loss', dev_loss, epoch)
        tb_logger.log_scalar('dev/f1', dev_metrics['f1'], epoch)
        tb_logger.log_scalar('dev/precision', dev_metrics['precision'], epoch)
        tb_logger.log_scalar('dev/recall', dev_metrics['recall'], epoch)
        tb_logger.log_scalar('train/learning_rate_epoch', current_lr, epoch)

        # Update learning rate (epoch-level scheduler)
        if scheduler is not None and not use_warmup:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(dev_metrics['f1'])
            else:
                scheduler.step()

        # Save best model
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            patience_counter = 0
            logger.best_f1 = best_f1
            logger.best_epoch = epoch

            # Save checkpoint
            checkpoint_path = config['artifacts']['model_checkpoint']
            save_checkpoint(
                checkpoint_path,
                model, optimizer, scheduler,
                epoch, global_step, best_f1,
                config, history
            )
            print(f"[OK] Best model saved to {checkpoint_path}")

            # Also save as best model separately
            best_model_path = checkpoint_path.replace('.pt', '_best.pt')
            save_checkpoint(
                best_model_path,
                model, optimizer, scheduler,
                epoch, global_step, best_f1,
                config, history
            )
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % config.get('logging', {}).get('checkpoint_interval', 5) == 0:
            periodic_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(
                periodic_path,
                model, optimizer, scheduler,
                epoch, global_step, best_f1,
                config, history
            )
            print(f"[OK] Periodic checkpoint saved to {periodic_path}")

        # Early stopping
        if config['training'].get('early_stopping', False):
            patience = config['training'].get('patience', 5)
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best F1 was {best_f1:.4f} at epoch {logger.best_epoch}")
                break

    # ========== Final Summary ==========
    logger.log_final()

    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Close TensorBoard logger
    tb_logger.close()

    print(f"\nTraining complete! Best dev F1: {best_f1:.4f}")
    print(f"Model saved to: {config['artifacts']['model_checkpoint']}")

    return best_f1, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='bilstm_crf',
                        choices=['baseline', 'baseline_bilstm', 'bilstm_crf'],
                        help='Model type to train')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train(args.config, args.model, args.resume)


if __name__ == '__main__':
    main()
