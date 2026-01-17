"""
Visualization utilities for NER models.

This module provides plotting functions for:
- Training curves (loss, F1, precision, recall)
- Entity distribution charts
- Attention weight heatmaps
- Confusion matrix visualization
- Entity length performance charts

Uses matplotlib for static plots.
Falls back gracefully if matplotlib is not available.

Built from scratch for NLP Course Project.

Author: NLP Course Project
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization functions will not work.")


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str,
    title: str = "Training Progress"
) -> None:
    """
    Plot training curves (loss, F1, precision, recall).

    Args:
        history: Dictionary with training history
            Expected keys: train_loss, dev_loss, dev_f1, dev_precision, dev_recall
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['dev_loss'], 'r-', label='Dev Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Development Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: F1 score
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['dev_f1'], 'g-', label='Dev F1', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Development F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Find best epoch
    best_idx = np.argmax(history['dev_f1'])
    ax2.axvline(x=best_idx + 1, color='r', linestyle='--', alpha=0.5, label=f'Best: {history["dev_f1"][best_idx]:.4f}')
    ax2.scatter([best_idx + 1], [history['dev_f1'][best_idx]], color='r', s=100, zorder=5)

    # Plot 3: Precision and Recall
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['dev_precision'], 'b-', label='Precision', linewidth=2)
    ax3.plot(epochs, history['dev_recall'], 'r-', label='Recall', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Development Precision and Recall')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate (if available)
    ax4 = axes[1, 1]
    if 'learning_rate' in history and history['learning_rate']:
        ax4.plot(epochs, history['learning_rate'], 'm-', label='Learning Rate', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        # Show all metrics together
        ax4.plot(epochs, history['dev_f1'], 'g-', label='F1', linewidth=2)
        ax4.plot(epochs, history['dev_precision'], 'b-', label='Precision', linewidth=2)
        ax4.plot(epochs, history['dev_recall'], 'r-', label='Recall', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('All Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_path}")


def plot_entity_distribution(
    label_counts: Dict[str, int],
    output_path: str,
    title: str = "Entity Distribution"
) -> None:
    """
    Plot entity type distribution as a bar chart.

    Args:
        label_counts: Dictionary mapping label names to counts
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    # Filter out 'O' tag for cleaner visualization
    filtered_counts = {k: v for k, v in label_counts.items() if k != 'O'}

    if not filtered_counts:
        print("No entity labels to plot")
        return

    labels = list(filtered_counts.keys())
    counts = list(filtered_counts.values())

    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Entity Label')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels if many
    if len(labels) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Entity distribution plot saved to {output_path}")


def plot_confusion_matrix(
    confusion_matrix: Dict[str, Dict[str, int]],
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: Nested dictionary of confusion counts
        output_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize by row (true labels)
    """
    check_matplotlib()

    labels = sorted(confusion_matrix.keys())
    n_labels = len(labels)

    # Build matrix
    matrix = np.zeros((n_labels, n_labels))
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = confusion_matrix[true_label].get(pred_label, 0)

    # Normalize if requested
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # Add text annotations
    thresh = matrix.max() / 2.
    for i in range(n_labels):
        for j in range(n_labels):
            value = matrix[i, j]
            text = f'{value:.2f}' if normalize else f'{int(value)}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if value > thresh else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_path}")


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    output_path: str,
    title: str = "Attention Weights"
) -> None:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: 2D array of attention weights (seq_len x seq_len)
        tokens: List of tokens
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    seq_len = len(tokens)
    attention_weights = attention_weights[:seq_len, :seq_len]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attention Weight', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Attention heatmap saved to {output_path}")


def plot_length_performance(
    length_analysis: Dict[int, Dict],
    output_path: str,
    title: str = "Performance by Entity Length"
) -> None:
    """
    Plot performance metrics by entity length.

    Args:
        length_analysis: Dictionary mapping length to performance metrics
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    lengths = sorted(length_analysis.keys())
    f1_scores = [length_analysis[l]['f1'] for l in lengths]
    precisions = [length_analysis[l]['precision'] for l in lengths]
    recalls = [length_analysis[l]['recall'] for l in lengths]
    counts = [length_analysis[l]['true_count'] for l in lengths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: F1, Precision, Recall by length
    x = np.arange(len(lengths))
    width = 0.25

    bars1 = ax1.bar(x - width, f1_scores, width, label='F1', color='green', alpha=0.8)
    bars2 = ax1.bar(x, precisions, width, label='Precision', color='blue', alpha=0.8)
    bars3 = ax1.bar(x + width, recalls, width, label='Recall', color='red', alpha=0.8)

    ax1.set_xlabel('Entity Length (tokens)')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics by Entity Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lengths)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Entity count distribution
    ax2.bar(lengths, counts, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Entity Length (tokens)')
    ax2.set_ylabel('Count')
    ax2.set_title('Entity Count by Length')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for i, (length, count) in enumerate(zip(lengths, counts)):
        ax2.text(length, count, str(count), ha='center', va='bottom', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Length performance plot saved to {output_path}")


def plot_per_entity_performance(
    entity_metrics: Dict[str, Dict],
    output_path: str,
    title: str = "Per-Entity Type Performance"
) -> None:
    """
    Plot performance metrics for each entity type.

    Args:
        entity_metrics: Dictionary mapping entity type to metrics
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    entity_types = sorted(entity_metrics.keys())
    f1_scores = [entity_metrics[e]['f1'] for e in entity_types]
    precisions = [entity_metrics[e]['precision'] for e in entity_types]
    recalls = [entity_metrics[e]['recall'] for e in entity_types]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(entity_types))
    width = 0.25

    bars1 = ax.bar(x - width, f1_scores, width, label='F1', color='green', alpha=0.8)
    bars2 = ax.bar(x, precisions, width, label='Precision', color='blue', alpha=0.8)
    bars3 = ax.bar(x + width, recalls, width, label='Recall', color='red', alpha=0.8)

    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(entity_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Add F1 value labels
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-entity performance plot saved to {output_path}")


def plot_error_distribution(
    error_analysis: Dict,
    output_path: str,
    title: str = "Error Distribution"
) -> None:
    """
    Plot error type distribution.

    Args:
        error_analysis: Error analysis dictionary
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    error_types = ['Boundary\nErrors', 'Type\nErrors', 'Missed\nEntities', 'Spurious\nEntities']
    counts = [
        error_analysis['boundary_errors'],
        error_analysis['type_errors'],
        error_analysis['missed_entities'],
        error_analysis['spurious_entities']
    ]

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(error_types, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Count')
    ax1.set_title('Error Count by Type')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)

    # Pie chart
    total = sum(counts)
    if total > 0:
        ax2.pie(counts, labels=error_types, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
        ax2.set_title('Error Proportion')
    else:
        ax2.text(0.5, 0.5, 'No errors', ha='center', va='center', fontsize=12)
        ax2.set_title('Error Proportion')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Error distribution plot saved to {output_path}")


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Ablation Study Results"
) -> None:
    """
    Plot ablation study results.

    Args:
        results: Dictionary mapping model name to metrics
            Expected format: {'model_name': {'f1': float, 'precision': float, 'recall': float}}
        output_path: Path to save the plot
        title: Plot title
    """
    check_matplotlib()

    models = list(results.keys())
    f1_scores = [results[m]['f1'] for m in models]
    precisions = [results[m].get('precision', 0) for m in models]
    recalls = [results[m].get('recall', 0) for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, f1_scores, width, label='F1', color='green', alpha=0.8)
    bars2 = ax.bar(x, precisions, width, label='Precision', color='blue', alpha=0.8)
    bars3 = ax.bar(x + width, recalls, width, label='Recall', color='red', alpha=0.8)

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add F1 value labels
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Ablation results plot saved to {output_path}")


def create_all_visualizations(
    history: Dict,
    error_analysis: Dict,
    length_analysis: Dict,
    entity_metrics: Dict,
    confusion_matrix: Dict,
    label_counts: Dict,
    output_dir: str
) -> None:
    """
    Create all visualization plots.

    Args:
        history: Training history dictionary
        error_analysis: Error analysis dictionary
        length_analysis: Length analysis dictionary
        entity_metrics: Per-entity metrics dictionary
        confusion_matrix: Confusion matrix dictionary
        label_counts: Label count dictionary
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # Training curves
    if history:
        plot_training_curves(
            history,
            os.path.join(output_dir, 'training_curves.png'),
            'Training Progress'
        )

    # Entity distribution
    if label_counts:
        plot_entity_distribution(
            label_counts,
            os.path.join(output_dir, 'entity_distribution.png'),
            'Entity Label Distribution'
        )

    # Confusion matrix
    if confusion_matrix:
        plot_confusion_matrix(
            confusion_matrix,
            os.path.join(output_dir, 'confusion_matrix.png'),
            'Token-Level Confusion Matrix'
        )

    # Error distribution
    if error_analysis:
        plot_error_distribution(
            error_analysis,
            os.path.join(output_dir, 'error_distribution.png'),
            'Error Type Distribution'
        )

    # Length performance
    if length_analysis:
        plot_length_performance(
            length_analysis,
            os.path.join(output_dir, 'length_performance.png'),
            'Performance by Entity Length'
        )

    # Per-entity performance
    if entity_metrics:
        plot_per_entity_performance(
            entity_metrics,
            os.path.join(output_dir, 'per_entity_performance.png'),
            'Per-Entity Type Performance'
        )

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with dummy data
    history = {
        'train_loss': [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28],
        'dev_loss': [2.3, 1.6, 1.1, 0.85, 0.72, 0.65, 0.58, 0.55, 0.54, 0.53],
        'dev_f1': [0.45, 0.58, 0.67, 0.73, 0.78, 0.81, 0.83, 0.84, 0.845, 0.85],
        'dev_precision': [0.42, 0.55, 0.65, 0.72, 0.77, 0.80, 0.82, 0.83, 0.835, 0.84],
        'dev_recall': [0.48, 0.61, 0.69, 0.74, 0.79, 0.82, 0.84, 0.85, 0.855, 0.86],
        'learning_rate': [0.001, 0.001, 0.001, 0.0008, 0.0006, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001]
    }

    # Create example plots
    output_dir = 'visualization_test'
    os.makedirs(output_dir, exist_ok=True)

    if MATPLOTLIB_AVAILABLE:
        plot_training_curves(
            history,
            os.path.join(output_dir, 'training_curves.png'),
            'Example Training Progress'
        )

        # Example entity distribution
        label_counts = {
            'B-Chemical': 5000,
            'I-Chemical': 3000,
            'B-Disease': 4500,
            'I-Disease': 2800,
            'O': 50000
        }
        plot_entity_distribution(
            label_counts,
            os.path.join(output_dir, 'entity_distribution.png'),
            'Example Entity Distribution'
        )

        print(f"Example plots saved to {output_dir}/")
    else:
        print("Matplotlib not available - skipping visualization examples")
