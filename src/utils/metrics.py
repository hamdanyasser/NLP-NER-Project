"""
Evaluation metrics for Named Entity Recognition.

This module provides:
- Entity-level precision, recall, and F1 score
- Per-entity-type metrics
- Token-level accuracy
- Confusion matrix utilities
"""

from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np


def get_entities(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """
    Extract entities from BIO-tagged sequence.

    Args:
        tags: List of BIO tags

    Returns:
        Set of (entity_type, start_idx, end_idx) tuples
    """
    entities = set()
    current_entity = None
    start_idx = None

    for idx, tag in enumerate(tags):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity is not None:
                entities.add((current_entity, start_idx, idx - 1))

            # Start new entity
            current_entity = tag[2:]  # Remove 'B-' prefix
            start_idx = idx

        elif tag.startswith('I-'):
            # Continue current entity
            entity_type = tag[2:]  # Remove 'I-' prefix

            # Check if this is continuation of current entity
            if current_entity != entity_type:
                # Save previous entity if exists
                if current_entity is not None:
                    entities.add((current_entity, start_idx, idx - 1))

                # Start new entity (treat I- as B- if not following matching B-)
                current_entity = entity_type
                start_idx = idx

        else:  # 'O' tag or other
            # Save previous entity if exists
            if current_entity is not None:
                entities.add((current_entity, start_idx, idx - 1))
                current_entity = None
                start_idx = None

    # Don't forget last entity
    if current_entity is not None:
        entities.add((current_entity, start_idx, len(tags) - 1))

    return entities


def compute_metrics(true_tags: List[List[str]],
                    pred_tags: List[List[str]]) -> Dict[str, float]:
    """
    Compute entity-level precision, recall, and F1 score.

    Args:
        true_tags: List of true tag sequences
        pred_tags: List of predicted tag sequences

    Returns:
        Dictionary with overall and per-entity metrics
    """
    assert len(true_tags) == len(pred_tags), "Number of sequences must match"

    # Count true positives, false positives, false negatives
    true_entities_all = []
    pred_entities_all = []

    # Per-entity-type counts
    entity_types = set()
    tp_by_type = defaultdict(int)
    fp_by_type = defaultdict(int)
    fn_by_type = defaultdict(int)

    for true_seq, pred_seq in zip(true_tags, pred_tags):
        # Extract entities
        true_entities = get_entities(true_seq)
        pred_entities = get_entities(pred_seq)

        true_entities_all.extend(true_entities)
        pred_entities_all.extend(pred_entities)

        # Collect entity types
        for entity_type, _, _ in true_entities | pred_entities:
            entity_types.add(entity_type)

        # Count matches
        for entity in pred_entities:
            entity_type = entity[0]
            if entity in true_entities:
                tp_by_type[entity_type] += 1
            else:
                fp_by_type[entity_type] += 1

        for entity in true_entities:
            entity_type = entity[0]
            if entity not in pred_entities:
                fn_by_type[entity_type] += 1

    # Compute overall metrics
    tp_overall = sum(tp_by_type.values())
    fp_overall = sum(fp_by_type.values())
    fn_overall = sum(fn_by_type.values())

    precision_overall = tp_overall / (tp_overall + fp_overall) if (tp_overall + fp_overall) > 0 else 0.0
    recall_overall = tp_overall / (tp_overall + fn_overall) if (tp_overall + fn_overall) > 0 else 0.0
    f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall) \
        if (precision_overall + recall_overall) > 0 else 0.0

    # Compute per-entity metrics
    metrics_by_type = {}
    for entity_type in sorted(entity_types):
        tp = tp_by_type[entity_type]
        fp = fp_by_type[entity_type]
        fn = fn_by_type[entity_type]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_by_type[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn  # Number of true entities
        }

    # Compile results
    results = {
        'precision': precision_overall,
        'recall': recall_overall,
        'f1': f1_overall,
        'num_true_entities': len(true_entities_all),
        'num_pred_entities': len(pred_entities_all),
        'tp': tp_overall,
        'fp': fp_overall,
        'fn': fn_overall,
        'by_type': metrics_by_type
    }

    return results


def compute_token_accuracy(true_tags: List[List[str]],
                           pred_tags: List[List[str]],
                           ignore_pad: bool = True,
                           pad_tag: str = '<PAD>') -> float:
    """
    Compute token-level accuracy.

    Args:
        true_tags: List of true tag sequences
        pred_tags: List of predicted tag sequences
        ignore_pad: Whether to ignore padding tokens
        pad_tag: Padding tag to ignore

    Returns:
        Token-level accuracy
    """
    assert len(true_tags) == len(pred_tags), "Number of sequences must match"

    total_tokens = 0
    correct_tokens = 0

    for true_seq, pred_seq in zip(true_tags, pred_tags):
        assert len(true_seq) == len(pred_seq), "Sequence lengths must match"

        for true_tag, pred_tag in zip(true_seq, pred_seq):
            if ignore_pad and (true_tag == pad_tag or pred_tag == pad_tag):
                continue

            total_tokens += 1
            if true_tag == pred_tag:
                correct_tokens += 1

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return accuracy


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary with metrics from compute_metrics
        prefix: Prefix for print statements (e.g., "Train", "Dev", "Test")
    """
    print("\n" + "=" * 60)
    if prefix:
        print(f"{prefix} Metrics")
    else:
        print("Evaluation Metrics")
    print("=" * 60)

    print(f"\nOverall Performance:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nEntity Counts:")
    print(f"  True entities:      {metrics['num_true_entities']}")
    print(f"  Predicted entities: {metrics['num_pred_entities']}")
    print(f"  True positives:     {metrics['tp']}")
    print(f"  False positives:    {metrics['fp']}")
    print(f"  False negatives:    {metrics['fn']}")

    if 'by_type' in metrics and metrics['by_type']:
        print(f"\nPer-Entity-Type Performance:")
        print(f"{'Entity Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 70)

        for entity_type, type_metrics in sorted(metrics['by_type'].items()):
            print(f"{entity_type:<20} "
                  f"{type_metrics['precision']:<12.4f} "
                  f"{type_metrics['recall']:<12.4f} "
                  f"{type_metrics['f1']:<12.4f} "
                  f"{type_metrics['support']:<10}")

    print("=" * 60)


def get_classification_report(true_tags: List[List[str]],
                               pred_tags: List[List[str]]) -> str:
    """
    Generate a detailed classification report.

    Args:
        true_tags: List of true tag sequences
        pred_tags: List of predicted tag sequences

    Returns:
        Formatted report string
    """
    metrics = compute_metrics(true_tags, pred_tags)
    token_acc = compute_token_accuracy(true_tags, pred_tags)

    report = []
    report.append("=" * 70)
    report.append("Named Entity Recognition - Classification Report")
    report.append("=" * 70)
    report.append(f"\nToken-level Accuracy: {token_acc:.4f}")
    report.append(f"\nEntity-level Metrics:")
    report.append(f"  Precision: {metrics['precision']:.4f}")
    report.append(f"  Recall:    {metrics['recall']:.4f}")
    report.append(f"  F1 Score:  {metrics['f1']:.4f}")

    report.append(f"\nPer-Entity Performance:")
    report.append(f"{'Entity Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    report.append("-" * 70)

    for entity_type, type_metrics in sorted(metrics['by_type'].items()):
        report.append(f"{entity_type:<20} "
                      f"{type_metrics['precision']:<12.4f} "
                      f"{type_metrics['recall']:<12.4f} "
                      f"{type_metrics['f1']:<12.4f} "
                      f"{type_metrics['support']:<10}")

    report.append("=" * 70)

    return "\n".join(report)
