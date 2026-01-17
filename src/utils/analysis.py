"""
Error analysis tools for NER models.

This module provides comprehensive error analysis including:
- Confusion matrix computation
- Error categorization (boundary, type, missed, spurious)
- Entity length analysis
- Per-entity-type breakdown
- HTML visualization of predictions

Built from scratch for NLP Course Project.

Author: NLP Course Project
"""

import os
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json


def extract_entities_with_positions(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract entities with their positions from BIO tags.

    Args:
        tags: List of BIO tags

    Returns:
        List of (entity_type, start_idx, end_idx) tuples
    """
    entities = []
    current_entity = None
    current_start = None

    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # Save previous entity
            if current_entity is not None:
                entities.append((current_entity, current_start, i - 1))

            # Start new entity
            current_entity = tag[2:]  # Remove 'B-' prefix
            current_start = i

        elif tag.startswith('I-'):
            entity_type = tag[2:]
            # Continue current entity if types match
            if current_entity != entity_type:
                # Type mismatch - save previous and start new
                if current_entity is not None:
                    entities.append((current_entity, current_start, i - 1))
                current_entity = entity_type
                current_start = i

        else:  # 'O' tag
            if current_entity is not None:
                entities.append((current_entity, current_start, i - 1))
                current_entity = None
                current_start = None

    # Don't forget last entity
    if current_entity is not None:
        entities.append((current_entity, current_start, len(tags) - 1))

    return entities


def compute_confusion_matrix(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
    label_names: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Compute token-level confusion matrix.

    Args:
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences
        label_names: List of label names

    Returns:
        Nested dictionary: confusion_matrix[true_label][pred_label] = count
    """
    # Initialize confusion matrix
    confusion = {label: {l: 0 for l in label_names} for label in label_names}

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            if true_tag in confusion and pred_tag in confusion[true_tag]:
                confusion[true_tag][pred_tag] += 1

    return confusion


def categorize_errors(
    tokens: List[str],
    true_tags: List[str],
    pred_tags: List[str]
) -> Dict[str, List[Dict]]:
    """
    Categorize errors into types.

    Error types:
    - boundary_error: Entity partially correct (wrong boundaries)
    - type_error: Entity boundaries correct but wrong type
    - missed: True entity not predicted at all
    - spurious: Predicted entity doesn't exist in true labels

    Args:
        tokens: List of tokens
        true_tags: List of true BIO tags
        pred_tags: List of predicted BIO tags

    Returns:
        Dictionary with error categories and their instances
    """
    errors = {
        'boundary_error': [],
        'type_error': [],
        'missed': [],
        'spurious': []
    }

    # Extract entities
    true_entities = extract_entities_with_positions(true_tags)
    pred_entities = extract_entities_with_positions(pred_tags)

    # Create position-based lookups
    true_positions = {(e[1], e[2]): e[0] for e in true_entities}
    pred_positions = {(e[1], e[2]): e[0] for e in pred_entities}

    # Track matched entities
    matched_true = set()
    matched_pred = set()

    # Check for exact matches and type errors
    for true_pos, true_type in true_positions.items():
        if true_pos in pred_positions:
            pred_type = pred_positions[true_pos]
            if true_type == pred_type:
                # Exact match
                matched_true.add(true_pos)
                matched_pred.add(true_pos)
            else:
                # Type error (boundaries match, type doesn't)
                errors['type_error'].append({
                    'tokens': tokens[true_pos[0]:true_pos[1] + 1],
                    'start': true_pos[0],
                    'end': true_pos[1],
                    'true_type': true_type,
                    'pred_type': pred_type
                })
                matched_true.add(true_pos)
                matched_pred.add(true_pos)

    # Check for boundary errors and missed entities
    for true_pos, true_type in true_positions.items():
        if true_pos in matched_true:
            continue

        # Check if there's an overlapping prediction
        found_overlap = False
        for pred_pos, pred_type in pred_positions.items():
            if pred_pos in matched_pred:
                continue

            # Check for overlap
            if (true_pos[0] <= pred_pos[1] and pred_pos[0] <= true_pos[1]):
                # Overlapping - boundary error
                errors['boundary_error'].append({
                    'tokens': tokens[min(true_pos[0], pred_pos[0]):max(true_pos[1], pred_pos[1]) + 1],
                    'true_span': (true_pos[0], true_pos[1]),
                    'pred_span': (pred_pos[0], pred_pos[1]),
                    'true_type': true_type,
                    'pred_type': pred_type,
                    'true_tokens': tokens[true_pos[0]:true_pos[1] + 1],
                    'pred_tokens': tokens[pred_pos[0]:pred_pos[1] + 1]
                })
                matched_true.add(true_pos)
                matched_pred.add(pred_pos)
                found_overlap = True
                break

        if not found_overlap:
            # Missed entity
            errors['missed'].append({
                'tokens': tokens[true_pos[0]:true_pos[1] + 1],
                'start': true_pos[0],
                'end': true_pos[1],
                'type': true_type
            })
            matched_true.add(true_pos)

    # Check for spurious predictions
    for pred_pos, pred_type in pred_positions.items():
        if pred_pos not in matched_pred:
            errors['spurious'].append({
                'tokens': tokens[pred_pos[0]:pred_pos[1] + 1],
                'start': pred_pos[0],
                'end': pred_pos[1],
                'type': pred_type
            })

    return errors


def analyze_errors(
    tokens_list: List[List[str]],
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]]
) -> Dict[str, any]:
    """
    Perform comprehensive error analysis.

    Args:
        tokens_list: List of token sequences
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences

    Returns:
        Dictionary with error analysis results
    """
    all_errors = {
        'boundary_error': [],
        'type_error': [],
        'missed': [],
        'spurious': []
    }

    # Collect errors from all sentences
    for i, (tokens, true_tags, pred_tags) in enumerate(
        zip(tokens_list, true_tags_list, pred_tags_list)
    ):
        errors = categorize_errors(tokens, true_tags, pred_tags)

        for error_type, error_list in errors.items():
            for error in error_list:
                error['sentence_idx'] = i
                all_errors[error_type].append(error)

    # Compute statistics
    stats = {
        'total_errors': sum(len(v) for v in all_errors.values()),
        'boundary_errors': len(all_errors['boundary_error']),
        'type_errors': len(all_errors['type_error']),
        'missed_entities': len(all_errors['missed']),
        'spurious_entities': len(all_errors['spurious']),
        'errors': all_errors
    }

    # Compute error distribution by entity type
    type_distribution = defaultdict(lambda: {'missed': 0, 'spurious': 0, 'type_error': 0, 'boundary_error': 0})

    for error in all_errors['missed']:
        type_distribution[error['type']]['missed'] += 1

    for error in all_errors['spurious']:
        type_distribution[error['type']]['spurious'] += 1

    for error in all_errors['type_error']:
        type_distribution[error['true_type']]['type_error'] += 1

    for error in all_errors['boundary_error']:
        type_distribution[error['true_type']]['boundary_error'] += 1

    stats['error_by_type'] = dict(type_distribution)

    return stats


def get_entity_length_analysis(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]]
) -> Dict[str, Dict]:
    """
    Analyze performance by entity length.

    Args:
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences

    Returns:
        Dictionary with length-based analysis
    """
    # Group by length
    length_stats = defaultdict(lambda: {'true_count': 0, 'pred_count': 0, 'correct': 0})

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
        true_entities = extract_entities_with_positions(true_tags)
        pred_entities = extract_entities_with_positions(pred_tags)

        # Count true entities by length
        for entity_type, start, end in true_entities:
            length = end - start + 1
            length_stats[length]['true_count'] += 1

        # Count predicted entities by length
        for entity_type, start, end in pred_entities:
            length = end - start + 1
            length_stats[length]['pred_count'] += 1

        # Count correct predictions by length
        true_set = {(e[0], e[1], e[2]) for e in true_entities}
        pred_set = {(e[0], e[1], e[2]) for e in pred_entities}

        for entity in true_set & pred_set:
            length = entity[2] - entity[1] + 1
            length_stats[length]['correct'] += 1

    # Compute precision, recall, F1 by length
    results = {}
    for length, stats in sorted(length_stats.items()):
        precision = stats['correct'] / stats['pred_count'] if stats['pred_count'] > 0 else 0
        recall = stats['correct'] / stats['true_count'] if stats['true_count'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[length] = {
            'true_count': stats['true_count'],
            'pred_count': stats['pred_count'],
            'correct': stats['correct'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results


def get_per_entity_metrics(
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]]
) -> Dict[str, Dict]:
    """
    Compute metrics for each entity type.

    Args:
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences

    Returns:
        Dictionary with per-entity metrics
    """
    entity_stats = defaultdict(lambda: {'true_count': 0, 'pred_count': 0, 'correct': 0})

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
        true_entities = extract_entities_with_positions(true_tags)
        pred_entities = extract_entities_with_positions(pred_tags)

        # Count true entities
        for entity_type, start, end in true_entities:
            entity_stats[entity_type]['true_count'] += 1

        # Count predicted entities
        for entity_type, start, end in pred_entities:
            entity_stats[entity_type]['pred_count'] += 1

        # Count correct predictions
        true_set = {(e[0], e[1], e[2]) for e in true_entities}
        pred_set = {(e[0], e[1], e[2]) for e in pred_entities}

        for entity in true_set & pred_set:
            entity_stats[entity[0]]['correct'] += 1

    # Compute metrics
    results = {}
    for entity_type, stats in sorted(entity_stats.items()):
        precision = stats['correct'] / stats['pred_count'] if stats['pred_count'] > 0 else 0
        recall = stats['correct'] / stats['true_count'] if stats['true_count'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[entity_type] = {
            'true_count': stats['true_count'],
            'pred_count': stats['pred_count'],
            'correct': stats['correct'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results


def generate_html_visualization(
    tokens_list: List[List[str]],
    true_tags_list: List[List[str]],
    pred_tags_list: List[List[str]],
    output_path: str,
    max_sentences: int = 100
) -> None:
    """
    Generate HTML visualization of predictions.

    Args:
        tokens_list: List of token sequences
        true_tags_list: List of true tag sequences
        pred_tags_list: List of predicted tag sequences
        output_path: Path to save HTML file
        max_sentences: Maximum number of sentences to visualize
    """
    # CSS styles
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .sentence { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .sentence-header { font-weight: bold; color: #666; margin-bottom: 10px; }
        .token { display: inline-block; margin: 2px; padding: 4px 8px; border-radius: 3px; }
        .correct { background: #d4edda; border: 1px solid #28a745; }
        .error { background: #f8d7da; border: 1px solid #dc3545; }
        .missed { background: #fff3cd; border: 1px solid #ffc107; }
        .spurious { background: #cce5ff; border: 1px solid #007bff; }
        .outside { background: #e9ecef; border: 1px solid #dee2e6; }
        .tag { font-size: 10px; color: #666; }
        .legend { margin: 20px 0; padding: 15px; background: white; border-radius: 5px; }
        .legend span { margin-right: 20px; padding: 5px 10px; border-radius: 3px; }
        .stats { background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .stats table { width: 100%; border-collapse: collapse; }
        .stats th, .stats td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .stats th { background: #f0f0f0; }
    </style>
    """

    # Legend
    legend = """
    <div class="legend">
        <strong>Legend:</strong>
        <span class="correct">Correct</span>
        <span class="error">Wrong Type/Boundary</span>
        <span class="missed">Missed Entity</span>
        <span class="spurious">Spurious Prediction</span>
        <span class="outside">Outside (O)</span>
    </div>
    """

    # Build HTML content
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>NER Prediction Visualization</title>",
        css,
        "</head>",
        "<body>",
        "<h1>NER Prediction Visualization</h1>",
        legend
    ]

    # Add statistics
    error_analysis = analyze_errors(tokens_list[:max_sentences],
                                   true_tags_list[:max_sentences],
                                   pred_tags_list[:max_sentences])

    html_content.append("<div class='stats'>")
    html_content.append("<h2>Error Statistics</h2>")
    html_content.append("<table>")
    html_content.append("<tr><th>Error Type</th><th>Count</th></tr>")
    html_content.append(f"<tr><td>Boundary Errors</td><td>{error_analysis['boundary_errors']}</td></tr>")
    html_content.append(f"<tr><td>Type Errors</td><td>{error_analysis['type_errors']}</td></tr>")
    html_content.append(f"<tr><td>Missed Entities</td><td>{error_analysis['missed_entities']}</td></tr>")
    html_content.append(f"<tr><td>Spurious Entities</td><td>{error_analysis['spurious_entities']}</td></tr>")
    html_content.append(f"<tr><th>Total Errors</th><th>{error_analysis['total_errors']}</th></tr>")
    html_content.append("</table>")
    html_content.append("</div>")

    # Add sentences
    for idx, (tokens, true_tags, pred_tags) in enumerate(
        zip(tokens_list[:max_sentences], true_tags_list[:max_sentences], pred_tags_list[:max_sentences])
    ):
        html_content.append(f"<div class='sentence'>")
        html_content.append(f"<div class='sentence-header'>Sentence {idx + 1}</div>")

        # True labels row
        html_content.append("<div><strong>True:</strong> ")
        for token, tag in zip(tokens, true_tags):
            if tag == 'O':
                html_content.append(f"<span class='token outside'>{token}</span>")
            else:
                html_content.append(f"<span class='token correct'>{token} <span class='tag'>[{tag}]</span></span>")
        html_content.append("</div>")

        # Predicted labels row
        html_content.append("<div><strong>Pred:</strong> ")
        for i, (token, true_tag, pred_tag) in enumerate(zip(tokens, true_tags, pred_tags)):
            if true_tag == pred_tag:
                if pred_tag == 'O':
                    html_content.append(f"<span class='token outside'>{token}</span>")
                else:
                    html_content.append(f"<span class='token correct'>{token} <span class='tag'>[{pred_tag}]</span></span>")
            elif pred_tag == 'O' and true_tag != 'O':
                html_content.append(f"<span class='token missed'>{token} <span class='tag'>[{pred_tag}]</span></span>")
            elif pred_tag != 'O' and true_tag == 'O':
                html_content.append(f"<span class='token spurious'>{token} <span class='tag'>[{pred_tag}]</span></span>")
            else:
                html_content.append(f"<span class='token error'>{token} <span class='tag'>[{pred_tag}]</span></span>")
        html_content.append("</div>")

        html_content.append("</div>")

    html_content.extend(["</body>", "</html>"])

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))

    print(f"HTML visualization saved to {output_path}")


def print_error_analysis(analysis: Dict) -> None:
    """
    Print error analysis in a formatted way.

    Args:
        analysis: Error analysis dictionary from analyze_errors()
    """
    print("\n" + "=" * 70)
    print("Error Analysis Report")
    print("=" * 70)

    print(f"\nTotal Errors: {analysis['total_errors']}")
    print(f"  - Boundary Errors: {analysis['boundary_errors']}")
    print(f"  - Type Errors: {analysis['type_errors']}")
    print(f"  - Missed Entities: {analysis['missed_entities']}")
    print(f"  - Spurious Entities: {analysis['spurious_entities']}")

    if analysis['error_by_type']:
        print("\nError Distribution by Entity Type:")
        print("-" * 50)
        print(f"{'Type':<15} {'Missed':<10} {'Spurious':<10} {'Type Err':<10} {'Boundary':<10}")
        print("-" * 50)

        for entity_type, errors in sorted(analysis['error_by_type'].items()):
            print(f"{entity_type:<15} {errors['missed']:<10} {errors['spurious']:<10} "
                  f"{errors['type_error']:<10} {errors['boundary_error']:<10}")

    print("=" * 70)


def print_length_analysis(length_analysis: Dict) -> None:
    """
    Print length analysis in a formatted way.

    Args:
        length_analysis: Length analysis dictionary
    """
    print("\n" + "=" * 70)
    print("Performance by Entity Length")
    print("=" * 70)
    print(f"{'Length':<8} {'True':<8} {'Pred':<8} {'Correct':<10} {'P':<8} {'R':<8} {'F1':<8}")
    print("-" * 70)

    for length, stats in sorted(length_analysis.items()):
        print(f"{length:<8} {stats['true_count']:<8} {stats['pred_count']:<8} "
              f"{stats['correct']:<10} {stats['precision']:.2%}  {stats['recall']:.2%}  {stats['f1']:.2%}")

    print("=" * 70)


def print_per_entity_metrics(entity_metrics: Dict) -> None:
    """
    Print per-entity metrics in a formatted way.

    Args:
        entity_metrics: Per-entity metrics dictionary
    """
    print("\n" + "=" * 70)
    print("Per-Entity Type Metrics")
    print("=" * 70)
    print(f"{'Entity Type':<15} {'True':<8} {'Pred':<8} {'Correct':<10} {'P':<8} {'R':<8} {'F1':<8}")
    print("-" * 70)

    for entity_type, stats in sorted(entity_metrics.items()):
        print(f"{entity_type:<15} {stats['true_count']:<8} {stats['pred_count']:<8} "
              f"{stats['correct']:<10} {stats['precision']:.2%}  {stats['recall']:.2%}  {stats['f1']:.2%}")

    print("=" * 70)


def save_analysis_report(
    analysis: Dict,
    length_analysis: Dict,
    entity_metrics: Dict,
    output_path: str
) -> None:
    """
    Save complete analysis report to JSON.

    Args:
        analysis: Error analysis dictionary
        length_analysis: Length analysis dictionary
        entity_metrics: Per-entity metrics dictionary
        output_path: Path to save JSON file
    """
    # Convert error examples to be JSON serializable
    serializable_analysis = {
        'total_errors': analysis['total_errors'],
        'boundary_errors': analysis['boundary_errors'],
        'type_errors': analysis['type_errors'],
        'missed_entities': analysis['missed_entities'],
        'spurious_entities': analysis['spurious_entities'],
        'error_by_type': analysis['error_by_type'],
        'sample_errors': {
            'boundary_error': analysis['errors']['boundary_error'][:10],
            'type_error': analysis['errors']['type_error'][:10],
            'missed': analysis['errors']['missed'][:10],
            'spurious': analysis['errors']['spurious'][:10]
        }
    }

    report = {
        'error_analysis': serializable_analysis,
        'length_analysis': {str(k): v for k, v in length_analysis.items()},
        'entity_metrics': entity_metrics
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Analysis report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    true_tags = [
        ['B-Chemical', 'I-Chemical', 'O', 'B-Disease', 'O'],
        ['O', 'B-Chemical', 'O', 'O', 'B-Disease', 'I-Disease']
    ]

    pred_tags = [
        ['B-Chemical', 'O', 'O', 'B-Disease', 'O'],  # Boundary error on Chemical
        ['O', 'B-Chemical', 'O', 'B-Disease', 'B-Disease', 'I-Disease']  # Spurious Disease
    ]

    tokens = [
        ['Aspirin', 'acid', 'treats', 'headache', '.'],
        ['The', 'ibuprofen', 'reduces', 'chronic', 'back', 'pain']
    ]

    # Run analysis
    analysis = analyze_errors(tokens, true_tags, pred_tags)
    print_error_analysis(analysis)

    length_analysis = get_entity_length_analysis(true_tags, pred_tags)
    print_length_analysis(length_analysis)

    entity_metrics = get_per_entity_metrics(true_tags, pred_tags)
    print_per_entity_metrics(entity_metrics)
