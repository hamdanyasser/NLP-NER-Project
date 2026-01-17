"""
Unit tests for metrics computation.

Tests:
- Entity extraction from BIO tags
- Precision, recall, F1 computation
- Edge cases in entity matching
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    get_entities,
    compute_metrics,
    compute_token_accuracy
)

# Alias for test compatibility
def extract_entities(tags):
    """Wrapper to convert set to list for tests."""
    entities = get_entities(tags)
    return list(entities)

def compute_token_level_metrics(true_tags, pred_tags):
    """Wrapper for token-level metrics."""
    accuracy = compute_token_accuracy(true_tags, pred_tags)
    return {'accuracy': accuracy}


class TestEntityExtraction:
    """Test suite for entity extraction from BIO tags."""

    def test_extract_single_entity(self):
        """Test extracting a single entity."""
        tags = ['O', 'B-Chemical', 'I-Chemical', 'O']
        entities = extract_entities(tags)

        assert len(entities) == 1
        assert entities[0] == ('Chemical', 1, 2)

    def test_extract_multiple_entities(self):
        """Test extracting multiple entities."""
        tags = ['B-Chemical', 'O', 'B-Disease', 'I-Disease']
        entities = extract_entities(tags)

        assert len(entities) == 2
        assert ('Chemical', 0, 0) in entities
        assert ('Disease', 2, 3) in entities

    def test_extract_consecutive_entities(self):
        """Test extracting consecutive entities of same type."""
        tags = ['B-Chemical', 'B-Chemical']
        entities = extract_entities(tags)

        assert len(entities) == 2

    def test_extract_no_entities(self):
        """Test with no entities (all O tags)."""
        tags = ['O', 'O', 'O']
        entities = extract_entities(tags)

        assert len(entities) == 0

    def test_extract_all_entity(self):
        """Test with all tokens being one entity."""
        tags = ['B-Disease', 'I-Disease', 'I-Disease']
        entities = extract_entities(tags)

        assert len(entities) == 1
        assert entities[0] == ('Disease', 0, 2)

    def test_extract_handles_missing_b_tag(self):
        """Test handling I-tag without B-tag."""
        tags = ['O', 'I-Chemical', 'I-Chemical', 'O']
        entities = extract_entities(tags)

        # Should still extract an entity (treating I as B)
        assert len(entities) >= 1


class TestMetricsComputation:
    """Test suite for metrics computation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        true_tags = [['B-Chemical', 'I-Chemical', 'O', 'B-Disease']]
        pred_tags = [['B-Chemical', 'I-Chemical', 'O', 'B-Disease']]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_no_predictions(self):
        """Test metrics with no predictions."""
        true_tags = [['B-Chemical', 'O', 'B-Disease']]
        pred_tags = [['O', 'O', 'O']]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        true_tags = [['B-Chemical', 'O']]
        pred_tags = [['O', 'B-Disease']]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_partial_match(self):
        """Test metrics with partial matches."""
        true_tags = [['B-Chemical', 'I-Chemical', 'O', 'B-Disease']]
        pred_tags = [['B-Chemical', 'I-Chemical', 'O', 'O']]  # Missed Disease

        metrics = compute_metrics(true_tags, pred_tags)

        # 1 correct / 1 predicted = 1.0 precision
        # 1 correct / 2 true = 0.5 recall
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 0.5

    def test_multiple_sequences(self):
        """Test metrics across multiple sequences."""
        true_tags = [
            ['B-Chemical', 'O'],
            ['O', 'B-Disease']
        ]
        pred_tags = [
            ['B-Chemical', 'O'],
            ['O', 'B-Disease']
        ]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['f1'] == 1.0

    def test_boundary_error_not_counted(self):
        """Test that boundary errors are not counted as correct."""
        true_tags = [['B-Chemical', 'I-Chemical', 'O']]
        pred_tags = [['B-Chemical', 'O', 'O']]  # Wrong boundary

        metrics = compute_metrics(true_tags, pred_tags)

        # Boundary error should result in no correct match
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0

    def test_type_error_not_counted(self):
        """Test that type errors are not counted as correct."""
        true_tags = [['B-Chemical', 'I-Chemical']]
        pred_tags = [['B-Disease', 'I-Disease']]  # Wrong type, same span

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['f1'] == 0.0

    def test_empty_sequences(self):
        """Test with empty sequences."""
        true_tags = [[]]
        pred_tags = [[]]

        metrics = compute_metrics(true_tags, pred_tags)

        # No entities to evaluate
        assert metrics['f1'] == 0.0


class TestTokenLevelMetrics:
    """Test suite for token-level metrics."""

    def test_token_level_perfect(self):
        """Test token-level metrics with perfect predictions."""
        true_tags = [['B-Chemical', 'I-Chemical', 'O']]
        pred_tags = [['B-Chemical', 'I-Chemical', 'O']]

        metrics = compute_token_level_metrics(true_tags, pred_tags)

        assert metrics['accuracy'] == 1.0

    def test_token_level_partial(self):
        """Test token-level metrics with partial match."""
        true_tags = [['B-Chemical', 'O', 'O', 'O']]
        pred_tags = [['O', 'O', 'O', 'O']]

        metrics = compute_token_level_metrics(true_tags, pred_tags)

        # 3/4 correct
        assert metrics['accuracy'] == 0.75


class TestMetricsEdgeCases:
    """Test edge cases in metrics computation."""

    def test_single_token_entity(self):
        """Test with single-token entities."""
        true_tags = [['B-Chemical']]
        pred_tags = [['B-Chemical']]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['f1'] == 1.0

    def test_long_entity(self):
        """Test with long entity spans."""
        true_tags = [['B-Chemical'] + ['I-Chemical'] * 10]
        pred_tags = [['B-Chemical'] + ['I-Chemical'] * 10]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['f1'] == 1.0

    def test_many_entities(self):
        """Test with many entities in one sequence."""
        # 5 single-token entities
        true_tags = [['B-Chemical', 'O', 'B-Disease', 'O', 'B-Chemical',
                      'O', 'B-Disease', 'O', 'B-Chemical', 'O']]
        pred_tags = [['B-Chemical', 'O', 'B-Disease', 'O', 'B-Chemical',
                      'O', 'B-Disease', 'O', 'B-Chemical', 'O']]

        metrics = compute_metrics(true_tags, pred_tags)

        assert metrics['f1'] == 1.0

    def test_only_outside_tags(self):
        """Test with only O tags."""
        true_tags = [['O', 'O', 'O']]
        pred_tags = [['O', 'O', 'O']]

        metrics = compute_metrics(true_tags, pred_tags)

        # No entities to evaluate - F1 should be 0
        assert metrics['f1'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
