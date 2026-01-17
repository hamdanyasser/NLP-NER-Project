"""
Unit tests for CRF layer.

Tests:
- Forward pass (log-likelihood computation)
- Viterbi decoding (correct backtracking)
- Transition constraint handling
- Batch processing
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.layers import CRF


class TestCRF:
    """Test suite for CRF layer."""

    @pytest.fixture
    def crf(self):
        """Create a CRF instance for testing."""
        return CRF(num_tags=5, pad_idx=0)

    @pytest.fixture
    def sample_emissions(self):
        """Create sample emission scores."""
        batch_size = 2
        seq_len = 4
        num_tags = 5
        return torch.randn(batch_size, seq_len, num_tags)

    @pytest.fixture
    def sample_tags(self):
        """Create sample tag sequences."""
        return torch.tensor([
            [1, 2, 3, 1],
            [2, 1, 1, 0]  # Last token is padding
        ])

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask."""
        return torch.tensor([
            [True, True, True, True],
            [True, True, True, False]
        ])

    def test_crf_initialization(self, crf):
        """Test CRF initializes correctly."""
        assert crf.num_tags == 5
        assert crf.pad_idx == 0
        assert crf.transitions.shape == (5, 5)
        assert crf.start_transitions.shape == (5,)
        assert crf.end_transitions.shape == (5,)

    def test_crf_forward_returns_scalar(self, crf, sample_emissions, sample_tags, sample_mask):
        """Test forward pass returns a scalar loss."""
        loss = crf(sample_emissions, sample_tags, sample_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # NLL should be non-negative

    def test_crf_forward_requires_grad(self, crf, sample_emissions, sample_tags, sample_mask):
        """Test that loss has gradient."""
        sample_emissions.requires_grad = True
        loss = crf(sample_emissions, sample_tags, sample_mask)

        loss.backward()
        assert sample_emissions.grad is not None

    def test_crf_decode_returns_list(self, crf, sample_emissions, sample_mask):
        """Test decode returns list of tag sequences."""
        predictions = crf.decode(sample_emissions, sample_mask)

        assert isinstance(predictions, list)
        assert len(predictions) == sample_emissions.shape[0]

        # Check lengths match mask
        for i, pred in enumerate(predictions):
            expected_len = sample_mask[i].sum().item()
            assert len(pred) == expected_len

    def test_crf_decode_returns_valid_tags(self, crf, sample_emissions, sample_mask):
        """Test decode returns valid tag indices."""
        predictions = crf.decode(sample_emissions, sample_mask)

        for pred in predictions:
            for tag in pred:
                assert 0 <= tag < crf.num_tags

    def test_crf_consistency(self, crf, sample_emissions, sample_mask):
        """Test that decoding is consistent (deterministic)."""
        pred1 = crf.decode(sample_emissions, sample_mask)
        pred2 = crf.decode(sample_emissions, sample_mask)

        for p1, p2 in zip(pred1, pred2):
            assert p1 == p2

    def test_crf_batch_size_one(self, crf):
        """Test CRF works with batch size 1."""
        emissions = torch.randn(1, 5, 5)
        tags = torch.tensor([[1, 2, 3, 1, 2]])
        mask = torch.ones(1, 5, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        predictions = crf.decode(emissions, mask)

        assert loss.dim() == 0
        assert len(predictions) == 1
        assert len(predictions[0]) == 5

    def test_crf_single_token(self, crf):
        """Test CRF works with single token sequences."""
        emissions = torch.randn(2, 1, 5)
        tags = torch.tensor([[1], [2]])
        mask = torch.ones(2, 1, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        predictions = crf.decode(emissions, mask)

        assert loss.dim() == 0
        assert len(predictions) == 2
        assert len(predictions[0]) == 1

    def test_crf_with_padding(self, crf):
        """Test CRF handles padding correctly."""
        emissions = torch.randn(2, 4, 5)
        tags = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]])
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False]
        ])

        predictions = crf.decode(emissions, mask)

        # Check that predictions respect sequence lengths
        assert len(predictions[0]) == 2  # First sequence has 2 tokens
        assert len(predictions[1]) == 3  # Second sequence has 3 tokens

    def test_crf_long_sequence(self, crf):
        """Test CRF with longer sequences."""
        seq_len = 100
        emissions = torch.randn(1, seq_len, 5)
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        tags = torch.randint(1, 5, (1, seq_len))

        loss = crf(emissions, tags, mask)
        predictions = crf.decode(emissions, mask)

        assert loss.dim() == 0
        assert len(predictions[0]) == seq_len

    def test_crf_optimal_path(self):
        """Test that Viterbi finds optimal path for simple case."""
        crf = CRF(num_tags=3, pad_idx=0)

        # Zero out transitions to make path predictable
        crf.transitions.data.zero_()
        crf.start_transitions.data.zero_()
        crf.end_transitions.data.zero_()

        # Create emissions that strongly favor specific tags
        emissions = torch.zeros(1, 3, 3)
        emissions[0, 0, 1] = 10.0  # Token 0 -> Tag 1
        emissions[0, 1, 2] = 10.0  # Token 1 -> Tag 2
        emissions[0, 2, 1] = 10.0  # Token 2 -> Tag 1

        mask = torch.ones(1, 3, dtype=torch.bool)
        predictions = crf.decode(emissions, mask)

        assert predictions[0] == [1, 2, 1]


class TestCRFGradients:
    """Test CRF gradient computation."""

    def test_gradient_flow(self):
        """Test gradients flow through CRF."""
        crf = CRF(num_tags=5, pad_idx=0)
        emissions = torch.randn(2, 4, 5, requires_grad=True)
        tags = torch.randint(1, 5, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        loss.backward()

        # Check gradients exist
        assert emissions.grad is not None
        assert crf.transitions.grad is not None
        assert crf.start_transitions.grad is not None
        assert crf.end_transitions.grad is not None

    def test_gradient_non_zero(self):
        """Test gradients are non-zero."""
        crf = CRF(num_tags=5, pad_idx=0)
        emissions = torch.randn(2, 4, 5, requires_grad=True)
        tags = torch.randint(1, 5, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        loss.backward()

        # At least some gradients should be non-zero
        assert emissions.grad.abs().sum() > 0


class TestCRFNumericalStability:
    """Test CRF numerical stability."""

    def test_large_emissions(self):
        """Test CRF handles large emission values."""
        crf = CRF(num_tags=5, pad_idx=0)
        emissions = torch.randn(2, 4, 5) * 100  # Large values
        tags = torch.randint(1, 5, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)

        loss = crf(emissions, tags, mask)

        assert torch.isfinite(loss)

    def test_small_emissions(self):
        """Test CRF handles small emission values."""
        crf = CRF(num_tags=5, pad_idx=0)
        emissions = torch.randn(2, 4, 5) * 0.001  # Small values
        tags = torch.randint(1, 5, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)

        loss = crf(emissions, tags, mask)

        assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
