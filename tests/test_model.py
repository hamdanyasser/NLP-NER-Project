"""
Unit tests for BiLSTM-CRF model.

Tests:
- Model initialization with various configurations
- Forward pass shapes
- Loss computation
- Prediction output
- Character feature integration
- Attention integration
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bilstm_crf import BiLSTMCRF, BaselineBiLSTM


class TestBiLSTMCRF:
    """Test suite for BiLSTM-CRF model."""

    @pytest.fixture
    def basic_model(self):
        """Create basic model without optional features."""
        return BiLSTMCRF(
            vocab_size=1000,
            num_tags=5,
            embedding_dim=100,
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            pad_idx=0,
            use_char_features=False,
            use_attention=False
        )

    @pytest.fixture
    def full_model(self):
        """Create model with all features."""
        return BiLSTMCRF(
            vocab_size=1000,
            num_tags=5,
            embedding_dim=100,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            pad_idx=0,
            use_char_features=True,
            num_chars=100,
            char_embedding_dim=30,
            char_hidden_size=50,
            char_kernel_sizes=[2, 3, 4],
            use_highway=True,
            use_attention=True,
            attention_heads=4
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        batch_size = 4
        seq_len = 10
        max_word_len = 15

        return {
            'token_ids': torch.randint(1, 1000, (batch_size, seq_len)),
            'label_ids': torch.randint(1, 5, (batch_size, seq_len)),
            'mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'char_ids': torch.randint(1, 100, (batch_size, seq_len, max_word_len))
        }

    def test_basic_model_init(self, basic_model):
        """Test basic model initialization."""
        assert basic_model.vocab_size == 1000
        assert basic_model.num_tags == 5
        assert basic_model.use_char_features == False
        assert basic_model.use_attention == False

    def test_full_model_init(self, full_model):
        """Test full model initialization."""
        assert full_model.use_char_features == True
        assert full_model.use_attention == True
        assert full_model.char_cnn is not None
        assert full_model.attention is not None

    def test_forward_shape(self, basic_model, sample_batch):
        """Test forward pass output shape."""
        output = basic_model(
            sample_batch['token_ids'],
            sample_batch['mask']
        )

        batch_size, seq_len = sample_batch['token_ids'].shape
        assert output.shape == (batch_size, seq_len, basic_model.num_tags)

    def test_loss_computation(self, basic_model, sample_batch):
        """Test loss computation returns scalar."""
        loss = basic_model.loss(
            sample_batch['token_ids'],
            sample_batch['label_ids'],
            sample_batch['mask']
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_predict_output(self, basic_model, sample_batch):
        """Test predict returns list of tag sequences."""
        predictions = basic_model.predict(
            sample_batch['token_ids'],
            sample_batch['mask']
        )

        assert isinstance(predictions, list)
        assert len(predictions) == sample_batch['token_ids'].shape[0]

        # Check each prediction has correct length
        for i, pred in enumerate(predictions):
            expected_len = sample_batch['mask'][i].sum().item()
            assert len(pred) == expected_len

    def test_gradient_flow(self, basic_model, sample_batch):
        """Test gradients flow through model."""
        loss = basic_model.loss(
            sample_batch['token_ids'],
            sample_batch['label_ids'],
            sample_batch['mask']
        )

        loss.backward()

        # Check gradients exist
        for name, param in basic_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_model_with_char_features(self, full_model, sample_batch):
        """Test model with character features."""
        output = full_model(
            sample_batch['token_ids'],
            sample_batch['mask'],
            sample_batch['char_ids']
        )

        batch_size, seq_len = sample_batch['token_ids'].shape
        assert output.shape == (batch_size, seq_len, full_model.num_tags)

    def test_model_with_attention(self, full_model, sample_batch):
        """Test model with attention."""
        _ = full_model(
            sample_batch['token_ids'],
            sample_batch['mask'],
            sample_batch['char_ids']
        )

        # Check attention weights are stored
        attn_weights = full_model.get_attention_weights()
        assert attn_weights is not None

    def test_pretrained_embeddings(self):
        """Test model with pretrained embeddings."""
        vocab_size = 100
        embedding_dim = 50
        pretrained = torch.randn(vocab_size, embedding_dim)

        model = BiLSTMCRF(
            vocab_size=vocab_size,
            num_tags=5,
            embedding_dim=embedding_dim,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            pad_idx=0,
            pretrained_embeddings=pretrained,
            freeze_embeddings=True
        )

        # Check embeddings are frozen
        assert not model.embedding.weight.requires_grad

        # Check embeddings are copied
        assert torch.allclose(model.embedding.weight.data, pretrained)

    def test_variable_sequence_lengths(self, basic_model):
        """Test model handles variable sequence lengths."""
        batch_size = 3
        max_len = 8

        # Create batch with different lengths
        token_ids = torch.randint(1, 1000, (batch_size, max_len))
        mask = torch.tensor([
            [True, True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, True, True, True, True, True, True]
        ])
        label_ids = torch.randint(1, 5, (batch_size, max_len))

        # Should not raise error
        loss = basic_model.loss(token_ids, label_ids, mask)
        predictions = basic_model.predict(token_ids, mask)

        assert loss.dim() == 0
        assert len(predictions[0]) == 5
        assert len(predictions[1]) == 3
        assert len(predictions[2]) == 8

    def test_model_summary(self, full_model):
        """Test get_model_summary method."""
        summary = full_model.get_model_summary()

        assert 'vocab_size' in summary
        assert 'num_tags' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert summary['vocab_size'] == 1000
        assert summary['num_tags'] == 5


class TestBaselineBiLSTM:
    """Test suite for baseline BiLSTM (no CRF)."""

    @pytest.fixture
    def baseline_model(self):
        """Create baseline model."""
        return BaselineBiLSTM(
            vocab_size=1000,
            num_tags=5,
            embedding_dim=100,
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            pad_idx=0
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        return {
            'token_ids': torch.randint(1, 1000, (4, 10)),
            'label_ids': torch.randint(1, 5, (4, 10)),
            'mask': torch.ones(4, 10, dtype=torch.bool)
        }

    def test_forward_shape(self, baseline_model, sample_batch):
        """Test forward pass output shape."""
        output = baseline_model(
            sample_batch['token_ids'],
            sample_batch['mask']
        )

        assert output.shape == (4, 10, 5)

    def test_loss_computation(self, baseline_model, sample_batch):
        """Test loss computation."""
        loss = baseline_model.loss(
            sample_batch['token_ids'],
            sample_batch['label_ids'],
            sample_batch['mask']
        )

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_predict_uses_argmax(self, baseline_model, sample_batch):
        """Test that predict uses argmax (not Viterbi)."""
        predictions = baseline_model.predict(
            sample_batch['token_ids'],
            sample_batch['mask']
        )

        # Get logits
        logits = baseline_model(sample_batch['token_ids'], sample_batch['mask'])
        argmax_preds = logits.argmax(dim=-1)

        # Check predictions match argmax
        for i, pred in enumerate(predictions):
            length = sample_batch['mask'][i].sum().item()
            assert pred == argmax_preds[i, :length].tolist()


class TestModelDevices:
    """Test model device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_cuda(self):
        """Test model can be moved to CUDA."""
        model = BiLSTMCRF(
            vocab_size=100,
            num_tags=5,
            embedding_dim=50,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            pad_idx=0
        ).cuda()

        token_ids = torch.randint(1, 100, (2, 5)).cuda()
        mask = torch.ones(2, 5, dtype=torch.bool).cuda()

        output = model(token_ids, mask)
        assert output.device.type == 'cuda'


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_sequence(self):
        """Test model with single token sequences."""
        model = BiLSTMCRF(
            vocab_size=100, num_tags=5, embedding_dim=50,
            hidden_size=64, num_layers=1, dropout=0.0, pad_idx=0
        )

        token_ids = torch.tensor([[5], [10]])
        mask = torch.ones(2, 1, dtype=torch.bool)
        labels = torch.tensor([[1], [2]])

        loss = model.loss(token_ids, labels, mask)
        predictions = model.predict(token_ids, mask)

        assert loss.dim() == 0
        assert len(predictions[0]) == 1

    def test_large_batch(self):
        """Test model with large batch."""
        model = BiLSTMCRF(
            vocab_size=100, num_tags=5, embedding_dim=50,
            hidden_size=64, num_layers=1, dropout=0.0, pad_idx=0
        )

        batch_size = 64
        seq_len = 50
        token_ids = torch.randint(1, 100, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        predictions = model.predict(token_ids, mask)
        assert len(predictions) == batch_size

    def test_model_eval_mode(self):
        """Test model behavior in eval mode."""
        model = BiLSTMCRF(
            vocab_size=100, num_tags=5, embedding_dim=50,
            hidden_size=64, num_layers=1, dropout=0.5, pad_idx=0
        )

        token_ids = torch.randint(1, 100, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            pred1 = model.predict(token_ids, mask)
            pred2 = model.predict(token_ids, mask)

        # Predictions should be deterministic in eval mode
        assert pred1 == pred2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
