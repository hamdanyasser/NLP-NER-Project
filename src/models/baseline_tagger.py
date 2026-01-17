"""
Baseline BiLSTM tagger for NER (without CRF).

This model serves as a baseline for comparison with BiLSTM-CRF.
It uses:
- Word embeddings
- Bidirectional LSTM
- Linear layer to tag space
- Softmax + cross-entropy loss (no CRF)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class BaselineBiLSTMTagger(nn.Module):
    """
    Baseline BiLSTM tagger without CRF.

    Args:
        vocab_size: Size of word vocabulary
        num_tags: Number of tags
        embedding_dim: Dimension of word embeddings
        hidden_size: Hidden size of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        pad_idx: Padding index for embeddings
        pretrained_embeddings: Optional pretrained embeddings
    """

    def __init__(self,
                 vocab_size: int,
                 num_tags: int,
                 embedding_dim: int = 100,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 pad_idx: int = 0,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(BaselineBiLSTMTagger, self).__init__()

        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Linear layer to tag space
        # *2 because bidirectional
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)

        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)

        Returns:
            Tag logits, shape (batch_size, seq_len, num_tags)
        """
        # Get embeddings
        # Shape: (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(token_ids)
        embeds = self.dropout(embeds)

        # Pack sequence for LSTM (to handle variable lengths efficiently)
        lengths = mask.sum(dim=1).cpu()
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward
        packed_output, _ = self.lstm(packed_embeds)

        # Unpack
        # Shape: (batch_size, seq_len, hidden_size * 2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        lstm_output = self.dropout(lstm_output)

        # Project to tag space
        # Shape: (batch_size, seq_len, num_tags)
        tag_logits = self.hidden2tag(lstm_output)

        return tag_logits

    def loss(self, token_ids: torch.Tensor, tags: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)

        Returns:
            Loss (scalar)
        """
        # Get logits
        logits = self.forward(token_ids, mask)

        # Reshape for cross-entropy
        # logits: (batch_size * seq_len, num_tags)
        # tags: (batch_size * seq_len)
        logits_flat = logits.view(-1, self.num_tags)
        tags_flat = tags.view(-1)

        # Compute loss (cross-entropy automatically ignores pad_idx)
        loss = self.criterion(logits_flat, tags_flat)

        return loss

    def predict(self, token_ids: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        Predict tag sequences.

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)

        Returns:
            List of predicted tag sequences
        """
        # Get logits
        logits = self.forward(token_ids, mask)

        # Get predictions (argmax)
        # Shape: (batch_size, seq_len)
        predictions = logits.argmax(dim=-1)

        # Convert to list, removing padding
        pred_tags = []
        for pred, m in zip(predictions, mask):
            # Get length
            length = m.sum().item()
            # Get tags up to length
            pred_tags.append(pred[:length].tolist())

        return pred_tags

    def get_embeddings(self) -> torch.Tensor:
        """
        Get embedding weights.

        Returns:
            Embedding weight matrix
        """
        return self.embedding.weight.data
