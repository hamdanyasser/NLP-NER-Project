"""
Attention mechanisms for NER models.

This module provides:
- Self-attention layer for sequence modeling
- Multi-head attention implementation
- Attention weight visualization utilities

Built from scratch using PyTorch primitives.

Author: NLP Course Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Computes attention weights using the formula:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        dropout: Dropout probability for attention weights
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor (batch, seq_len, d_k)
            key: Key tensor (batch, seq_len, d_k)
            value: Value tensor (batch, seq_len, d_v)
            mask: Optional mask (batch, seq_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)

        # Compute attention scores: (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention: (batch, 1, seq_len) -> broadcast to (batch, seq_len, seq_len)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Handle case where all positions are masked
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )

        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        d_model: Model dimension (input/output size)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask (batch, seq_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Expand mask for multi-head
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)

        # Apply attention
        # Output shape: (batch, num_heads, seq_len, d_k)
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # Concatenate heads and apply output projection
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.w_o(attn_output)

        # Average attention weights across heads for visualization
        attn_weights = attn_weights.mean(dim=1)

        return output, attn_weights


class SelfAttention(nn.Module):
    """
    Self-Attention layer for sequence modeling.

    Combines multi-head self-attention with residual connection
    and layer normalization.

    Args:
        hidden_size: Input/output dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention
        self.multihead_attn = MultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout for residual connection
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply self-attention with residual connection.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            mask: Optional mask (batch, seq_len), True for valid positions

        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        # Self-attention (query, key, value are all the same)
        attn_output, self.attention_weights = self.multihead_attn(x, x, x, mask)

        # Residual connection and dropout
        output = x + self.dropout(attn_output)

        # Layer normalization
        if self.use_layer_norm:
            output = self.layer_norm(output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the last computed attention weights.

        Returns:
            Attention weights tensor (batch, seq_len, seq_len) or None
        """
        return self.attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for adding position information to embeddings.

    Uses sinusoidal positional encoding as in "Attention Is All You Need".

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating sequence representations.

    Learns to weight different positions based on their importance.

    Args:
        hidden_size: Input dimension
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence using learned attention weights.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            mask: Optional mask (batch, seq_len)

        Returns:
            Pooled representation (batch, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)

        # Weighted sum
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return output
