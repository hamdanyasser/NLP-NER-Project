"""
BiLSTM-CRF model for Named Entity Recognition.

This is the main model combining:
- Word embeddings (random or pretrained GloVe)
- Character-level CNN with highway networks
- Bidirectional LSTM encoder
- Self-attention mechanism
- CRF layer for sequence labeling

Built from scratch for NLP Course Project.

Author: NLP Course Project
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.models.layers import CRF, CharCNN, EnhancedCharCNN, Highway
from src.models.attention import SelfAttention


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for NER with optional enhancements.

    Supports:
    - Pre-trained word embeddings (GloVe)
    - Character-level features (CNN with highway networks)
    - Self-attention after BiLSTM
    - Dropout throughout the network

    Args:
        vocab_size: Size of word vocabulary
        num_tags: Number of tags (including PAD)
        embedding_dim: Dimension of word embeddings
        hidden_size: Hidden size of LSTM (per direction)
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        pad_idx: Padding index
        pretrained_embeddings: Optional pretrained embedding tensor
        freeze_embeddings: Whether to freeze pretrained embeddings
        use_char_features: Whether to use character-level features
        num_chars: Number of unique characters
        char_embedding_dim: Dimension of character embeddings
        char_hidden_size: Size of character features
        char_kernel_sizes: Kernel sizes for char CNN
        use_highway: Whether to use highway networks in char CNN
        use_attention: Whether to use self-attention
        attention_heads: Number of attention heads
        attention_dropout: Dropout for attention
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        use_char_features: bool = False,
        num_chars: int = 0,
        char_embedding_dim: int = 30,
        char_hidden_size: int = 50,
        char_kernel_sizes: List[int] = [2, 3, 4],
        use_highway: bool = True,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.1
    ):
        super(BiLSTMCRF, self).__init__()

        # Store configuration
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx
        self.use_char_features = use_char_features
        self.use_attention = use_attention

        # ========== Word Embedding Layer ==========
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                print("Word embeddings frozen (not trainable)")

        # ========== Character-level Features (Optional) ==========
        if use_char_features:
            assert num_chars > 0, "num_chars must be provided if use_char_features=True"

            # Use enhanced CharCNN with multiple kernels and highway
            self.char_cnn = EnhancedCharCNN(
                num_chars=num_chars,
                char_embedding_dim=char_embedding_dim,
                char_hidden_size=char_hidden_size,
                kernel_sizes=char_kernel_sizes,
                padding_idx=pad_idx,
                use_highway=use_highway,
                num_highway_layers=2,
                dropout=dropout,
                use_batch_norm=True
            )
            lstm_input_size = embedding_dim + char_hidden_size
        else:
            self.char_cnn = None
            lstm_input_size = embedding_dim

        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)

        # ========== BiLSTM Encoder ==========
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # BiLSTM output dimension (bidirectional = hidden_size * 2)
        lstm_output_dim = hidden_size * 2

        # ========== Self-Attention (Optional) ==========
        if use_attention:
            self.attention = SelfAttention(
                hidden_size=lstm_output_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                use_layer_norm=True
            )
        else:
            self.attention = None

        # ========== Projection to Tag Space ==========
        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)

        # ========== CRF Layer ==========
        self.crf = CRF(num_tags=num_tags, pad_idx=pad_idx)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize linear layer
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    def _get_lstm_features(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get BiLSTM features (emissions for CRF).

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            Emission scores, shape (batch_size, seq_len, num_tags)
        """
        # ========== Word Embeddings ==========
        # Shape: (batch_size, seq_len, embedding_dim)
        word_embeds = self.embedding(token_ids)

        # ========== Character Features ==========
        if self.use_char_features and char_ids is not None:
            # Shape: (batch_size, seq_len, char_hidden_size)
            char_features = self.char_cnn(char_ids)
            # Concatenate word and char features
            embeds = torch.cat([word_embeds, char_features], dim=-1)
        else:
            embeds = word_embeds

        embeds = self.dropout(embeds)

        # ========== BiLSTM ==========
        # Pack sequence for efficient processing of variable-length sequences
        lengths = mask.sum(dim=1).cpu()

        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_embeds)

        # Unpack: (batch_size, seq_len, hidden_size * 2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        lstm_output = self.dropout(lstm_output)

        # ========== Self-Attention (Optional) ==========
        if self.use_attention and self.attention is not None:
            lstm_output = self.attention(lstm_output, mask)

        # ========== Project to Tag Space ==========
        # Shape: (batch_size, seq_len, num_tags)
        emissions = self.hidden2tag(lstm_output)

        return emissions

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (returns emissions).

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            Emission scores, shape (batch_size, seq_len, num_tags)
        """
        return self._get_lstm_features(token_ids, mask, char_ids)

    def loss(
        self,
        token_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            Loss (scalar)
        """
        emissions = self._get_lstm_features(token_ids, mask, char_ids)
        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Predict tag sequences using Viterbi decoding.

        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            List of predicted tag sequences
        """
        emissions = self._get_lstm_features(token_ids, mask, char_ids)
        pred_tags = self.crf.decode(emissions, mask)
        return pred_tags

    def get_embeddings(self) -> torch.Tensor:
        """Get word embedding weights."""
        return self.embedding.weight.data

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last attention weights (if using attention)."""
        if self.attention is not None:
            return self.attention.get_attention_weights()
        return None

    def get_model_summary(self) -> Dict:
        """Get model configuration summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'vocab_size': self.vocab_size,
            'num_tags': self.num_tags,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'use_char_features': self.use_char_features,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class BaselineBiLSTM(nn.Module):
    """
    Baseline BiLSTM model (no CRF) for comparison.

    Uses softmax + cross-entropy loss instead of CRF.
    Useful for ablation studies to show CRF benefit.

    Args:
        vocab_size: Size of word vocabulary
        num_tags: Number of tags
        embedding_dim: Dimension of word embeddings
        hidden_size: Hidden size of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        pad_idx: Padding index
        pretrained_embeddings: Optional pretrained embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 100,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(BaselineBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        # Word embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

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

        # Projection
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)

        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        embeds = self.embedding(token_ids)
        embeds = self.dropout(embeds)

        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)

        logits = self.hidden2tag(output)
        return logits

    def loss(
        self,
        token_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        logits = self.forward(token_ids, mask)

        # Reshape for loss computation
        batch_size, seq_len, num_tags = logits.shape
        logits = logits.view(-1, num_tags)
        tags = tags.view(-1)

        loss = self.criterion(logits, tags)
        return loss

    def predict(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Predict using argmax (no Viterbi)."""
        logits = self.forward(token_ids, mask)
        predictions = logits.argmax(dim=-1)

        # Convert to list and respect actual lengths
        pred_list = []
        for pred, m in zip(predictions, mask):
            length = m.sum().item()
            pred_list.append(pred[:length].tolist())

        return pred_list
