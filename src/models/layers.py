"""
Neural network layers for NER models.

This module provides:
- CRF (Conditional Random Field) layer
- Character-level CNN (optional enhancement)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.

    This implements:
    - Forward algorithm for computing log partition function
    - Viterbi algorithm for finding best tag sequence
    - Negative log-likelihood loss computation

    Args:
        num_tags: Number of tags (including PAD tag)
        pad_idx: Index of PAD tag (transitions to/from PAD will be impossible)
    """

    def __init__(self, num_tags: int, pad_idx: int = 0):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx

        # Transition parameters: transitions[i, j] = score of transitioning from tag i to tag j
        # Shape: (num_tags, num_tags)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Initialize transitions
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize transition parameters."""
        # Start and end transitions
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Impossible transitions to/from PAD
        self.transitions.data[self.pad_idx, :] = -10000.0
        self.transitions.data[:, self.pad_idx] = -10000.0

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            emissions: Emission scores from BiLSTM, shape (batch_size, seq_len, num_tags)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask for valid positions, shape (batch_size, seq_len)

        Returns:
            Negative log-likelihood loss (scalar)
        """
        # Compute log partition function (normalizer)
        log_partition = self._compute_log_partition(emissions, mask)

        # Compute score of true sequence
        gold_score = self._compute_score(emissions, tags, mask)

        # NLL = log_partition - gold_score
        # Average over batch
        nll = (log_partition - gold_score).mean()

        return nll

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Compute score of a given tag sequence.

        Args:
            emissions: Emission scores, shape (batch_size, seq_len, num_tags)
            tags: Tag sequence, shape (batch_size, seq_len)
            mask: Mask, shape (batch_size, seq_len)

        Returns:
            Sequence scores, shape (batch_size,)
        """
        batch_size, seq_len = tags.shape

        # Emission scores for the given tags
        # Shape: (batch_size, seq_len)
        emission_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)

        # Mask out padding positions
        emission_scores = emission_scores * mask.float()

        # Transition scores
        # For each position, get transition from previous tag to current tag
        # Shape: (batch_size, seq_len - 1)
        transition_scores = torch.zeros(batch_size, device=emissions.device)

        for i in range(1, seq_len):
            # Get transition score from tags[:, i-1] to tags[:, i]
            # Only add if both positions are valid (not padding)
            valid = mask[:, i] & mask[:, i - 1]

            if valid.any():
                # Get previous and current tags
                prev_tags = tags[:, i - 1]
                curr_tags = tags[:, i]

                # Get transition scores
                # transitions[prev_tag, curr_tag]
                trans = self.transitions[prev_tags, curr_tags]

                # Add transition scores where valid
                transition_scores = transition_scores + trans * valid.float()

        # Total score = sum of emission scores + sum of transition scores
        total_score = emission_scores.sum(dim=1) + transition_scores

        return total_score

    def _compute_log_partition(self, emissions: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm.

        Args:
            emissions: Emission scores, shape (batch_size, seq_len, num_tags)
            mask: Mask, shape (batch_size, seq_len)

        Returns:
            Log partition function, shape (batch_size,)
        """
        batch_size, seq_len, num_tags = emissions.shape

        # Initialize forward variables
        # alpha[t] = log sum of scores of all paths ending at time t
        # Start with first position
        alpha = emissions[:, 0, :]  # Shape: (batch_size, num_tags)

        # Iterate through sequence
        for i in range(1, seq_len):
            # Get emission scores for current position
            emit_scores = emissions[:, i, :].unsqueeze(1)  # Shape: (batch_size, 1, num_tags)

            # Get transition scores: from all previous tags to all current tags
            # alpha[:, :, None] has shape (batch_size, num_tags, 1)
            # transitions[None, :, :] has shape (1, num_tags, num_tags)
            # Result shape: (batch_size, num_tags, num_tags)
            trans_scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)

            # Add emission scores
            # Shape: (batch_size, num_tags, num_tags)
            next_alpha = trans_scores + emit_scores

            # Log-sum-exp over previous tags (dim=1)
            # Shape: (batch_size, num_tags)
            next_alpha = torch.logsumexp(next_alpha, dim=1)

            # Update alpha, but only for valid positions
            # Where mask is 0, keep previous alpha
            alpha = torch.where(mask[:, i].unsqueeze(1), next_alpha, alpha)

        # Sum over all possible ending tags
        log_partition = torch.logsumexp(alpha, dim=1)

        return log_partition

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        Find the best tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores, shape (batch_size, seq_len, num_tags)
            mask: Mask, shape (batch_size, seq_len)

        Returns:
            List of best tag sequences (one per batch item)
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Viterbi algorithm
        # viterbi[tag] = max score of sequences ending in that tag
        # backpointers[t][tag] = best previous tag for sequences ending in tag at time t

        # Initialize with first position emission scores
        viterbi = emissions[:, 0, :].clone()  # Shape: (batch_size, num_tags)

        # Store backpointers for each time step
        # backpointers[t] has shape (batch_size, num_tags) and stores the best previous tag
        backpointers = []

        # Forward pass: compute viterbi scores and backpointers
        for t in range(1, seq_len):
            # Expand dimensions for broadcasting
            # prev_viterbi: (batch_size, num_tags, 1) - scores from previous tags
            # transitions: (1, num_tags, num_tags) - transition[i,j] = score from i to j
            # emit: (batch_size, 1, num_tags) - emission scores at current position
            prev_viterbi = viterbi.unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            emit = emissions[:, t, :].unsqueeze(1)

            # Compute score for each (prev_tag, curr_tag) combination
            # Shape: (batch_size, num_tags, num_tags)
            # scores[b, i, j] = score of being at tag i at t-1 and tag j at t
            scores = prev_viterbi + trans + emit

            # Find best previous tag for each current tag
            # next_viterbi: (batch_size, num_tags) - best score ending at each tag
            # best_prev_tags: (batch_size, num_tags) - which prev tag gave best score
            next_viterbi, best_prev_tags = scores.max(dim=1)

            # Only update viterbi scores for valid (non-padded) positions
            mask_t = mask[:, t].unsqueeze(1)  # (batch_size, 1)
            viterbi = torch.where(mask_t, next_viterbi, viterbi)

            # Store backpointers
            backpointers.append(best_prev_tags)

        # Backtrack to find best paths for each sample in batch
        best_paths = []

        for b in range(batch_size):
            # Get actual sequence length for this sample
            seq_length = int(mask[b].sum().item())

            if seq_length == 0:
                best_paths.append([])
                continue

            if seq_length == 1:
                # Only one position, just take the best tag
                _, best_tag = viterbi[b].max(dim=0)
                best_paths.append([best_tag.item()])
                continue

            # Find best tag at the last valid position
            _, best_tag = viterbi[b].max(dim=0)

            # Build path by backtracking
            # We need to go through backpointers[seq_length-2] down to backpointers[0]
            # backpointers[t] tells us: given the tag at position t+1, what's the best tag at position t
            path = [best_tag.item()]

            for t in range(seq_length - 2, -1, -1):
                # backpointers[t] gives the best previous tag at position t
                # given the current tag at position t+1
                best_tag = backpointers[t][b, best_tag]
                path.append(best_tag.item())

            # Reverse to get path from start to end
            path.reverse()
            best_paths.append(path)

        return best_paths


class CharCNN(nn.Module):
    """
    Character-level CNN for learning character-based word representations.

    Args:
        num_chars: Number of unique characters
        char_embedding_dim: Dimension of character embeddings
        char_hidden_size: Size of character-level features
        kernel_size: Kernel size for CNN
        padding_idx: Index for padding character
    """

    def __init__(self,
                 num_chars: int,
                 char_embedding_dim: int = 30,
                 char_hidden_size: int = 50,
                 kernel_size: int = 3,
                 padding_idx: int = 0):
        super(CharCNN, self).__init__()

        self.char_embedding = nn.Embedding(
            num_chars,
            char_embedding_dim,
            padding_idx=padding_idx
        )

        self.conv = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            Character-level features, shape (batch_size, seq_len, char_hidden_size)
        """
        batch_size, seq_len, max_word_len = char_ids.shape

        # Reshape for character embedding
        # (batch_size * seq_len, max_word_len)
        char_ids_flat = char_ids.view(-1, max_word_len)

        # Character embeddings
        # (batch_size * seq_len, max_word_len, char_embedding_dim)
        char_embeds = self.char_embedding(char_ids_flat)

        # Transpose for conv1d: (batch_size * seq_len, char_embedding_dim, max_word_len)
        char_embeds = char_embeds.transpose(1, 2)

        # Apply convolution
        # (batch_size * seq_len, char_hidden_size, max_word_len)
        char_features = self.conv(char_embeds)

        # Max pooling over character dimension
        # (batch_size * seq_len, char_hidden_size)
        char_features = torch.max(char_features, dim=2)[0]

        # Reshape back
        # (batch_size, seq_len, char_hidden_size)
        char_features = char_features.view(batch_size, seq_len, -1)

        char_features = self.dropout(char_features)

        return char_features


class Highway(nn.Module):
    """
    Highway network layer.

    Implements the highway network transformation:
    y = g * H(x) + (1 - g) * x

    where g is the gate (transform gate) and H is a nonlinear transform.

    Reference: Srivastava et al., 2015 - "Highway Networks"

    Args:
        input_size: Input dimension
        num_layers: Number of highway layers
        activation: Activation function (default: ReLU)
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        # Transform layers (H)
        self.transforms = nn.ModuleList([
            nn.Linear(input_size, input_size)
            for _ in range(num_layers)
        ])

        # Gate layers (g)
        self.gates = nn.ModuleList([
            nn.Linear(input_size, input_size)
            for _ in range(num_layers)
        ])

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # Initialize gate bias to negative value for better gradient flow
        for gate in self.gates:
            nn.init.constant_(gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through highway layers.

        Args:
            x: Input tensor (..., input_size)

        Returns:
            Output tensor (..., input_size)
        """
        for transform, gate in zip(self.transforms, self.gates):
            # Transform
            h = self.activation(transform(x))

            # Gate (sigmoid for [0, 1] range)
            g = torch.sigmoid(gate(x))

            # Highway connection
            x = g * h + (1 - g) * x

        return x


class EnhancedCharCNN(nn.Module):
    """
    Enhanced Character-level CNN with multiple kernel sizes and highway networks.

    Features:
    - Multiple parallel convolutions with different kernel sizes
    - Highway network for better gradient flow
    - Batch normalization
    - Configurable dropout

    Based on Ma & Hovy (2016) architecture.

    Args:
        num_chars: Number of unique characters
        char_embedding_dim: Dimension of character embeddings
        char_hidden_size: Total size of character features
        kernel_sizes: List of kernel sizes for parallel convolutions
        padding_idx: Index for padding character
        use_highway: Whether to use highway network
        num_highway_layers: Number of highway layers
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        num_chars: int,
        char_embedding_dim: int = 30,
        char_hidden_size: int = 50,
        kernel_sizes: List[int] = [2, 3, 4],
        padding_idx: int = 0,
        use_highway: bool = True,
        num_highway_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super(EnhancedCharCNN, self).__init__()

        self.char_hidden_size = char_hidden_size
        self.use_highway = use_highway
        self.use_batch_norm = use_batch_norm

        # Character embedding
        self.char_embedding = nn.Embedding(
            num_chars,
            char_embedding_dim,
            padding_idx=padding_idx
        )

        # Multiple parallel convolutions with different kernel sizes
        # Each conv outputs char_hidden_size // len(kernel_sizes) channels
        self.num_filters_per_size = char_hidden_size // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=self.num_filters_per_size,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])

        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.num_filters_per_size * len(kernel_sizes))

        # Highway network
        total_conv_output = self.num_filters_per_size * len(kernel_sizes)
        if use_highway:
            self.highway = Highway(
                input_size=total_conv_output,
                num_layers=num_highway_layers,
                activation='relu'
            )

        # Optional projection if sizes don't match
        if total_conv_output != char_hidden_size:
            self.projection = nn.Linear(total_conv_output, char_hidden_size)
        else:
            self.projection = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.activation = nn.ReLU()

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            char_ids: Character IDs, shape (batch_size, seq_len, max_word_len)

        Returns:
            Character-level features, shape (batch_size, seq_len, char_hidden_size)
        """
        batch_size, seq_len, max_word_len = char_ids.shape

        # Reshape for processing: (batch_size * seq_len, max_word_len)
        char_ids_flat = char_ids.view(-1, max_word_len)

        # Character embeddings: (batch_size * seq_len, max_word_len, char_embedding_dim)
        char_embeds = self.char_embedding(char_ids_flat)

        # Transpose for conv1d: (batch_size * seq_len, char_embedding_dim, max_word_len)
        char_embeds = char_embeds.transpose(1, 2)

        # Apply parallel convolutions and max-pool
        conv_outputs = []
        for conv in self.convs:
            # Convolution: (batch_size * seq_len, num_filters_per_size, max_word_len)
            conv_out = self.activation(conv(char_embeds))

            # Max pooling over character dimension
            # (batch_size * seq_len, num_filters_per_size)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        # Concatenate outputs from all convolutions
        # (batch_size * seq_len, num_filters_per_size * num_kernel_sizes)
        char_features = torch.cat(conv_outputs, dim=1)

        # Batch normalization
        if self.use_batch_norm:
            char_features = self.batch_norm(char_features)

        # Highway network
        if self.use_highway:
            char_features = self.highway(char_features)

        # Projection if needed
        if self.projection is not None:
            char_features = self.projection(char_features)

        # Dropout
        char_features = self.dropout(char_features)

        # Reshape back: (batch_size, seq_len, char_hidden_size)
        char_features = char_features.view(batch_size, seq_len, -1)

        return char_features
