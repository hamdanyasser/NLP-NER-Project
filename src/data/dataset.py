"""
PyTorch Dataset and DataLoader for NER tasks.

This module provides:
- NERDataset class for loading BIO-formatted data
- Support for character-level features
- Collate function for batching variable-length sequences
- DataLoader creation utilities

Author: NLP Course Project
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.utils.vocab import Vocabulary, LabelVocabulary, CharVocabulary


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER data in BIO format.

    Supports both word-level and character-level features.

    Args:
        filepath: Path to BIO-formatted file
        word_vocab: Word vocabulary
        label_vocab: Label vocabulary
        char_vocab: Character vocabulary (optional, for char features)
        max_seq_length: Maximum sequence length (for truncation)
        max_word_length: Maximum word length for character features
        use_char_features: Whether to include character features
    """

    def __init__(self,
                 filepath: str,
                 word_vocab: Vocabulary,
                 label_vocab: LabelVocabulary,
                 char_vocab: Optional[CharVocabulary] = None,
                 max_seq_length: Optional[int] = None,
                 max_word_length: int = 20,
                 use_char_features: bool = False):
        self.filepath = filepath
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.use_char_features = use_char_features and char_vocab is not None

        # Load data
        self.sentences = []  # List of token lists
        self.labels = []     # List of label lists
        self._load_data()

    def _load_data(self) -> None:
        """Load sentences and labels from file."""
        current_tokens = []
        current_labels = []
        truncated_count = 0

        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Sentence boundary
                    if current_tokens:
                        # Truncate if needed
                        if self.max_seq_length and len(current_tokens) > self.max_seq_length:
                            current_tokens = current_tokens[:self.max_seq_length]
                            current_labels = current_labels[:self.max_seq_length]
                            truncated_count += 1

                        self.sentences.append(current_tokens)
                        self.labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        current_tokens.append(token)
                        current_labels.append(label)

            # Don't forget last sentence
            if current_tokens:
                if self.max_seq_length and len(current_tokens) > self.max_seq_length:
                    current_tokens = current_tokens[:self.max_seq_length]
                    current_labels = current_labels[:self.max_seq_length]
                    truncated_count += 1
                self.sentences.append(current_tokens)
                self.labels.append(current_labels)

        print(f"Loaded {len(self.sentences)} sentences from {self.filepath}")
        if truncated_count > 0:
            print(f"  (truncated {truncated_count} sentences to max length {self.max_seq_length})")

    def __len__(self) -> int:
        """Return number of sentences."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of sample

        Returns:
            Dictionary with 'token_ids', 'label_ids', 'length', and optionally 'char_ids'
        """
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        # Convert to indices
        token_ids = self.word_vocab.encode(tokens)
        label_ids = self.label_vocab.encode(labels)

        result = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'length': len(tokens),
            'tokens': tokens  # Keep original tokens for reference
        }

        # Add character features if enabled
        if self.use_char_features:
            char_ids = self.char_vocab.encode_sequence(tokens, self.max_word_length)
            result['char_ids'] = torch.tensor(char_ids, dtype=torch.long)

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]],
               pad_token_id: int,
               pad_label_id: int,
               pad_char_id: int = 0,
               use_char_features: bool = False,
               max_word_length: int = 20) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.

    Handles both word-level and character-level padding.

    Args:
        batch: List of samples from NERDataset
        pad_token_id: Padding index for tokens
        pad_label_id: Padding index for labels
        pad_char_id: Padding index for characters
        use_char_features: Whether to process character features
        max_word_length: Maximum word length for character features

    Returns:
        Dictionary with padded batches and masks
    """
    # Get sequences
    token_ids = [item['token_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    lengths = [item['length'] for item in batch]

    # Find max length in this batch
    max_len = max(lengths)

    # Pad sequences
    padded_tokens = []
    padded_labels = []
    masks = []

    for tokens, labels, length in zip(token_ids, label_ids, lengths):
        # Padding needed
        padding_length = max_len - length

        # Pad tokens
        padded_token = torch.cat([
            tokens,
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        padded_tokens.append(padded_token)

        # Pad labels
        padded_label = torch.cat([
            labels,
            torch.full((padding_length,), pad_label_id, dtype=torch.long)
        ])
        padded_labels.append(padded_label)

        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.cat([
            torch.ones(length, dtype=torch.bool),
            torch.zeros(padding_length, dtype=torch.bool)
        ])
        masks.append(mask)

    # Stack into batches
    batch_tokens = torch.stack(padded_tokens)
    batch_labels = torch.stack(padded_labels)
    batch_masks = torch.stack(masks)
    batch_lengths = torch.tensor(lengths, dtype=torch.long)

    result = {
        'token_ids': batch_tokens,
        'label_ids': batch_labels,
        'mask': batch_masks,
        'lengths': batch_lengths
    }

    # Handle character features
    if use_char_features and 'char_ids' in batch[0]:
        char_ids_list = [item['char_ids'] for item in batch]

        # Pad character sequences (shape: batch_size x max_seq_len x max_word_len)
        padded_char_ids = []

        for char_ids, length in zip(char_ids_list, lengths):
            padding_length = max_len - length

            if padding_length > 0:
                # Create padding tensor (padding_length x max_word_length)
                char_padding = torch.full(
                    (padding_length, max_word_length),
                    pad_char_id,
                    dtype=torch.long
                )
                padded_char = torch.cat([char_ids, char_padding], dim=0)
            else:
                padded_char = char_ids

            padded_char_ids.append(padded_char)

        result['char_ids'] = torch.stack(padded_char_ids)

    return result


def create_dataloader(filepath: str,
                      word_vocab: Vocabulary,
                      label_vocab: LabelVocabulary,
                      char_vocab: Optional[CharVocabulary] = None,
                      batch_size: int = 32,
                      shuffle: bool = False,
                      max_seq_length: Optional[int] = None,
                      max_word_length: int = 20,
                      use_char_features: bool = False,
                      num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for NER data.

    Args:
        filepath: Path to BIO-formatted file
        word_vocab: Word vocabulary
        label_vocab: Label vocabulary
        char_vocab: Character vocabulary (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_seq_length: Maximum sequence length
        max_word_length: Maximum word length for character features
        use_char_features: Whether to use character features
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance
    """
    dataset = NERDataset(
        filepath=filepath,
        word_vocab=word_vocab,
        label_vocab=label_vocab,
        char_vocab=char_vocab,
        max_seq_length=max_seq_length,
        max_word_length=max_word_length,
        use_char_features=use_char_features
    )

    # Determine pad indices
    pad_char_id = char_vocab.pad_idx if char_vocab else 0

    # Create collate function with vocabulary pad indices
    def collate_wrapper(batch):
        return collate_fn(
            batch,
            pad_token_id=word_vocab.pad_idx,
            pad_label_id=label_vocab.pad_idx,
            pad_char_id=pad_char_id,
            use_char_features=use_char_features,
            max_word_length=max_word_length
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=num_workers
    )

    return dataloader


def compute_class_weights(filepath: str,
                          label_vocab: LabelVocabulary) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced labels.

    Uses inverse frequency weighting normalized so minimum weight is 1.0.

    Args:
        filepath: Path to BIO-formatted file
        label_vocab: Label vocabulary

    Returns:
        Tensor of weights indexed by label ID
    """
    label_counts = {label: 0 for label in label_vocab.label2idx.keys()}

    current_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                for label in current_labels:
                    if label in label_counts:
                        label_counts[label] += 1
                current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current_labels.append(parts[1])

        # Don't forget last sentence
        for label in current_labels:
            if label in label_counts:
                label_counts[label] += 1

    # Compute total and weights
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # Create weight tensor
    weights = torch.zeros(len(label_vocab))

    for label, idx in label_vocab.label2idx.items():
        count = label_counts.get(label, 1)  # Avoid division by zero
        # Inverse frequency weight
        weights[idx] = total / (num_classes * count)

    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()

    return weights


def get_data_statistics(filepath: str) -> Dict[str, float]:
    """
    Compute statistics about the dataset.

    Args:
        filepath: Path to BIO-formatted file

    Returns:
        Dictionary with statistics
    """
    sentence_lengths = []
    word_lengths = []
    label_counts = {}
    entity_counts = {'Chemical': 0, 'Disease': 0}

    current_tokens = []
    current_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentence_lengths.append(len(current_tokens))
                    word_lengths.extend([len(t) for t in current_tokens])

                    for label in current_labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                        if label.startswith('B-Chemical'):
                            entity_counts['Chemical'] += 1
                        elif label.startswith('B-Disease'):
                            entity_counts['Disease'] += 1

                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)

        if current_tokens:
            sentence_lengths.append(len(current_tokens))
            word_lengths.extend([len(t) for t in current_tokens])
            for label in current_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                if label.startswith('B-Chemical'):
                    entity_counts['Chemical'] += 1
                elif label.startswith('B-Disease'):
                    entity_counts['Disease'] += 1

    stats = {
        'num_sentences': len(sentence_lengths),
        'num_tokens': sum(sentence_lengths),
        'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
        'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
        'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
        'max_word_length': max(word_lengths) if word_lengths else 0,
        'label_distribution': label_counts,
        'entity_counts': entity_counts
    }

    return stats


def print_data_statistics(train_file: str, dev_file: str, test_file: str) -> None:
    """
    Print statistics for train, dev, and test sets.

    Args:
        train_file: Path to training file
        dev_file: Path to development file
        test_file: Path to test file
    """
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    for split_name, filepath in [("Train", train_file), ("Dev", dev_file), ("Test", test_file)]:
        stats = get_data_statistics(filepath)
        print(f"\n{split_name} Set:")
        print(f"  Sentences: {stats['num_sentences']}")
        print(f"  Tokens: {stats['num_tokens']}")
        print(f"  Avg sentence length: {stats['avg_sentence_length']:.2f} tokens")
        print(f"  Max sentence length: {stats['max_sentence_length']} tokens")
        print(f"  Avg word length: {stats['avg_word_length']:.2f} chars")
        print(f"  Max word length: {stats['max_word_length']} chars")
        print(f"  Entities: Chemical={stats['entity_counts']['Chemical']}, Disease={stats['entity_counts']['Disease']}")
        print(f"  Label distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            print(f"    {label}: {count}")

    print("=" * 70)
