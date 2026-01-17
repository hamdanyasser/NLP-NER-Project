"""
Vocabulary management for words and tags in NER.

This module provides:
- Building vocabularies from training data
- Token-to-index and index-to-token mappings
- Handling special tokens (PAD, UNK)
- Saving and loading vocabularies
"""

import pickle
from typing import List, Dict, Optional
from collections import Counter


class Vocabulary:
    """
    Vocabulary class for managing token-to-index mappings.

    Args:
        pad_token: Padding token
        unk_token: Unknown token for out-of-vocabulary words
        min_freq: Minimum frequency for a token to be included
    """

    def __init__(self,
                 pad_token: str = "<PAD>",
                 unk_token: str = "<UNK>",
                 min_freq: int = 1):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq

        # Index mappings
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

        # Add special tokens
        self._add_token(self.pad_token)
        self._add_token(self.unk_token)

        # Special indices
        self.pad_idx = self.token2idx[pad_token]
        self.unk_idx = self.token2idx[unk_token]

    def _add_token(self, token: str) -> int:
        """
        Add a token to vocabulary.

        Args:
            token: Token to add

        Returns:
            Index of the token
        """
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            return idx
        return self.token2idx[token]

    def build_vocab(self, tokens_list: List[List[str]]) -> None:
        """
        Build vocabulary from a list of token sequences.

        Args:
            tokens_list: List of token sequences
        """
        # Count token frequencies
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)

        # Add tokens that meet minimum frequency
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self._add_token(token)

        print(f"Built vocabulary with {len(self.token2idx)} tokens "
              f"(min_freq={self.min_freq})")

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to indices.

        Args:
            tokens: List of tokens

        Returns:
            List of indices
        """
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert indices to tokens.

        Args:
            indices: List of indices

        Returns:
            List of tokens
        """
        return [self.idx2token.get(idx, self.unk_token) for idx in indices]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2idx)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token2idx

    def save(self, filepath: str) -> None:
        """
        Save vocabulary to file.

        Args:
            filepath: Path to save file
        """
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'min_freq': self.min_freq,
            'pad_idx': self.pad_idx,
            'unk_idx': self.unk_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        Load vocabulary from file.

        Args:
            filepath: Path to vocabulary file

        Returns:
            Loaded Vocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(
            pad_token=vocab_data['pad_token'],
            unk_token=vocab_data['unk_token'],
            min_freq=vocab_data['min_freq']
        )
        vocab.token2idx = vocab_data['token2idx']
        vocab.idx2token = vocab_data['idx2token']
        vocab.pad_idx = vocab_data['pad_idx']
        vocab.unk_idx = vocab_data['unk_idx']

        print(f"Vocabulary loaded from {filepath} ({len(vocab)} tokens)")
        return vocab


class LabelVocabulary:
    """
    Label vocabulary for NER tags (no UNK token needed).

    Args:
        pad_token: Padding token for labels
    """

    def __init__(self, pad_token: str = "<PAD>"):
        self.pad_token = pad_token

        # Index mappings
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}

        # Add PAD token
        self._add_label(self.pad_token)
        self.pad_idx = self.label2idx[pad_token]

    def _add_label(self, label: str) -> int:
        """
        Add a label to vocabulary.

        Args:
            label: Label to add

        Returns:
            Index of the label
        """
        if label not in self.label2idx:
            idx = len(self.label2idx)
            self.label2idx[label] = idx
            self.idx2label[idx] = label
            return idx
        return self.label2idx[label]

    def build_vocab(self, labels_list: List[List[str]]) -> None:
        """
        Build label vocabulary from a list of label sequences.

        Args:
            labels_list: List of label sequences
        """
        unique_labels = set()
        for labels in labels_list:
            unique_labels.update(labels)

        # Add labels in sorted order for consistency
        for label in sorted(unique_labels):
            if label != self.pad_token:  # PAD already added
                self._add_label(label)

        print(f"Built label vocabulary with {len(self.label2idx)} labels: "
              f"{list(self.label2idx.keys())}")

    def encode(self, labels: List[str]) -> List[int]:
        """
        Convert labels to indices.

        Args:
            labels: List of labels

        Returns:
            List of indices
        """
        return [self.label2idx[label] for label in labels]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert indices to labels.

        Args:
            indices: List of indices

        Returns:
            List of labels
        """
        return [self.idx2label[idx] for idx in indices]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.label2idx)

    def __contains__(self, label: str) -> bool:
        """Check if label is in vocabulary."""
        return label in self.label2idx

    def save(self, filepath: str) -> None:
        """
        Save label vocabulary to file.

        Args:
            filepath: Path to save file
        """
        vocab_data = {
            'label2idx': self.label2idx,
            'idx2label': self.idx2label,
            'pad_token': self.pad_token,
            'pad_idx': self.pad_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Label vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LabelVocabulary':
        """
        Load label vocabulary from file.

        Args:
            filepath: Path to vocabulary file

        Returns:
            Loaded LabelVocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(pad_token=vocab_data['pad_token'])
        vocab.label2idx = vocab_data['label2idx']
        vocab.idx2label = vocab_data['idx2label']
        vocab.pad_idx = vocab_data['pad_idx']

        print(f"Label vocabulary loaded from {filepath} ({len(vocab)} labels)")
        return vocab


class CharVocabulary:
    """
    Character vocabulary for character-level features.

    Builds vocabulary from all unique characters in the token set.
    Includes PAD and UNK characters for padding and unknown chars.

    Args:
        pad_char: Padding character
        unk_char: Unknown character
    """

    def __init__(self, pad_char: str = "<PAD>", unk_char: str = "<UNK>"):
        self.pad_char = pad_char
        self.unk_char = unk_char

        # Index mappings
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        # Add special characters
        self._add_char(self.pad_char)
        self._add_char(self.unk_char)

        # Special indices
        self.pad_idx = self.char2idx[pad_char]
        self.unk_idx = self.char2idx[unk_char]

    def _add_char(self, char: str) -> int:
        """
        Add a character to vocabulary.

        Args:
            char: Character to add

        Returns:
            Index of the character
        """
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            return idx
        return self.char2idx[char]

    def build_vocab(self, tokens_list: List[List[str]]) -> None:
        """
        Build character vocabulary from a list of token sequences.

        Extracts all unique characters from all tokens.

        Args:
            tokens_list: List of token sequences
        """
        unique_chars = set()
        for tokens in tokens_list:
            for token in tokens:
                unique_chars.update(token)

        # Add characters in sorted order for consistency
        for char in sorted(unique_chars):
            self._add_char(char)

        print(f"Built character vocabulary with {len(self.char2idx)} characters")

    def encode_word(self, word: str, max_len: Optional[int] = None) -> List[int]:
        """
        Convert a word to character indices.

        Args:
            word: Word to encode
            max_len: Maximum length (truncate if longer, pad if shorter)

        Returns:
            List of character indices
        """
        char_ids = [self.char2idx.get(c, self.unk_idx) for c in word]

        if max_len is not None:
            if len(char_ids) > max_len:
                char_ids = char_ids[:max_len]
            else:
                char_ids = char_ids + [self.pad_idx] * (max_len - len(char_ids))

        return char_ids

    def encode_sequence(self, tokens: List[str], max_word_len: int = 20) -> List[List[int]]:
        """
        Convert a sequence of tokens to character index matrix.

        Args:
            tokens: List of tokens
            max_word_len: Maximum word length

        Returns:
            List of character index lists (shape: num_tokens x max_word_len)
        """
        return [self.encode_word(token, max_word_len) for token in tokens]

    def decode(self, indices: List[int]) -> str:
        """
        Convert character indices back to string.

        Args:
            indices: List of character indices

        Returns:
            Decoded string
        """
        chars = [self.idx2char.get(idx, self.unk_char) for idx in indices
                 if idx != self.pad_idx]
        return ''.join(chars)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)

    def __contains__(self, char: str) -> bool:
        """Check if character is in vocabulary."""
        return char in self.char2idx

    def save(self, filepath: str) -> None:
        """
        Save character vocabulary to file.

        Args:
            filepath: Path to save file
        """
        vocab_data = {
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'pad_char': self.pad_char,
            'unk_char': self.unk_char,
            'pad_idx': self.pad_idx,
            'unk_idx': self.unk_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Character vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CharVocabulary':
        """
        Load character vocabulary from file.

        Args:
            filepath: Path to vocabulary file

        Returns:
            Loaded CharVocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(
            pad_char=vocab_data['pad_char'],
            unk_char=vocab_data['unk_char']
        )
        vocab.char2idx = vocab_data['char2idx']
        vocab.idx2char = vocab_data['idx2char']
        vocab.pad_idx = vocab_data['pad_idx']
        vocab.unk_idx = vocab_data['unk_idx']

        print(f"Character vocabulary loaded from {filepath} ({len(vocab)} characters)")
        return vocab


def build_vocabularies_from_file(filepath: str,
                                  min_word_freq: int = 1,
                                  pad_token: str = "<PAD>",
                                  unk_token: str = "<UNK>",
                                  build_char_vocab: bool = True) -> tuple:
    """
    Build word, label, and optionally character vocabularies from a BIO-formatted file.

    Args:
        filepath: Path to BIO-formatted file
        min_word_freq: Minimum frequency for words
        pad_token: Padding token
        unk_token: Unknown token
        build_char_vocab: Whether to build character vocabulary

    Returns:
        (word_vocab, label_vocab) or (word_vocab, label_vocab, char_vocab) tuple
    """
    sentences_tokens = []
    sentences_labels = []

    current_tokens = []
    current_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line = sentence boundary
                if current_tokens:
                    sentences_tokens.append(current_tokens)
                    sentences_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)

        # Don't forget the last sentence
        if current_tokens:
            sentences_tokens.append(current_tokens)
            sentences_labels.append(current_labels)

    # Build vocabularies
    word_vocab = Vocabulary(pad_token=pad_token, unk_token=unk_token, min_freq=min_word_freq)
    word_vocab.build_vocab(sentences_tokens)

    label_vocab = LabelVocabulary(pad_token=pad_token)
    label_vocab.build_vocab(sentences_labels)

    if build_char_vocab:
        char_vocab = CharVocabulary(pad_char=pad_token, unk_char=unk_token)
        char_vocab.build_vocab(sentences_tokens)
        return word_vocab, label_vocab, char_vocab

    return word_vocab, label_vocab
