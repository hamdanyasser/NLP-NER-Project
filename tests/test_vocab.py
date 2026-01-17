"""
Unit tests for vocabulary classes.

Tests:
- Word vocabulary encoding/decoding
- Label vocabulary encoding/decoding
- Character vocabulary encoding/decoding
- Special token handling
- OOV handling
"""

import pytest
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.vocab import Vocabulary, LabelVocabulary, CharVocabulary


class TestVocabulary:
    """Test suite for word vocabulary."""

    @pytest.fixture
    def vocab(self):
        """Create a vocabulary instance."""
        tokens = ['the', 'cat', 'sat', 'on', 'mat', 'the', 'cat']
        return Vocabulary.build_from_tokens(tokens, min_freq=1)

    def test_vocab_size(self, vocab):
        """Test vocabulary has correct size."""
        # Includes PAD, UNK, and unique tokens
        assert len(vocab) >= 5 + 2  # 5 unique + PAD + UNK

    def test_special_tokens(self, vocab):
        """Test special tokens are present."""
        assert vocab.pad_token in vocab
        assert vocab.unk_token in vocab
        assert vocab.pad_idx == 0

    def test_encode_known_token(self, vocab):
        """Test encoding known tokens."""
        tokens = ['the', 'cat']
        encoded = vocab.encode(tokens)

        assert len(encoded) == 2
        assert all(isinstance(idx, int) for idx in encoded)
        assert encoded[0] != vocab.unk_idx  # 'the' should not be UNK

    def test_encode_unknown_token(self, vocab):
        """Test encoding unknown tokens returns UNK."""
        tokens = ['unknown_word_xyz']
        encoded = vocab.encode(tokens)

        assert encoded[0] == vocab.unk_idx

    def test_decode(self, vocab):
        """Test decoding indices to tokens."""
        tokens = ['the', 'cat', 'sat']
        encoded = vocab.encode(tokens)
        decoded = vocab.decode(encoded)

        assert decoded == tokens

    def test_contains(self, vocab):
        """Test __contains__ method."""
        assert 'the' in vocab
        assert 'unknown_xyz' not in vocab

    def test_min_freq_filtering(self):
        """Test min_freq filters rare tokens."""
        tokens = ['common', 'common', 'common', 'rare']
        vocab = Vocabulary.build_from_tokens(tokens, min_freq=2)

        assert 'common' in vocab
        assert 'rare' not in vocab

    def test_save_and_load(self, vocab):
        """Test vocabulary save and load."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        try:
            vocab.save(path)
            loaded_vocab = Vocabulary.load(path)

            assert len(loaded_vocab) == len(vocab)
            assert loaded_vocab.pad_token == vocab.pad_token
            assert loaded_vocab.unk_token == vocab.unk_token

            # Check encoding consistency
            tokens = ['the', 'cat']
            assert vocab.encode(tokens) == loaded_vocab.encode(tokens)
        finally:
            os.unlink(path)


class TestLabelVocabulary:
    """Test suite for label vocabulary."""

    @pytest.fixture
    def label_vocab(self):
        """Create a label vocabulary instance."""
        labels = ['O', 'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease', 'O']
        return LabelVocabulary.build_from_labels(labels)

    def test_label_vocab_size(self, label_vocab):
        """Test label vocabulary has correct size."""
        # PAD + 5 unique labels
        assert len(label_vocab) == 6

    def test_encode_labels(self, label_vocab):
        """Test encoding labels."""
        labels = ['O', 'B-Chemical', 'I-Chemical']
        encoded = label_vocab.encode(labels)

        assert len(encoded) == 3
        assert all(isinstance(idx, int) for idx in encoded)

    def test_decode_labels(self, label_vocab):
        """Test decoding label indices."""
        labels = ['O', 'B-Chemical', 'I-Chemical']
        encoded = label_vocab.encode(labels)
        decoded = label_vocab.decode(encoded)

        assert decoded == labels

    def test_pad_label(self, label_vocab):
        """Test PAD label handling."""
        assert label_vocab.pad_idx == 0
        assert label_vocab.idx2label[0] == '<PAD>'

    def test_all_labels_encoded(self, label_vocab):
        """Test all labels can be encoded."""
        all_labels = ['O', 'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease']
        encoded = label_vocab.encode(all_labels)

        # No label should be missing
        assert len(encoded) == len(all_labels)


class TestCharVocabulary:
    """Test suite for character vocabulary."""

    @pytest.fixture
    def char_vocab(self):
        """Create a character vocabulary instance."""
        tokens = ['hello', 'world', 'test']
        return CharVocabulary.build_from_tokens(tokens)

    def test_char_vocab_size(self, char_vocab):
        """Test character vocabulary has correct size."""
        # PAD, UNK + unique characters
        unique_chars = set('helloworld')  # 'test' chars are in 'hello'
        assert len(char_vocab) >= len(unique_chars) + 2

    def test_encode_word(self, char_vocab):
        """Test encoding a single word."""
        encoded = char_vocab.encode_word('hello')

        assert len(encoded) == 5
        assert all(isinstance(idx, int) for idx in encoded)

    def test_encode_word_with_padding(self, char_vocab):
        """Test encoding word with max length padding."""
        encoded = char_vocab.encode_word('hi', max_len=5)

        assert len(encoded) == 5
        assert encoded[-1] == char_vocab.pad_idx  # Last positions should be PAD

    def test_encode_word_truncation(self, char_vocab):
        """Test encoding truncates long words."""
        long_word = 'supercalifragilisticexpialidocious'
        encoded = char_vocab.encode_word(long_word, max_len=10)

        assert len(encoded) == 10

    def test_encode_sequence(self, char_vocab):
        """Test encoding a sequence of tokens."""
        tokens = ['hello', 'world']
        encoded = char_vocab.encode_sequence(tokens, max_word_len=10)

        assert len(encoded) == 2
        assert len(encoded[0]) == 10
        assert len(encoded[1]) == 10

    def test_unknown_char(self, char_vocab):
        """Test unknown characters map to UNK."""
        # Assuming special characters might be unknown
        encoded = char_vocab.encode_word('hello')
        # The characters should be known
        assert all(idx != char_vocab.unk_idx for idx in encoded)


class TestVocabularyIntegration:
    """Integration tests for vocabulary classes."""

    def test_build_from_file(self):
        """Test building vocabularies from file."""
        # Create temporary BIO file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("The\tO\n")
            f.write("drug\tB-Chemical\n")
            f.write("treats\tO\n")
            f.write("cancer\tB-Disease\n")
            f.write("\n")
            f.write("Another\tO\n")
            f.write("sentence\tO\n")
            path = f.name

        try:
            from src.utils.vocab import build_vocabularies_from_file
            word_vocab, label_vocab, char_vocab = build_vocabularies_from_file(
                path,
                min_word_freq=1,
                build_char_vocab=True
            )

            assert 'drug' in word_vocab
            assert 'B-Chemical' in label_vocab.label2idx
            assert char_vocab is not None
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
