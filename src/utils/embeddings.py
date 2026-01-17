"""
Pre-trained word embeddings loader.

This module provides:
- GloVe embeddings loading and processing
- Vocabulary-aligned embedding matrix creation
- Download utilities for embeddings

Author: NLP Course Project
"""

import os
import urllib.request
import zipfile
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.utils.vocab import Vocabulary


# GloVe download URLs
GLOVE_URLS = {
    '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_glove(
    output_dir: str,
    dim: int = 100,
    corpus: str = '6B',
    force: bool = False
) -> str:
    """
    Download GloVe embeddings.

    Args:
        output_dir: Directory to save embeddings
        dim: Embedding dimension (50, 100, 200, 300 for 6B; 300 for 840B)
        corpus: Corpus name ('6B' or '840B')
        force: Re-download even if exists

    Returns:
        Path to the embeddings file
    """
    os.makedirs(output_dir, exist_ok=True)

    if corpus == '6B':
        filename = f'glove.6B.{dim}d.txt'
    else:
        filename = f'glove.840B.300d.txt'

    filepath = os.path.join(output_dir, filename)

    # Check if already exists
    if os.path.exists(filepath) and not force:
        print(f"GloVe embeddings already exist at {filepath}")
        return filepath

    # Download
    url = GLOVE_URLS.get(corpus)
    if url is None:
        raise ValueError(f"Unknown corpus: {corpus}")

    zip_path = os.path.join(output_dir, f'glove.{corpus}.zip')

    print(f"Downloading GloVe {corpus} embeddings...")
    print(f"URL: {url}")
    print("This may take a while (several GB download)...")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Download") as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    except Exception as e:
        print(f"\nError downloading: {e}")
        print("\nManual download instructions:")
        print(f"1. Visit: {url}")
        print(f"2. Download the zip file")
        print(f"3. Extract {filename} to {output_dir}")
        raise

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Clean up zip file
    os.remove(zip_path)

    print(f"GloVe embeddings saved to {filepath}")
    return filepath


def load_glove_embeddings(
    glove_path: str,
    vocab: Vocabulary,
    embedding_dim: int
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load GloVe embeddings and create vocabulary-aligned embedding matrix.

    Args:
        glove_path: Path to GloVe file (e.g., glove.6B.100d.txt)
        vocab: Vocabulary object
        embedding_dim: Expected embedding dimension

    Returns:
        Tuple of (embedding_matrix, coverage_stats)
        - embedding_matrix: Tensor of shape (vocab_size, embedding_dim)
        - coverage_stats: Dictionary with coverage statistics
    """
    print(f"Loading GloVe embeddings from {glove_path}...")

    # Initialize embedding matrix with random values
    embedding_matrix = torch.randn(len(vocab), embedding_dim) * 0.1

    # Set PAD embedding to zeros
    embedding_matrix[vocab.pad_idx] = torch.zeros(embedding_dim)

    # Load GloVe vectors
    glove_vectors = {}
    found_count = 0

    with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading GloVe"):
            values = line.strip().split()
            if len(values) < embedding_dim + 1:
                continue  # Skip malformed lines

            word = values[0]
            try:
                vector = np.array(values[1:embedding_dim + 1], dtype=np.float32)
                if len(vector) == embedding_dim:
                    glove_vectors[word] = vector
            except ValueError:
                continue

    print(f"Loaded {len(glove_vectors):,} GloVe vectors")

    # Map GloVe vectors to vocabulary
    oov_words = []
    for word, idx in vocab.token2idx.items():
        if word in glove_vectors:
            embedding_matrix[idx] = torch.from_numpy(glove_vectors[word])
            found_count += 1
        elif word.lower() in glove_vectors:
            # Try lowercase
            embedding_matrix[idx] = torch.from_numpy(glove_vectors[word.lower()])
            found_count += 1
        else:
            if word not in [vocab.pad_token, vocab.unk_token]:
                oov_words.append(word)

    # Calculate coverage statistics
    total_vocab = len(vocab) - 2  # Exclude PAD and UNK
    coverage = found_count / total_vocab if total_vocab > 0 else 0.0

    stats = {
        'total_vocab': len(vocab),
        'found_in_glove': found_count,
        'oov_count': len(oov_words),
        'coverage_percent': coverage * 100
    }

    print(f"Vocabulary coverage: {found_count}/{total_vocab} ({coverage*100:.1f}%)")
    print(f"OOV words: {len(oov_words)}")

    if oov_words and len(oov_words) <= 20:
        print(f"Sample OOV words: {oov_words[:20]}")

    return embedding_matrix, stats


def initialize_embeddings(
    vocab: Vocabulary,
    embedding_dim: int,
    pretrained_path: Optional[str] = None,
    freeze: bool = False
) -> Tuple[torch.Tensor, bool]:
    """
    Initialize word embeddings, optionally from pretrained vectors.

    Args:
        vocab: Vocabulary object
        embedding_dim: Embedding dimension
        pretrained_path: Optional path to pretrained embeddings
        freeze: Whether embeddings should be frozen during training

    Returns:
        Tuple of (embedding_matrix, should_freeze)
    """
    if pretrained_path and os.path.exists(pretrained_path):
        embedding_matrix, stats = load_glove_embeddings(
            pretrained_path,
            vocab,
            embedding_dim
        )
        return embedding_matrix, freeze
    else:
        # Random initialization
        print("No pretrained embeddings found. Using random initialization.")
        embedding_matrix = torch.randn(len(vocab), embedding_dim) * 0.1
        embedding_matrix[vocab.pad_idx] = torch.zeros(embedding_dim)
        return embedding_matrix, False


def compute_embedding_statistics(embedding_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about embedding matrix.

    Args:
        embedding_matrix: Embedding tensor

    Returns:
        Dictionary with statistics
    """
    stats = {
        'vocab_size': embedding_matrix.size(0),
        'embedding_dim': embedding_matrix.size(1),
        'mean': embedding_matrix.mean().item(),
        'std': embedding_matrix.std().item(),
        'min': embedding_matrix.min().item(),
        'max': embedding_matrix.max().item(),
        'norm_mean': embedding_matrix.norm(dim=1).mean().item()
    }
    return stats


def get_word_similarity(
    word1: str,
    word2: str,
    vocab: Vocabulary,
    embedding_matrix: torch.Tensor
) -> Optional[float]:
    """
    Compute cosine similarity between two words.

    Args:
        word1: First word
        word2: Second word
        vocab: Vocabulary object
        embedding_matrix: Embedding tensor

    Returns:
        Cosine similarity or None if word not found
    """
    if word1 not in vocab or word2 not in vocab:
        return None

    idx1 = vocab.token2idx[word1]
    idx2 = vocab.token2idx[word2]

    vec1 = embedding_matrix[idx1]
    vec2 = embedding_matrix[idx2]

    similarity = torch.cosine_similarity(
        vec1.unsqueeze(0),
        vec2.unsqueeze(0)
    ).item()

    return similarity


def get_most_similar(
    word: str,
    vocab: Vocabulary,
    embedding_matrix: torch.Tensor,
    top_k: int = 10
) -> list:
    """
    Find most similar words to a given word.

    Args:
        word: Query word
        vocab: Vocabulary object
        embedding_matrix: Embedding tensor
        top_k: Number of similar words to return

    Returns:
        List of (word, similarity) tuples
    """
    if word not in vocab:
        return []

    idx = vocab.token2idx[word]
    query_vec = embedding_matrix[idx].unsqueeze(0)

    # Compute similarities with all words
    similarities = torch.cosine_similarity(
        query_vec,
        embedding_matrix,
        dim=1
    )

    # Get top-k (excluding the query word itself)
    values, indices = similarities.topk(top_k + 1)

    results = []
    for i, (sim, word_idx) in enumerate(zip(values.tolist(), indices.tolist())):
        if word_idx != idx:  # Exclude query word
            results.append((vocab.idx2token[word_idx], sim))
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test download (optional)
    import argparse

    parser = argparse.ArgumentParser(description='Download GloVe embeddings')
    parser.add_argument('--output-dir', type=str, default='data/embeddings',
                        help='Output directory')
    parser.add_argument('--dim', type=int, default=100,
                        help='Embedding dimension')
    parser.add_argument('--corpus', type=str, default='6B',
                        choices=['6B', '840B'],
                        help='GloVe corpus')

    args = parser.parse_args()

    filepath = download_glove(
        args.output_dir,
        dim=args.dim,
        corpus=args.corpus
    )

    print(f"\nGloVe embeddings ready at: {filepath}")
