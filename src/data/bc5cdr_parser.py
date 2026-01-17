"""
BC5CDR Dataset Parser for Named Entity Recognition.

This module provides functionality to:
1. Download the BC5CDR (BioCreative V Chemical Disease Relation) dataset
2. Parse the PubTator format
3. Convert annotations to BIO (Begin-Inside-Outside) tagging scheme
4. Validate BIO sequences for consistency
5. Generate train/dev/test splits

The BC5CDR corpus contains:
- 1500 PubMed articles (title + abstract)
- Chemical and Disease entity annotations
- Train: 500 articles, Dev: 500 articles, Test: 500 articles

Author: NLP Course Project
"""

import os
import re
import urllib.request
import zipfile
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Entity:
    """Represents a named entity annotation."""
    start: int  # Character offset start
    end: int    # Character offset end
    text: str   # Entity text
    entity_type: str  # 'Chemical' or 'Disease'
    mesh_id: Optional[str] = None  # MeSH identifier


@dataclass
class Document:
    """Represents a PubMed document with annotations."""
    pmid: str  # PubMed ID
    title: str
    abstract: str
    entities: List[Entity]

    @property
    def full_text(self) -> str:
        """Return title + abstract with a space separator."""
        return f"{self.title} {self.abstract}"


def download_bc5cdr(output_dir: str, force: bool = False) -> str:
    """
    Download BC5CDR dataset from official source.

    The dataset is available from BioCreative:
    https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/

    Args:
        output_dir: Directory to save downloaded files
        force: If True, re-download even if files exist

    Returns:
        Path to the extracted dataset directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # BC5CDR dataset URL (from BioCreative)
    # Note: This URL may need updating if the source changes
    dataset_url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"

    zip_path = os.path.join(output_dir, "CDR_Data.zip")
    extract_dir = os.path.join(output_dir, "CDR_Data")

    # Check if already downloaded
    if os.path.exists(extract_dir) and not force:
        print(f"BC5CDR dataset already exists at {extract_dir}")
        return extract_dir

    print(f"Downloading BC5CDR dataset from {dataset_url}...")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/")
        print("2. Download 'CDR_Data.zip'")
        print(f"3. Extract to: {output_dir}")
        raise

    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Dataset extracted to {extract_dir}")
    return extract_dir


def parse_pubtator_file(filepath: str) -> List[Document]:
    """
    Parse a PubTator format file.

    PubTator format:
    - Title line: PMID|t|Title text
    - Abstract line: PMID|a|Abstract text
    - Entity lines: PMID\tstart\tend\tentity_text\tentity_type\tMeSH_ID
    - Blank line separates documents

    Args:
        filepath: Path to PubTator file

    Returns:
        List of Document objects
    """
    documents = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines (document separator)
    doc_texts = content.strip().split('\n\n')

    for doc_text in doc_texts:
        if not doc_text.strip():
            continue

        lines = doc_text.strip().split('\n')

        pmid = None
        title = ""
        abstract = ""
        entities = []

        for line in lines:
            if '|t|' in line:
                # Title line
                parts = line.split('|t|', 1)
                pmid = parts[0]
                title = parts[1] if len(parts) > 1 else ""

            elif '|a|' in line:
                # Abstract line
                parts = line.split('|a|', 1)
                abstract = parts[1] if len(parts) > 1 else ""

            elif '\t' in line:
                # Entity annotation line
                parts = line.split('\t')
                if len(parts) >= 5:
                    try:
                        entity_pmid = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        text = parts[3]
                        entity_type = parts[4]
                        mesh_id = parts[5] if len(parts) > 5 else None

                        # Normalize entity type
                        if entity_type.lower() in ['chemical', 'drug']:
                            entity_type = 'Chemical'
                        elif entity_type.lower() == 'disease':
                            entity_type = 'Disease'

                        entities.append(Entity(
                            start=start,
                            end=end,
                            text=text,
                            entity_type=entity_type,
                            mesh_id=mesh_id
                        ))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse entity line: {line}")
                        continue

        if pmid:
            documents.append(Document(
                pmid=pmid,
                title=title,
                abstract=abstract,
                entities=entities
            ))

    return documents


def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text and return tokens with their character offsets.

    Uses a simple but effective tokenization strategy:
    - Split on whitespace
    - Separate punctuation as individual tokens
    - Track character offsets for each token

    Args:
        text: Input text

    Returns:
        List of (token, start_offset, end_offset) tuples
    """
    tokens = []

    # Pattern to match words, numbers, or individual punctuation
    # This handles cases like "COVID-19" and "IL-6"
    pattern = r'\S+'

    for match in re.finditer(pattern, text):
        word = match.group()
        start = match.start()

        # Further split on punctuation boundaries while keeping offsets correct
        sub_tokens = _split_on_punctuation(word, start)
        tokens.extend(sub_tokens)

    return tokens


def _split_on_punctuation(word: str, start_offset: int) -> List[Tuple[str, int, int]]:
    """
    Split a word on punctuation while preserving character offsets.

    Handles cases like:
    - "word." -> ["word", "."]
    - "(word)" -> ["(", "word", ")"]
    - "word-word" -> ["word", "-", "word"] (optional, configurable)

    Args:
        word: Input word
        start_offset: Starting character offset

    Returns:
        List of (token, start, end) tuples
    """
    tokens = []
    current_token = ""
    current_start = start_offset

    for i, char in enumerate(word):
        if char in '.,;:!?()[]{}"\'/':
            # Save current token if any
            if current_token:
                tokens.append((
                    current_token,
                    current_start,
                    current_start + len(current_token)
                ))
                current_token = ""

            # Add punctuation as separate token
            tokens.append((char, start_offset + i, start_offset + i + 1))
            current_start = start_offset + i + 1
        else:
            if not current_token:
                current_start = start_offset + i
            current_token += char

    # Don't forget the last token
    if current_token:
        tokens.append((
            current_token,
            current_start,
            current_start + len(current_token)
        ))

    return tokens


def assign_bio_tags(
    tokens: List[Tuple[str, int, int]],
    entities: List[Entity]
) -> List[str]:
    """
    Assign BIO tags to tokens based on entity annotations.

    BIO scheme:
    - B-X: Beginning of entity type X
    - I-X: Inside entity type X
    - O: Outside any entity

    Args:
        tokens: List of (token, start, end) tuples
        entities: List of Entity annotations

    Returns:
        List of BIO tags corresponding to tokens
    """
    tags = ['O'] * len(tokens)

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e.start)

    for entity in sorted_entities:
        entity_tokens = []

        # Find tokens that overlap with this entity
        for idx, (token, tok_start, tok_end) in enumerate(tokens):
            # Check for overlap
            if tok_start < entity.end and tok_end > entity.start:
                entity_tokens.append(idx)

        # Assign tags
        for i, tok_idx in enumerate(entity_tokens):
            if i == 0:
                tags[tok_idx] = f'B-{entity.entity_type}'
            else:
                tags[tok_idx] = f'I-{entity.entity_type}'

    return tags


def validate_bio_sequence(tags: List[str]) -> Tuple[bool, str]:
    """
    Validate a BIO tag sequence for consistency.

    Rules:
    - I-X can only follow B-X or I-X of the same type
    - No other constraints for B- and O tags

    Args:
        tags: List of BIO tags

    Returns:
        Tuple of (is_valid, error_message)
    """
    prev_tag = 'O'

    for i, tag in enumerate(tags):
        if tag.startswith('I-'):
            entity_type = tag[2:]
            expected_prev = [f'B-{entity_type}', f'I-{entity_type}']

            if prev_tag not in expected_prev:
                return False, f"Invalid I- tag at position {i}: {tag} follows {prev_tag}"

        prev_tag = tag

    return True, "Valid"


def fix_bio_sequence(tags: List[str]) -> List[str]:
    """
    Fix invalid BIO sequences by converting orphan I- tags to B- tags.

    Args:
        tags: List of BIO tags (potentially invalid)

    Returns:
        Fixed list of BIO tags
    """
    fixed_tags = tags.copy()
    prev_tag = 'O'

    for i, tag in enumerate(fixed_tags):
        if tag.startswith('I-'):
            entity_type = tag[2:]
            expected_prev = [f'B-{entity_type}', f'I-{entity_type}']

            if prev_tag not in expected_prev:
                # Convert orphan I- to B-
                fixed_tags[i] = f'B-{entity_type}'

        prev_tag = fixed_tags[i]

    return fixed_tags


def document_to_bio_sentences(
    doc: Document,
    max_sentence_length: int = 128
) -> List[Tuple[List[str], List[str]]]:
    """
    Convert a document to BIO-tagged sentences.

    Performs sentence segmentation on the document text and assigns
    BIO tags based on entity annotations.

    Args:
        doc: Document object
        max_sentence_length: Maximum tokens per sentence (for splitting)

    Returns:
        List of (tokens, tags) tuples, one per sentence
    """
    sentences = []

    # Get full text with proper offset handling
    # Title ends at len(title), abstract starts at len(title) + 1 (space)
    full_text = doc.full_text
    title_offset = len(doc.title) + 1  # +1 for the space

    # Adjust entity offsets for abstract entities
    adjusted_entities = []
    for entity in doc.entities:
        adjusted_entities.append(entity)

    # Tokenize
    tokens = tokenize_text(full_text)

    if not tokens:
        return sentences

    # Assign BIO tags
    tags = assign_bio_tags(tokens, adjusted_entities)

    # Fix any invalid sequences
    tags = fix_bio_sequence(tags)

    # Simple sentence segmentation based on periods
    sentence_boundaries = []
    current_start = 0

    for i, (token, _, _) in enumerate(tokens):
        if token in '.!?' and i > 0:
            sentence_boundaries.append((current_start, i + 1))
            current_start = i + 1

    # Add last sentence if not ended with punctuation
    if current_start < len(tokens):
        sentence_boundaries.append((current_start, len(tokens)))

    # Extract sentences
    for start, end in sentence_boundaries:
        sent_tokens = [t[0] for t in tokens[start:end]]
        sent_tags = tags[start:end]

        # Skip empty sentences
        if not sent_tokens:
            continue

        # Split long sentences
        if len(sent_tokens) > max_sentence_length:
            for chunk_start in range(0, len(sent_tokens), max_sentence_length):
                chunk_end = min(chunk_start + max_sentence_length, len(sent_tokens))
                chunk_tokens = sent_tokens[chunk_start:chunk_end]
                chunk_tags = sent_tags[chunk_start:chunk_end]

                # Fix chunk tags (first token should be B- if it was I-)
                chunk_tags = fix_bio_sequence(chunk_tags)
                sentences.append((chunk_tokens, chunk_tags))
        else:
            sentences.append((sent_tokens, sent_tags))

    return sentences


def convert_to_bio_format(
    documents: List[Document],
    output_file: str,
    validate: bool = True
) -> Dict[str, int]:
    """
    Convert documents to BIO format and save to file.

    Output format:
    TOKEN\tTAG
    TOKEN\tTAG
    (blank line between sentences)

    Args:
        documents: List of Document objects
        output_file: Path to output file
        validate: Whether to validate BIO sequences

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_sentences': 0,
        'total_tokens': 0,
        'chemical_entities': 0,
        'disease_entities': 0,
        'invalid_sequences_fixed': 0
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            sentences = document_to_bio_sentences(doc)

            for tokens, tags in sentences:
                if validate:
                    is_valid, _ = validate_bio_sequence(tags)
                    if not is_valid:
                        tags = fix_bio_sequence(tags)
                        stats['invalid_sequences_fixed'] += 1

                # Write sentence
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")

                    # Count entities
                    if tag.startswith('B-Chemical'):
                        stats['chemical_entities'] += 1
                    elif tag.startswith('B-Disease'):
                        stats['disease_entities'] += 1

                f.write("\n")  # Blank line between sentences

                stats['total_sentences'] += 1
                stats['total_tokens'] += len(tokens)

    return stats


def process_bc5cdr_dataset(
    raw_dir: str,
    processed_dir: str,
    download: bool = True
) -> Dict[str, str]:
    """
    Process the complete BC5CDR dataset.

    Downloads (if needed), parses, and converts to BIO format.

    Args:
        raw_dir: Directory for raw data
        processed_dir: Directory for processed data
        download: Whether to download if not present

    Returns:
        Dictionary with paths to processed files
    """
    # Download if needed
    if download:
        try:
            data_dir = download_bc5cdr(raw_dir)
        except Exception as e:
            print(f"Could not download dataset: {e}")
            print("Please download manually or use sample data.")
            data_dir = raw_dir

    # Find PubTator files
    # BC5CDR typically has: CDR_TrainingSet.PubTator.txt, CDR_DevelopmentSet.PubTator.txt, CDR_TestSet.PubTator.txt
    train_file = os.path.join(raw_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TrainingSet.PubTator.txt")
    dev_file = os.path.join(raw_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_DevelopmentSet.PubTator.txt")
    test_file = os.path.join(raw_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TestSet.PubTator.txt")

    # Alternative paths
    if not os.path.exists(train_file):
        train_file = os.path.join(raw_dir, "CDR_TrainingSet.PubTator.txt")
        dev_file = os.path.join(raw_dir, "CDR_DevelopmentSet.PubTator.txt")
        test_file = os.path.join(raw_dir, "CDR_TestSet.PubTator.txt")

    output_files = {}
    os.makedirs(processed_dir, exist_ok=True)

    # Process each split
    splits = [
        ('train', train_file),
        ('dev', dev_file),
        ('test', test_file)
    ]

    for split_name, input_file in splits:
        output_file = os.path.join(processed_dir, f"{split_name}.txt")

        if os.path.exists(input_file):
            print(f"\nProcessing {split_name} set from {input_file}...")
            documents = parse_pubtator_file(input_file)
            stats = convert_to_bio_format(documents, output_file)

            print(f"  Documents: {len(documents)}")
            print(f"  Sentences: {stats['total_sentences']}")
            print(f"  Tokens: {stats['total_tokens']}")
            print(f"  Chemical entities: {stats['chemical_entities']}")
            print(f"  Disease entities: {stats['disease_entities']}")

            output_files[split_name] = output_file
        else:
            print(f"Warning: {split_name} file not found at {input_file}")

    return output_files


def get_dataset_statistics(bio_file: str) -> Dict:
    """
    Compute detailed statistics for a BIO-formatted dataset file.

    Args:
        bio_file: Path to BIO format file

    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_sentences': 0,
        'num_tokens': 0,
        'tag_counts': defaultdict(int),
        'entity_counts': defaultdict(int),
        'sentence_lengths': [],
        'entity_lengths': defaultdict(list),
        'unique_tokens': set()
    }

    current_sentence = []
    current_entity_length = 0
    current_entity_type = None

    with open(bio_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                # End of sentence
                if current_sentence:
                    stats['num_sentences'] += 1
                    stats['sentence_lengths'].append(len(current_sentence))
                    current_sentence = []

                # End current entity if any
                if current_entity_type:
                    stats['entity_lengths'][current_entity_type].append(current_entity_length)
                    current_entity_length = 0
                    current_entity_type = None
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                token, tag = parts[0], parts[1]

                current_sentence.append((token, tag))
                stats['num_tokens'] += 1
                stats['tag_counts'][tag] += 1
                stats['unique_tokens'].add(token.lower())

                # Track entities
                if tag.startswith('B-'):
                    # End previous entity
                    if current_entity_type:
                        stats['entity_lengths'][current_entity_type].append(current_entity_length)

                    # Start new entity
                    current_entity_type = tag[2:]
                    current_entity_length = 1
                    stats['entity_counts'][current_entity_type] += 1

                elif tag.startswith('I-'):
                    current_entity_length += 1

                else:  # O tag
                    # End current entity
                    if current_entity_type:
                        stats['entity_lengths'][current_entity_type].append(current_entity_length)
                        current_entity_length = 0
                        current_entity_type = None

    # Handle last sentence
    if current_sentence:
        stats['num_sentences'] += 1
        stats['sentence_lengths'].append(len(current_sentence))

    # Convert set to count
    stats['vocab_size'] = len(stats['unique_tokens'])
    del stats['unique_tokens']

    # Compute averages
    if stats['sentence_lengths']:
        stats['avg_sentence_length'] = sum(stats['sentence_lengths']) / len(stats['sentence_lengths'])
        stats['max_sentence_length'] = max(stats['sentence_lengths'])
        stats['min_sentence_length'] = min(stats['sentence_lengths'])

    for entity_type, lengths in stats['entity_lengths'].items():
        if lengths:
            stats[f'avg_{entity_type.lower()}_length'] = sum(lengths) / len(lengths)

    return stats


def print_dataset_statistics(stats: Dict, name: str = "Dataset"):
    """Print formatted dataset statistics."""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print('='*60)
    print(f"Sentences: {stats['num_sentences']:,}")
    print(f"Tokens: {stats['num_tokens']:,}")
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Avg sentence length: {stats.get('avg_sentence_length', 0):.1f}")
    print(f"Max sentence length: {stats.get('max_sentence_length', 0)}")

    print("\nTag distribution:")
    for tag, count in sorted(stats['tag_counts'].items()):
        pct = 100 * count / stats['num_tokens']
        print(f"  {tag}: {count:,} ({pct:.1f}%)")

    print("\nEntity counts:")
    for entity_type, count in sorted(stats['entity_counts'].items()):
        avg_len = stats.get(f'avg_{entity_type.lower()}_length', 0)
        print(f"  {entity_type}: {count:,} (avg length: {avg_len:.1f} tokens)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process BC5CDR dataset')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Directory for raw data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory for processed data')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not present')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only compute statistics for existing files')

    args = parser.parse_args()

    if args.stats_only:
        # Just compute statistics
        for split in ['train', 'dev', 'test']:
            filepath = os.path.join(args.processed_dir, f"{split}.txt")
            if os.path.exists(filepath):
                stats = get_dataset_statistics(filepath)
                print_dataset_statistics(stats, f"{split.upper()} Set")
    else:
        # Process dataset
        output_files = process_bc5cdr_dataset(
            args.raw_dir,
            args.processed_dir,
            download=args.download
        )

        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
        for split, path in output_files.items():
            print(f"  {split}: {path}")
