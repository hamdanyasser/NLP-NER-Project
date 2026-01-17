#!/usr/bin/env python3
"""
Interactive Demo for BiLSTM-CRF NER Model

This script provides a command-line interface for testing the trained
NER model on custom biomedical text. Users can input sentences and see
extracted Chemical and Disease entities.

Usage:
    python demo.py
    python demo.py --model artifacts/best_model.pt

Author: Yasser Hamdan & Hassan Najdi
Course: NLP Course Project
"""

import os
import re
import sys
import argparse
import yaml
import torch
from typing import List, Tuple, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.vocab import Vocabulary, LabelVocabulary, CharVocabulary
from src.models.bilstm_crf import BiLSTMCRF


# ANSI color codes for terminal output
class Colors:
    """Terminal colors for highlighting entities."""
    CHEMICAL = '\033[94m'  # Blue
    DISEASE = '\033[91m'   # Red
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for biomedical text.

    Splits on whitespace and separates punctuation.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    # Split on whitespace first
    tokens = text.split()

    # Further split punctuation from words
    result = []
    for token in tokens:
        # Separate leading punctuation
        while token and token[0] in '([{':
            result.append(token[0])
            token = token[1:]

        # Separate trailing punctuation
        trailing = []
        while token and token[-1] in '.,;:!?)]}':
            trailing.insert(0, token[-1])
            token = token[:-1]

        if token:
            result.append(token)
        result.extend(trailing)

    return result


def extract_entities(tokens: List[str], tags: List[str]) -> List[Dict]:
    """
    Extract entities from BIO tags.

    Args:
        tokens: List of tokens
        tags: List of BIO tags

    Returns:
        List of entity dictionaries with text, type, start, end
    """
    entities = []
    current_entity = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append(current_entity)

            # Start new entity
            entity_type = tag[2:]
            current_entity = {
                'text': token,
                'type': entity_type,
                'start': i,
                'end': i
            }

        elif tag.startswith('I-') and current_entity:
            # Continue current entity
            entity_type = tag[2:]
            if entity_type == current_entity['type']:
                current_entity['text'] += ' ' + token
                current_entity['end'] = i
            else:
                # Type mismatch, save current and start new
                entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': entity_type,
                    'start': i,
                    'end': i
                }

        else:  # O tag
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)

    return entities


def highlight_text(tokens: List[str], tags: List[str], use_colors: bool = True) -> str:
    """
    Create highlighted text with entity annotations.

    Args:
        tokens: List of tokens
        tags: List of BIO tags
        use_colors: Whether to use ANSI colors

    Returns:
        Formatted string with entity highlighting
    """
    result = []
    i = 0

    while i < len(tokens):
        tag = tags[i]

        if tag.startswith('B-'):
            entity_type = tag[2:]
            entity_tokens = [tokens[i]]

            # Collect all tokens in this entity
            j = i + 1
            while j < len(tags) and tags[j].startswith('I-'):
                entity_tokens.append(tokens[j])
                j += 1

            entity_text = ' '.join(entity_tokens)

            # Format based on entity type
            if use_colors:
                if entity_type == 'Chemical':
                    formatted = f"{Colors.BOLD}{Colors.CHEMICAL}[{entity_text}]{Colors.RESET}"
                elif entity_type == 'Disease':
                    formatted = f"{Colors.BOLD}{Colors.DISEASE}[{entity_text}]{Colors.RESET}"
                else:
                    formatted = f"[{entity_text}]"
            else:
                formatted = f"[{entity_text}|{entity_type}]"

            result.append(formatted)
            i = j
        else:
            result.append(tokens[i])
            i += 1

    return ' '.join(result)


class NERDemo:
    """Interactive NER demo class."""

    def __init__(self,
                 model_path: str = 'artifacts/best_model.pt',
                 config_path: str = 'config/config.yaml',
                 vocab_dir: str = 'artifacts'):
        """
        Initialize the demo.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            vocab_dir: Directory containing vocabulary files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load vocabularies
        print("Loading vocabularies...")
        self.word_vocab = Vocabulary.load(os.path.join(vocab_dir, 'vocab_word.pkl'))
        self.label_vocab = LabelVocabulary.load(os.path.join(vocab_dir, 'vocab_label.pkl'))

        # Check for character vocabulary
        self.char_vocab = None
        self.use_char_features = self.config.get('model', {}).get('use_char_features', False)
        if self.use_char_features:
            char_vocab_path = os.path.join(vocab_dir, 'vocab_char.pkl')
            if os.path.exists(char_vocab_path):
                self.char_vocab = CharVocabulary.load(char_vocab_path)
            else:
                print("Warning: Character vocabulary not found, disabling char features")
                self.use_char_features = False

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")
        print(f"Word vocabulary: {len(self.word_vocab)} tokens")
        print(f"Label vocabulary: {len(self.label_vocab)} tags")
        if self.char_vocab:
            print(f"Character vocabulary: {len(self.char_vocab)} characters")

    def _load_model(self, model_path: str) -> BiLSTMCRF:
        """Load the trained model."""
        config = self.config

        model = BiLSTMCRF(
            vocab_size=len(self.word_vocab),
            num_tags=len(self.label_vocab),
            embedding_dim=config['model']['embedding_dim'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=self.word_vocab.pad_idx,
            use_char_features=self.use_char_features,
            num_chars=len(self.char_vocab) if self.char_vocab else 0,
            char_embedding_dim=config.get('model', {}).get('char_embedding_dim', 30),
            char_hidden_size=config.get('model', {}).get('char_hidden_size', 50),
            char_kernel_sizes=config.get('model', {}).get('char_kernel_sizes', [2, 3, 4]),
            use_highway=config.get('model', {}).get('use_highway', True),
            use_attention=config.get('model', {}).get('use_attention', False),
            attention_heads=config.get('model', {}).get('attention_heads', 4),
            attention_dropout=config.get('model', {}).get('attention_dropout', 0.1)
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def predict(self, text: str) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Predict entities in text.

        Args:
            text: Input text string

        Returns:
            Tuple of (tokens, tags, entities)
        """
        # Tokenize
        tokens = tokenize(text)

        if not tokens:
            return [], [], []

        # Encode tokens
        token_ids = self.word_vocab.encode(tokens)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Create mask
        mask = torch.ones(1, len(tokens), dtype=torch.bool, device=self.device)

        # Encode characters if needed
        char_tensor = None
        if self.use_char_features and self.char_vocab:
            max_word_len = self.config.get('model', {}).get('max_word_length', 20)
            char_ids = self.char_vocab.encode_sequence(tokens, max_word_len)
            char_tensor = torch.tensor([char_ids], dtype=torch.long, device=self.device)

        # Predict
        with torch.no_grad():
            pred_ids = self.model.predict(token_tensor, mask, char_tensor)

        # Decode tags
        tags = self.label_vocab.decode(pred_ids[0])

        # Extract entities
        entities = extract_entities(tokens, tags)

        return tokens, tags, entities

    def run_interactive(self):
        """Run interactive demo loop."""
        print("\n" + "=" * 60)
        print("  BiLSTM-CRF Named Entity Recognition Demo")
        print("  Biomedical NER for Chemical and Disease Entities")
        print("=" * 60)
        print("\nEntity Types:")
        print(f"  {Colors.CHEMICAL}[Chemical]{Colors.RESET} - Drugs, compounds, chemicals")
        print(f"  {Colors.DISEASE}[Disease]{Colors.RESET}  - Conditions, symptoms, diseases")
        print("\nCommands:")
        print("  Type a sentence to analyze")
        print("  Type 'quit' or 'exit' to exit")
        print("  Type 'examples' to see example sentences")
        print("=" * 60 + "\n")

        while True:
            try:
                text = input("Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not text:
                continue

            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if text.lower() == 'examples':
                self._show_examples()
                continue

            # Process the text
            tokens, tags, entities = self.predict(text)

            if not tokens:
                print("No tokens found in input.")
                continue

            # Display results
            print("\n" + "-" * 50)
            print("Results:")
            print("-" * 50)

            # Highlighted text
            highlighted = highlight_text(tokens, tags, use_colors=True)
            print(f"\n{highlighted}\n")

            # Entity list
            if entities:
                print("Entities found:")
                for ent in entities:
                    if ent['type'] == 'Chemical':
                        color = Colors.CHEMICAL
                    elif ent['type'] == 'Disease':
                        color = Colors.DISEASE
                    else:
                        color = Colors.RESET
                    print(f"  - {color}{ent['text']}{Colors.RESET} ({ent['type']})")
            else:
                print("No entities found.")

            print("-" * 50 + "\n")

    def _show_examples(self):
        """Show example sentences."""
        examples = [
            "Aspirin can cause gastrointestinal bleeding and ulcers.",
            "Metformin is commonly used to treat type 2 diabetes.",
            "Ibuprofen may lead to kidney damage in some patients.",
            "Chemotherapy drugs like doxorubicin can cause cardiotoxicity.",
            "Penicillin allergies may result in anaphylaxis.",
        ]

        print("\n" + "=" * 50)
        print("Example Sentences:")
        print("=" * 50)
        for i, ex in enumerate(examples, 1):
            print(f"{i}. {ex}")
        print("=" * 50 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive BiLSTM-CRF NER Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py
  python demo.py --model artifacts/best_model.pt
  python demo.py --config config/config.yaml

This demo allows you to test the trained NER model on custom text.
Enter biomedical sentences and see Chemical and Disease entities extracted.
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='artifacts/best_model.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--vocab-dir', '-v',
        type=str,
        default='artifacts',
        help='Directory containing vocabulary files'
    )
    parser.add_argument(
        '--text', '-t',
        type=str,
        default=None,
        help='Single text to analyze (non-interactive mode)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using: python -m src.training.train")
        sys.exit(1)

    # Initialize demo
    try:
        demo = NERDemo(
            model_path=args.model,
            config_path=args.config,
            vocab_dir=args.vocab_dir
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run in single-text or interactive mode
    if args.text:
        tokens, tags, entities = demo.predict(args.text)
        print("\nInput:", args.text)
        print("\nResult:", highlight_text(tokens, tags))
        if entities:
            print("\nEntities:")
            for ent in entities:
                print(f"  - {ent['text']} ({ent['type']})")
    else:
        demo.run_interactive()


if __name__ == '__main__':
    main()
