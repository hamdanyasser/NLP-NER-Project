"""
Evaluation script for NER models.

This script:
1. Loads trained model and vocabularies
2. Runs inference on test set
3. Computes and prints evaluation metrics
4. Optionally saves predictions to file

Author: NLP Course Project
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from typing import Optional

from src.utils.vocab import Vocabulary, LabelVocabulary, CharVocabulary
from src.data.dataset import create_dataloader
from src.models.baseline_tagger import BaselineBiLSTMTagger
from src.models.bilstm_crf import BiLSTMCRF
from src.utils.metrics import compute_metrics, print_metrics, get_classification_report


def load_model(
    checkpoint_path: str,
    vocab_size: int,
    num_tags: int,
    pad_idx: int,
    model_type: str,
    config: dict,
    device: torch.device,
    num_chars: int = 0
):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        vocab_size: Vocabulary size
        num_tags: Number of tags
        pad_idx: Padding index
        model_type: Type of model
        config: Configuration dictionary
        device: Device to load model on
        num_chars: Number of characters (for char features)

    Returns:
        Loaded model
    """
    use_char_features = config.get('model', {}).get('use_char_features', False)
    use_attention = config.get('model', {}).get('use_attention', False)

    # Initialize model
    if model_type == 'baseline':
        model = BaselineBiLSTMTagger(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=config['model']['embedding_dim'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=pad_idx
        )
    else:  # bilstm_crf
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=config['model']['embedding_dim'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            pad_idx=pad_idx,
            use_char_features=use_char_features,
            num_chars=num_chars if use_char_features else 0,
            char_embedding_dim=config.get('model', {}).get('char_embedding_dim', 30),
            char_hidden_size=config.get('model', {}).get('char_hidden_size', 50),
            char_kernel_sizes=config.get('model', {}).get('char_kernel_sizes', [2, 3, 4]),
            use_highway=config.get('model', {}).get('use_highway', True),
            use_attention=use_attention,
            attention_heads=config.get('model', {}).get('attention_heads', 4),
            attention_dropout=config.get('model', {}).get('attention_dropout', 0.1)
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Best dev F1 during training: {checkpoint.get('best_f1', 'N/A'):.4f}")
    print(f"Trained for {checkpoint.get('epoch', 'N/A')} epochs")

    return model


def evaluate_model(
    model,
    dataloader,
    device,
    label_vocab,
    word_vocab,
    use_char_features: bool = False,
    save_predictions: bool = False,
    output_file: str = None
):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device
        label_vocab: Label vocabulary
        word_vocab: Word vocabulary
        use_char_features: Whether char features are used
        save_predictions: Whether to save predictions
        output_file: Path to save predictions

    Returns:
        Tuple of (metrics_dict, all_true_tags, all_pred_tags)
    """
    model.eval()

    all_true_tags = []
    all_pred_tags = []
    all_tokens = []

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)

            # Get character IDs if available
            char_ids = None
            if use_char_features and 'char_ids' in batch:
                char_ids = batch['char_ids'].to(device)

            # Get predictions
            pred_tag_ids = model.predict(token_ids, mask, char_ids)

            # Convert to tag strings
            for i, (pred_ids, true_ids, m) in enumerate(zip(pred_tag_ids, label_ids, mask)):
                length = m.sum().item()

                # Decode predictions
                pred_tags = label_vocab.decode(pred_ids)
                true_tags = label_vocab.decode(true_ids[:length].tolist())

                all_pred_tags.append(pred_tags)
                all_true_tags.append(true_tags)

                # Get tokens for this sequence
                tokens = [word_vocab.idx2token.get(idx.item(), '<UNK>')
                          for idx in token_ids[i, :length]]
                all_tokens.append(tokens)

    # Compute metrics
    metrics = compute_metrics(all_true_tags, all_pred_tags)

    # Save predictions if requested
    if save_predictions and output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Token\tTrue\tPred\n")
            f.write("=" * 50 + "\n\n")

            for tokens, true_tags, pred_tags in zip(all_tokens, all_true_tags, all_pred_tags):
                for token, true_tag, pred_tag in zip(tokens, true_tags, pred_tags):
                    marker = "" if true_tag == pred_tag else " <-- ERROR"
                    f.write(f"{token}\t{true_tag}\t{pred_tag}{marker}\n")
                f.write("\n")

        print(f"Predictions saved to {output_file}")

    return metrics, all_true_tags, all_pred_tags


def evaluate(config_path: str, model_type: str = 'bilstm_crf',
             checkpoint_path: str = None):
    """
    Main evaluation function.

    Args:
        config_path: Path to configuration file
        model_type: Type of model
        checkpoint_path: Path to model checkpoint (overrides config)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check for character features
    use_char_features = config.get('model', {}).get('use_char_features', False)

    # Load vocabularies
    print("\nLoading vocabularies...")
    vocab_path = config['artifacts']['vocab_file']
    word_vocab = Vocabulary.load(vocab_path.replace('.pkl', '_word.pkl'))
    label_vocab = LabelVocabulary.load(vocab_path.replace('.pkl', '_label.pkl'))

    char_vocab = None
    if use_char_features:
        char_vocab_path = vocab_path.replace('.pkl', '_char.pkl')
        if os.path.exists(char_vocab_path):
            char_vocab = CharVocabulary.load(char_vocab_path)
            print(f"Character vocabulary loaded: {len(char_vocab)} characters")
        else:
            print("Warning: Character vocabulary not found, disabling char features")
            use_char_features = False

    # Create test dataloader
    print("\nCreating test dataloader...")
    max_word_length = config.get('model', {}).get('max_word_length', 20)

    test_loader = create_dataloader(
        config['data']['test_file'],
        word_vocab,
        label_vocab,
        char_vocab=char_vocab,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        max_seq_length=config['preprocessing']['max_seq_length'],
        max_word_length=max_word_length,
        use_char_features=use_char_features
    )

    # Load model
    print("\nLoading model...")
    if checkpoint_path is None:
        checkpoint_path = config['artifacts']['model_checkpoint']

    model = load_model(
        checkpoint_path,
        len(word_vocab),
        len(label_vocab),
        word_vocab.pad_idx,
        model_type,
        config,
        device,
        num_chars=len(char_vocab) if char_vocab else 0
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)

    metrics, all_true_tags, all_pred_tags = evaluate_model(
        model,
        test_loader,
        device,
        label_vocab,
        word_vocab,
        use_char_features=use_char_features,
        save_predictions=config['evaluation']['output_predictions'],
        output_file=config['evaluation']['predictions_file']
    )

    # Print results
    print_metrics(metrics, prefix="Test")

    # Print detailed classification report with actual predictions
    print("\n" + get_classification_report(all_true_tags, all_pred_tags))

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Model: {model_type.upper()}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    print(f"Test F1:        {metrics['f1']:.4f}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate NER model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='bilstm_crf',
                        choices=['baseline', 'bilstm_crf'],
                        help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')

    args = parser.parse_args()

    evaluate(args.config, args.model, args.checkpoint)


if __name__ == '__main__':
    main()
