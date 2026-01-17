#!/usr/bin/env python3
"""
Gradio Web Demo for BiLSTM-CRF NER Model

This script provides a web-based interface for testing the trained
NER model on custom biomedical text. Users can input sentences and see
extracted Chemical and Disease entities with visual highlighting.

Usage:
    python app.py

Then open http://localhost:7860 in your browser.

Requirements:
    pip install gradio

Author: Yasser Hamdan & Hassan Najdi
Course: NLP Course Project
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio is not installed.")
    print("Please install it with: pip install gradio")
    sys.exit(1)

import yaml
import torch
from typing import List, Tuple, Dict

from src.utils.vocab import Vocabulary, LabelVocabulary, CharVocabulary
from src.models.bilstm_crf import BiLSTMCRF


# Global model instance (loaded once)
MODEL = None
WORD_VOCAB = None
LABEL_VOCAB = None
CHAR_VOCAB = None
CONFIG = None
DEVICE = None


def load_model():
    """Load model and vocabularies (cached globally)."""
    global MODEL, WORD_VOCAB, LABEL_VOCAB, CHAR_VOCAB, CONFIG, DEVICE

    if MODEL is not None:
        return

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        CONFIG = yaml.safe_load(f)

    # Load vocabularies
    vocab_dir = 'artifacts'
    WORD_VOCAB = Vocabulary.load(os.path.join(vocab_dir, 'vocab_word.pkl'))
    LABEL_VOCAB = LabelVocabulary.load(os.path.join(vocab_dir, 'vocab_label.pkl'))

    # Check for character vocabulary
    use_char_features = CONFIG.get('model', {}).get('use_char_features', False)
    if use_char_features:
        char_vocab_path = os.path.join(vocab_dir, 'vocab_char.pkl')
        if os.path.exists(char_vocab_path):
            CHAR_VOCAB = CharVocabulary.load(char_vocab_path)
        else:
            use_char_features = False

    # Load model
    model_path = 'artifacts/best_model.pt'

    MODEL = BiLSTMCRF(
        vocab_size=len(WORD_VOCAB),
        num_tags=len(LABEL_VOCAB),
        embedding_dim=CONFIG['model']['embedding_dim'],
        hidden_size=CONFIG['model']['hidden_size'],
        num_layers=CONFIG['model']['num_layers'],
        dropout=CONFIG['model']['dropout'],
        pad_idx=WORD_VOCAB.pad_idx,
        use_char_features=use_char_features,
        num_chars=len(CHAR_VOCAB) if CHAR_VOCAB else 0,
        char_embedding_dim=CONFIG.get('model', {}).get('char_embedding_dim', 30),
        char_hidden_size=CONFIG.get('model', {}).get('char_hidden_size', 50),
        char_kernel_sizes=CONFIG.get('model', {}).get('char_kernel_sizes', [2, 3, 4]),
        use_highway=CONFIG.get('model', {}).get('use_highway', True),
        use_attention=CONFIG.get('model', {}).get('use_attention', False),
        attention_heads=CONFIG.get('model', {}).get('attention_heads', 4),
        attention_dropout=CONFIG.get('model', {}).get('attention_dropout', 0.1)
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Model loaded on {DEVICE}")


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for biomedical text."""
    tokens = text.split()
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
    """Extract entities from BIO tags."""
    entities = []
    current_entity = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            current_entity = {
                'text': token,
                'type': entity_type,
                'start': i,
                'end': i
            }
        elif tag.startswith('I-') and current_entity:
            entity_type = tag[2:]
            if entity_type == current_entity['type']:
                current_entity['text'] += ' ' + token
                current_entity['end'] = i
            else:
                entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': entity_type,
                    'start': i,
                    'end': i
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def predict_ner(text: str) -> Tuple[str, List[List]]:
    """
    Predict NER entities in text.

    Args:
        text: Input text

    Returns:
        Tuple of (highlighted_html, entities_table)
    """
    # Load model if not loaded
    load_model()

    if not text.strip():
        return "<p>Please enter some text.</p>", []

    # Tokenize
    tokens = tokenize(text)

    if not tokens:
        return "<p>No tokens found.</p>", []

    # Encode tokens
    token_ids = WORD_VOCAB.encode(tokens)
    token_tensor = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    mask = torch.ones(1, len(tokens), dtype=torch.bool, device=DEVICE)

    # Encode characters if needed
    char_tensor = None
    use_char_features = CONFIG.get('model', {}).get('use_char_features', False) and CHAR_VOCAB is not None
    if use_char_features:
        max_word_len = CONFIG.get('model', {}).get('max_word_length', 20)
        char_ids = CHAR_VOCAB.encode_sequence(tokens, max_word_len)
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=DEVICE)

    # Predict
    with torch.no_grad():
        pred_ids = MODEL.predict(token_tensor, mask, char_tensor)

    # Decode tags
    tags = LABEL_VOCAB.decode(pred_ids[0])

    # Extract entities
    entities = extract_entities(tokens, tags)

    # Generate highlighted HTML
    html_parts = []
    i = 0

    while i < len(tokens):
        tag = tags[i]

        if tag.startswith('B-'):
            entity_type = tag[2:]
            entity_tokens = [tokens[i]]

            j = i + 1
            while j < len(tags) and tags[j].startswith('I-'):
                entity_tokens.append(tokens[j])
                j += 1

            entity_text = ' '.join(entity_tokens)

            # Style based on entity type
            if entity_type == 'Chemical':
                color = '#3b82f6'  # Blue
                bg_color = '#dbeafe'
            elif entity_type == 'Disease':
                color = '#ef4444'  # Red
                bg_color = '#fee2e2'
            else:
                color = '#6b7280'
                bg_color = '#f3f4f6'

            html_parts.append(
                f'<span style="background-color: {bg_color}; color: {color}; '
                f'padding: 2px 6px; border-radius: 4px; font-weight: bold; '
                f'border: 1px solid {color};" title="{entity_type}">'
                f'{entity_text}</span>'
            )
            i = j
        else:
            html_parts.append(tokens[i])
            i += 1

    highlighted_html = f'<p style="font-size: 16px; line-height: 2;">{" ".join(html_parts)}</p>'

    # Add legend
    legend = '''
    <div style="margin-top: 20px; padding: 10px; background: #f9fafb; border-radius: 8px;">
        <strong>Legend:</strong>
        <span style="background-color: #dbeafe; color: #3b82f6; padding: 2px 8px; border-radius: 4px; margin-left: 10px; border: 1px solid #3b82f6;">Chemical</span>
        <span style="background-color: #fee2e2; color: #ef4444; padding: 2px 8px; border-radius: 4px; margin-left: 10px; border: 1px solid #ef4444;">Disease</span>
    </div>
    '''

    full_html = highlighted_html + legend

    # Create entities table
    entities_table = [[ent['text'], ent['type']] for ent in entities]

    return full_html, entities_table


def create_demo():
    """Create Gradio demo interface."""

    # Example sentences
    examples = [
        "Aspirin can cause gastrointestinal bleeding and ulcers.",
        "Metformin is commonly used to treat type 2 diabetes mellitus.",
        "Ibuprofen may lead to acute kidney injury in some patients.",
        "Chemotherapy drugs like doxorubicin can cause cardiotoxicity.",
        "Penicillin allergies may result in anaphylaxis and skin rashes.",
        "Acetaminophen overdose can cause severe hepatotoxicity.",
        "Statins are prescribed for hypercholesterolemia but may cause myopathy.",
    ]

    # Create interface
    with gr.Blocks(
        title="Biomedical NER Demo",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # Biomedical Named Entity Recognition Demo

        This demo uses a **BiLSTM-CRF** model trained on the **BC5CDR** dataset to extract
        **Chemical** and **Disease** entities from biomedical text.

        ### Model Architecture
        - BiLSTM encoder with 2 layers (256 hidden units)
        - Character-level CNN with highway networks
        - CRF decoder for sequence labeling

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter Biomedical Text",
                    placeholder="Type a sentence about chemicals, drugs, or diseases...",
                    lines=3
                )
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Quick Examples")
                gr.Markdown("Click any example to try it:")

        with gr.Row():
            highlighted_output = gr.HTML(label="Highlighted Text")

        with gr.Row():
            entities_output = gr.Dataframe(
                headers=["Entity", "Type"],
                label="Extracted Entities",
                wrap=True
            )

        # Examples
        gr.Examples(
            examples=examples,
            inputs=text_input,
            outputs=[highlighted_output, entities_output],
            fn=predict_ner,
            cache_examples=False
        )

        # Connect events
        submit_btn.click(
            fn=predict_ner,
            inputs=text_input,
            outputs=[highlighted_output, entities_output]
        )

        text_input.submit(
            fn=predict_ner,
            inputs=text_input,
            outputs=[highlighted_output, entities_output]
        )

        gr.Markdown("""
        ---
        ### About

        **Authors:** Yasser Hamdan & Hassan Najdi

        **Course:** NLP Course Project

        **Dataset:** BC5CDR (BioCreative V Chemical Disease Relation)

        **Entity Types:**
        - **Chemical:** Drugs, compounds, and chemical substances
        - **Disease:** Medical conditions, symptoms, and diseases
        """)

    return demo


def main():
    """Main entry point."""
    # Check if model exists
    if not os.path.exists('artifacts/best_model.pt'):
        print("Error: Model not found at artifacts/best_model.pt")
        print("Please train the model first using: python -m src.training.train")
        sys.exit(1)

    # Pre-load model
    print("Loading model...")
    load_model()

    # Create and launch demo
    demo = create_demo()
    print("\nLaunching Gradio demo...")
    print("Open http://localhost:7860 in your browser\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == '__main__':
    main()
