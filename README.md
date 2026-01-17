# BiLSTM-CRF Named Entity Recognition for Biomedical Text

A **production-quality, state-of-the-art** implementation of **BiLSTM-CRF** for **Named Entity Recognition (NER)** in biomedical text. Built entirely **from scratch** using PyTorch, this project targets the BC5CDR dataset to recognize Chemical and Disease entities.

## Highlights

- **From-Scratch Implementation**: All components (CRF, attention, highway networks) implemented using PyTorch primitives
- **State-of-the-Art Architecture**: BiLSTM + CRF + Character CNN + Self-Attention + GloVe embeddings
- **Production Quality**: Comprehensive testing, error analysis, visualization, and ablation studies
- **Expected Performance**: ~87% F1 score on BC5CDR dataset

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BiLSTM-CRF-Attention Model                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌──────────────┐   ┌──────────────┐                                 │
│    │ Word Tokens  │   │  Characters  │                                 │
│    └──────┬───────┘   └──────┬───────┘                                 │
│           │                  │                                          │
│           ▼                  ▼                                          │
│    ┌──────────────┐   ┌──────────────────────┐                         │
│    │   GloVe      │   │ Character CNN        │                         │
│    │ Embeddings   │   │ (Multi-kernel 2,3,4) │                         │
│    │   (100d)     │   │       ▼              │                         │
│    └──────┬───────┘   │ Highway Networks     │                         │
│           │           │       ▼              │                         │
│           │           │  Char Features (50d) │                         │
│           │           └──────────┬───────────┘                         │
│           │                      │                                      │
│           └──────────┬───────────┘                                      │
│                      │ Concatenate                                      │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │  Bidirectional LSTM  │                                      │
│           │    (2 layers, 256)   │                                      │
│           └──────────┬───────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │   Self-Attention     │                                      │
│           │   (4 heads + LayerN) │                                      │
│           └──────────┬───────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │    Linear Layer      │                                      │
│           │  (hidden → num_tags) │                                      │
│           └──────────┬───────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │      CRF Layer       │                                      │
│           │  (Viterbi Decoding)  │                                      │
│           └──────────┬───────────┘                                      │
│                      │                                                  │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │   BIO Tag Sequence   │                                      │
│           │  O B-Chem I-Chem O   │                                      │
│           └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Model Components (All Built From Scratch)

| Component | Implementation | Description |
|-----------|----------------|-------------|
| **CRF Layer** | `src/models/layers.py` | Forward-backward algorithm, Viterbi decoding |
| **Character CNN** | `src/models/layers.py` | Multi-kernel (2,3,4) with highway networks |
| **Self-Attention** | `src/models/attention.py` | Multi-head attention with layer normalization |
| **Highway Networks** | `src/models/layers.py` | Gradient-friendly deep networks |
| **GloVe Loader** | `src/utils/embeddings.py` | Pre-trained embedding integration |

### Training Enhancements

- Learning rate warmup with linear decay
- Class weight balancing for imbalanced labels
- TensorBoard logging
- Comprehensive checkpointing with resume support
- Early stopping

### Analysis Tools

- Detailed error analysis (boundary, type, missed, spurious)
- Per-entity-type metrics
- Performance by entity length
- Confusion matrix visualization
- HTML prediction visualization

---

## Project Structure

```
.
├── config/
│   └── config.yaml              # Complete configuration
├── data/
│   ├── raw/                     # BC5CDR PubTator files
│   ├── processed/               # BIO-formatted data
│   └── embeddings/              # GloVe embeddings
├── src/
│   ├── data/
│   │   ├── bc5cdr_parser.py     # BC5CDR dataset parser
│   │   ├── preprocess.py        # Data preprocessing
│   │   └── dataset.py           # PyTorch Dataset/DataLoader
│   ├── models/
│   │   ├── bilstm_crf.py        # Main BiLSTM-CRF model
│   │   ├── layers.py            # CRF, CharCNN, Highway
│   │   ├── attention.py         # Self-attention mechanism
│   │   └── baseline_tagger.py   # Baseline model
│   ├── training/
│   │   ├── train.py             # Training pipeline
│   │   └── eval.py              # Evaluation script
│   └── utils/
│       ├── vocab.py             # Word, Label, Char vocabularies
│       ├── metrics.py           # Entity-level metrics
│       ├── embeddings.py        # GloVe loader
│       ├── analysis.py          # Error analysis tools
│       ├── visualization.py     # Plotting utilities
│       └── logging_utils.py     # Logging utilities
├── scripts/
│   ├── ablation_study.py        # Ablation experiments
│   ├── run_train.sh             # Training script
│   └── run_eval.sh              # Evaluation script
├── tests/
│   ├── test_crf.py              # CRF unit tests
│   ├── test_model.py            # Model unit tests
│   ├── test_vocab.py            # Vocabulary tests
│   └── test_metrics.py          # Metrics tests
├── artifacts/                    # Saved models and vocabularies
├── reports/                      # Outputs and visualizations
└── requirements.txt              # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd NLP_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download GloVe Embeddings (Optional but Recommended)

```bash
# Download GloVe 100d embeddings
python -m src.utils.embeddings --output-dir data/embeddings --dim 100 --corpus 6B
```

---

## Quick Start

### 1. Prepare Data

```bash
# Process BC5CDR dataset (downloads if not present)
python -m src.data.preprocess
```

### 2. Train Full Model

```bash
# Train with all features (char CNN, attention, GloVe)
python -m src.training.train --config config/config.yaml --model bilstm_crf
```

### 3. Evaluate

```bash
# Evaluate on test set
python -m src.training.eval --config config/config.yaml --model bilstm_crf
```

### 4. Interactive Demo

```bash
# Run command-line demo
python demo.py

# Or run web demo (requires: pip install gradio)
python app.py
# Then open http://localhost:7860 in your browser
```

### 5. Run Ablation Study

```bash
# Compare different model configurations
python scripts/ablation_study.py --config config/config.yaml --output results/ablation
```

---

## Training Options

### Model Types

```bash
# Baseline BiLSTM (no CRF)
python -m src.training.train --model baseline_bilstm

# BiLSTM-CRF (default)
python -m src.training.train --model bilstm_crf
```

### Resume Training

```bash
python -m src.training.train --resume artifacts/best_model.pt
```

### Configuration Options

Key options in `config/config.yaml`:

```yaml
model:
  use_char_features: true     # Enable character CNN
  use_attention: true         # Enable self-attention
  use_pretrained_embeddings: true  # Use GloVe

training:
  use_warmup: true            # Enable LR warmup
  warmup_epochs: 2
  use_class_weights: false    # Balance imbalanced classes
```

---

## Expected Results

### Ablation Study Results

| Model Configuration | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| BiLSTM only | ~82% | ~78% | ~80% |
| BiLSTM + CRF | ~85% | ~83% | ~84% |
| BiLSTM + CRF + CharCNN | ~87% | ~85% | ~86% |
| BiLSTM + CRF + Attention | ~86% | ~85% | ~85.5% |
| Full Model (no pretrained) | ~87% | ~86% | ~86.5% |
| **Full Model + GloVe** | **~88%** | **~86%** | **~87%** |

### Component Contributions

- **CRF Layer**: +3-4% F1 (enforces valid tag sequences)
- **Character CNN**: +2% F1 (captures morphological patterns)
- **Self-Attention**: +1-2% F1 (models long-range dependencies)
- **GloVe Embeddings**: +0.5-1% F1 (better word representations)

---

## Code Quality

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_crf.py -v
```

### Test Coverage

- CRF forward pass and Viterbi decoding
- Model initialization and forward pass
- Vocabulary encoding/decoding
- Metrics computation

---

## Analysis and Visualization

### Error Analysis

```python
from src.utils.analysis import analyze_errors, print_error_analysis

analysis = analyze_errors(tokens_list, true_tags, pred_tags)
print_error_analysis(analysis)
```

Output includes:
- Boundary errors (wrong entity span)
- Type errors (wrong entity type)
- Missed entities
- Spurious predictions

### Visualization

```python
from src.utils.visualization import plot_training_curves, create_all_visualizations

# Plot training progress
plot_training_curves(history, 'training_curves.png')

# Generate all visualizations
create_all_visualizations(history, error_analysis, ...)
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir artifacts/tensorboard
```

---

## Interactive Demo

The project includes two demo interfaces for testing the trained model:

### Command-Line Demo

```bash
python demo.py
```

**Features:**
- Interactive text input
- Color-coded entity highlighting (blue for Chemical, red for Disease)
- Entity list with types
- Example sentences included

**Example session:**
```
=== BiLSTM-CRF NER Demo ===
Enter text: Aspirin can cause gastrointestinal bleeding.

Results:
[Aspirin] can cause [gastrointestinal bleeding].

Entities found:
  - Aspirin (Chemical)
  - gastrointestinal bleeding (Disease)
```

**Single-text mode:**
```bash
python demo.py --text "Metformin treats diabetes"
```

### Web Demo (Gradio)

```bash
# Install Gradio first
pip install gradio

# Run web demo
python app.py
```

Then open **http://localhost:7860** in your browser.

**Features:**
- Modern web interface
- Visual entity highlighting with colors
- Clickable example sentences
- Entity table output

---

## API Reference

### BiLSTMCRF Model

```python
from src.models.bilstm_crf import BiLSTMCRF

model = BiLSTMCRF(
    vocab_size=10000,
    num_tags=6,
    embedding_dim=100,
    hidden_size=256,
    num_layers=2,
    dropout=0.5,
    pad_idx=0,
    pretrained_embeddings=None,     # Optional GloVe tensor
    freeze_embeddings=False,
    use_char_features=True,
    num_chars=100,
    char_embedding_dim=30,
    char_hidden_size=50,
    char_kernel_sizes=[2, 3, 4],
    use_highway=True,
    use_attention=True,
    attention_heads=4,
    attention_dropout=0.1
)

# Training
loss = model.loss(token_ids, label_ids, mask, char_ids)

# Inference
predictions = model.predict(token_ids, mask, char_ids)
```

### CRF Layer

```python
from src.models.layers import CRF

crf = CRF(num_tags=6, pad_idx=0)

# Compute negative log-likelihood
loss = crf(emissions, tags, mask)

# Viterbi decoding
best_paths = crf.decode(emissions, mask)
```

---

## References

### Papers

1. **Neural Architectures for Named Entity Recognition**
   Lample et al. (2016) - BiLSTM-CRF architecture

2. **End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF**
   Ma & Hovy (2016) - Character CNN + Highway networks

3. **Attention Is All You Need**
   Vaswani et al. (2017) - Multi-head attention

4. **BioCreative V CDR Task Corpus**
   Li et al. (2016) - BC5CDR dataset

### Resources

- [BC5CDR Dataset](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Troubleshooting

### Out of Memory

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Default: 32
```

### Slow Training

```yaml
# Disable optional features for faster training
model:
  use_char_features: false
  use_attention: false
```

### Poor Performance

1. Ensure GloVe embeddings are loaded correctly
2. Try increasing `num_epochs` to 50
3. Use learning rate warmup
4. Check data preprocessing for errors

---

## Contributing

This is an educational project for NLP coursework. Contributions welcome via:
1. Bug reports and feature requests
2. Code improvements and optimizations
3. Documentation enhancements
4. Additional test cases

---

## License

This project is for educational purposes as part of an NLP course project.

---

## Acknowledgments

- BC5CDR dataset creators at NCBI
- PyTorch team for the deep learning framework
- GloVe team at Stanford NLP
- Course instructors and teaching materials

---

**Author**: NLP Course Project
**Built From Scratch**: All neural network components implemented using PyTorch primitives
