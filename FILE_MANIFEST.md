# BiLSTM-CRF NER Project - File Manifest

This document lists all files in the project with descriptions.

## Documentation Files

| File | Lines | Description |
|------|-------|-------------|
| `README.md` | ~350 | Comprehensive project documentation with setup, usage, and examples |
| `PROJECT_SUMMARY.md` | ~450 | Complete implementation summary and overview |
| `QUICK_START.md` | ~200 | 5-minute quick start guide with common commands |
| `FILE_MANIFEST.md` | This file | Complete file listing and manifest |
| `requirements.txt` | ~20 | Python package dependencies |

## Configuration Files

| File | Lines | Description |
|------|-------|-------------|
| `config/config.yaml` | ~80 | Main configuration with full hyperparameters |
| `config/config_test.yaml` | ~80 | Test configuration (3 epochs for quick testing) |

## Data Files

| File | Sentences | Description |
|------|-----------|-------------|
| `data/processed/train.txt` | 280 | Training data in BIO format |
| `data/processed/dev.txt` | 60 | Development/validation data in BIO format |
| `data/processed/test.txt` | 60 | Test data in BIO format |

**Format**: One token per line with tab-separated tag, blank lines between sentences.

## Source Code - Core Modules

### Data Processing (`src/data/`)

| File | Lines | Classes/Functions | Description |
|------|-------|-------------------|-------------|
| `preprocess.py` | ~250 | `create_sample_biomedical_data()`, `preprocess_bc5cdr()` | Creates sample data and preprocesses BC5CDR corpus |
| `dataset.py` | ~250 | `NERDataset`, `collate_fn()`, `create_dataloader()` | PyTorch Dataset for NER with batching and padding |

### Models (`src/models/`)

| File | Lines | Classes | Description |
|------|-------|---------|-------------|
| `layers.py` | ~350 | `CRF`, `CharCNN` | CRF layer with forward-backward algorithm and Viterbi decoding |
| `baseline_tagger.py` | ~180 | `BaselineBiLSTMTagger` | BiLSTM tagger without CRF (baseline model) |
| `bilstm_crf.py` | ~230 | `BiLSTMCRF` | Main BiLSTM-CRF model for NER |

**Key Implementation Details**:
- `CRF`: Full CRF implementation from scratch
  - `forward()`: Computes negative log-likelihood loss
  - `_compute_log_partition()`: Forward algorithm using logsumexp
  - `decode()`: Viterbi algorithm for inference
  - Transition matrix learning
  - Proper masking for variable-length sequences

### Training & Evaluation (`src/training/`)

| File | Lines | Functions | Description |
|------|-------|-----------|-------------|
| `train.py` | ~300 | `train()`, `train_epoch()`, `evaluate()` | Complete training loop with checkpointing and early stopping |
| `eval.py` | ~230 | `evaluate()`, `load_model()` | Evaluation script for test set with prediction saving |

### Utilities (`src/utils/`)

| File | Lines | Classes/Functions | Description |
|------|-------|-------------------|-------------|
| `vocab.py` | ~280 | `Vocabulary`, `LabelVocabulary` | Vocabulary management with save/load functionality |
| `metrics.py` | ~280 | `compute_metrics()`, `get_entities()` | Entity-level NER evaluation metrics (P/R/F1) |
| `logging_utils.py` | ~170 | `TrainingLogger`, `print_model_summary()` | Training logger and formatting utilities |

## Scripts

| File | Lines | Description |
|------|-------|-------------|
| `scripts/run_train.sh` | ~35 | Shell script for training with argument parsing |
| `scripts/run_eval.sh` | ~40 | Shell script for evaluation with argument parsing |
| `verify_structure.py` | ~230 | Project structure verification script |

## Reports & Templates

| File | Lines | Description |
|------|-------|-------------|
| `reports/report_outline.md` | ~550 | Complete academic report template with all sections |
| `reports/experiments.md` | ~330 | Experiment tracking template with example entries |
| `reports/predictions.txt` | Generated | Model predictions in readable format (created after eval) |

## Notebooks

| File | Cells | Description |
|------|-------|-------------|
| `notebooks/exploration.ipynb` | ~15 | Jupyter notebook for EDA, visualization, and analysis |

**Includes**:
- Dataset statistics visualization
- Sentence length distribution
- Label distribution charts
- Sample sentence display
- CRF transition matrix heatmap
- Error analysis tools

## Package Structure

```
src/
├── __init__.py           # Package marker
├── data/
│   ├── __init__.py      # Package marker
│   ├── preprocess.py    # Data preprocessing
│   └── dataset.py       # PyTorch Dataset
├── models/
│   ├── __init__.py      # Package marker
│   ├── layers.py        # CRF and char-CNN layers
│   ├── baseline_tagger.py  # Baseline model
│   └── bilstm_crf.py    # Main model
├── training/
│   ├── __init__.py      # Package marker
│   ├── train.py         # Training loop
│   └── eval.py          # Evaluation
└── utils/
    ├── __init__.py      # Package marker
    ├── vocab.py         # Vocabulary management
    ├── metrics.py       # Evaluation metrics
    └── logging_utils.py # Logging utilities
```

## Generated Files (After Training)

These files are created during training/evaluation:

| File | Description |
|------|-------------|
| `artifacts/vocab_word.pkl` | Word vocabulary (serialized) |
| `artifacts/vocab_label.pkl` | Label vocabulary (serialized) |
| `artifacts/best_model.pt` | Best model checkpoint with state dict |
| `reports/predictions.txt` | Detailed predictions with gold/pred tags |

## Code Statistics

### Total Lines of Code

| Category | Files | Total Lines |
|----------|-------|-------------|
| Models | 3 | ~760 |
| Data Processing | 2 | ~500 |
| Training/Eval | 2 | ~530 |
| Utils | 3 | ~730 |
| Scripts | 3 | ~305 |
| **Total Python Code** | **13** | **~2,825** |
| Documentation | 5 | ~1,550 |
| Config | 2 | ~160 |
| **Total Project** | **20+** | **~4,535** |

### Implementation Complexity

**Most Complex Files**:
1. `src/models/layers.py` (~350 lines) - CRF implementation with forward-backward and Viterbi
2. `src/training/train.py` (~300 lines) - Complete training pipeline
3. `src/utils/vocab.py` (~280 lines) - Vocabulary management
4. `src/utils/metrics.py` (~280 lines) - Entity-level metrics

**Key Algorithms Implemented**:
- CRF Forward Algorithm (log-space dynamic programming)
- Viterbi Decoding (optimal sequence prediction)
- Entity-level F1 computation (span-based evaluation)
- BIO tag sequence processing

## File Dependencies

```
train.py
  ├── models/bilstm_crf.py
  │   ├── models/layers.py (CRF)
  │   └── torch.nn
  ├── data/dataset.py
  │   └── utils/vocab.py
  ├── utils/metrics.py
  └── utils/logging_utils.py

eval.py
  ├── models/bilstm_crf.py
  ├── data/dataset.py
  └── utils/metrics.py
```

## Import Structure

All modules use absolute imports from `src.*`:
```python
from src.utils.vocab import Vocabulary
from src.models.bilstm_crf import BiLSTMCRF
from src.data.dataset import create_dataloader
```

This allows running scripts as modules:
```bash
python -m src.training.train
python -m src.data.preprocess
```

## Version Control Ready

**Git-friendly structure**:
- `.gitignore` patterns included in README
- Artifacts directory for generated files
- Clear separation of code/data/configs
- No hardcoded paths

**Ignore patterns** (add to `.gitignore`):
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.ipynb_checkpoints/
artifacts/*.pt
artifacts/*.pkl
data/raw/*
*.log
```

## Testing & Verification

| File | Purpose |
|------|---------|
| `verify_structure.py` | Verifies all files exist and Python syntax is valid |
| `config/config_test.yaml` | Quick test configuration (3 epochs) |

Run verification:
```bash
python verify_structure.py
```

## Documentation Quality

Each Python file includes:
- ✅ Module-level docstring
- ✅ Class docstrings with Args description
- ✅ Function docstrings with Args/Returns
- ✅ Type hints (where applicable)
- ✅ Inline comments for complex logic
- ✅ Clear variable names

Example:
```python
def compute_metrics(true_tags: List[List[str]],
                    pred_tags: List[List[str]]) -> Dict[str, float]:
    """
    Compute entity-level precision, recall, and F1 score.

    Args:
        true_tags: List of true tag sequences
        pred_tags: List of predicted tag sequences

    Returns:
        Dictionary with overall and per-entity metrics
    """
```

## Maintenance

**Regular Updates Needed**:
- `requirements.txt`: Update package versions
- `config/config.yaml`: Tune hyperparameters
- `reports/experiments.md`: Log new experiments

**No Updates Needed**:
- Core algorithm implementations (CRF, metrics)
- Data preprocessing logic
- Model architectures

---

**Total Project Size**: ~4,500+ lines of code and documentation

**Estimated Development Time**: ~16-20 hours for complete implementation

**Code Quality**: Production-ready with educational focus

**Documentation**: Comprehensive with examples and templates

**Last Updated**: 2025-11-09
