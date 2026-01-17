# Experiment Log: BiLSTM-CRF NER

This document tracks all experiments conducted for the BiLSTM-CRF NER project.

## Experiment Template

For each experiment, record:
- **Experiment ID**: Unique identifier
- **Date**: When the experiment was run
- **Model**: baseline or bilstm_crf
- **Configuration**: Key hyperparameters
- **Results**: Precision, Recall, F1 on dev and test sets
- **Notes**: Observations, issues, insights

---

## Experiment 1: Baseline BiLSTM Tagger

**Date**: [To be filled]

**Model**: Baseline BiLSTM (no CRF)

**Configuration**:
- Embedding dimension: 100
- Hidden size: 256
- Number of layers: 2
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 32
- Epochs: 30
- Optimizer: Adam

**Results**:

*Development Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

*Test Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

**Per-Entity Results**:

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| Chemical | TBD | TBD | TBD |
| Disease | TBD | TBD | TBD |

**Training Details**:
- Total training time: TBD
- Best epoch: TBD
- Convergence: TBD

**Notes**:
- First baseline experiment
- Model makes independent predictions per token
- No constraints on label sequences
- Observations: [Add your observations here]

---

## Experiment 2: BiLSTM-CRF (Main Model)

**Date**: [To be filled]

**Model**: BiLSTM-CRF

**Configuration**:
- Embedding dimension: 100
- Hidden size: 256
- Number of layers: 2
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 32
- Epochs: 30
- Optimizer: Adam
- CRF: Enabled

**Results**:

*Development Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

*Test Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

**Per-Entity Results**:

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| Chemical | TBD | TBD | TBD |
| Disease | TBD | TBD | TBD |

**Training Details**:
- Total training time: TBD
- Best epoch: TBD
- Convergence: TBD

**Comparison with Baseline**:
- F1 improvement: TBD
- Precision improvement: TBD
- Recall improvement: TBD

**Notes**:
- Main model with CRF layer
- Should show improvement over baseline
- CRF enforces valid tag transitions
- Observations: [Add your observations here]

---

## Experiment 3: Hyperparameter Tuning - Hidden Size

**Date**: [To be filled]

**Model**: BiLSTM-CRF

**Variants Tested**:

### 3a: Hidden Size = 128
- Configuration: Same as Exp 2, but hidden_size=128
- Dev F1: TBD
- Test F1: TBD

### 3b: Hidden Size = 256 (baseline)
- Configuration: Same as Exp 2
- Dev F1: TBD
- Test F1: TBD

### 3c: Hidden Size = 512
- Configuration: Same as Exp 2, but hidden_size=512
- Dev F1: TBD
- Test F1: TBD

**Best Configuration**: TBD

**Notes**:
- Exploring impact of model capacity
- Larger hidden size may capture more context
- Trade-off with training time and memory
- Observations: [Add your observations here]

---

## Experiment 4: Hyperparameter Tuning - Learning Rate

**Date**: [To be filled]

**Model**: BiLSTM-CRF

**Variants Tested**:

### 4a: LR = 0.0001
- Configuration: Best config from Exp 3, but lr=0.0001
- Dev F1: TBD
- Test F1: TBD
- Training time: TBD

### 4b: LR = 0.001 (baseline)
- Configuration: Best config from Exp 3
- Dev F1: TBD
- Test F1: TBD
- Training time: TBD

### 4c: LR = 0.01
- Configuration: Best config from Exp 3, but lr=0.01
- Dev F1: TBD
- Test F1: TBD
- Training time: TBD

**Best Configuration**: TBD

**Notes**:
- Learning rate significantly affects convergence
- Too high: unstable training
- Too low: slow convergence
- Observations: [Add your observations here]

---

## Experiment 5: Optional Enhancement - Character Features

**Date**: [To be filled]

**Model**: BiLSTM-CRF with Character-level CNN

**Configuration**:
- Base configuration: Best from previous experiments
- Character embedding dim: 30
- Character hidden size: 50
- Use char features: True

**Results**:

*Test Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

**Comparison with Base Model**:
- F1 improvement: TBD

**Notes**:
- Character features capture morphology
- Especially useful for out-of-vocabulary words
- Additional computational cost
- Observations: [Add your observations here]

---

## Experiment 6: Pretrained Embeddings

**Date**: [To be filled]

**Model**: BiLSTM-CRF with pretrained embeddings

**Configuration**:
- Base configuration: Best from previous experiments
- Pretrained embeddings: [Source, e.g., Word2Vec, GloVe, BioWordVec]
- Fine-tuning: Yes/No

**Results**:

*Test Set*:
- Precision: TBD
- Recall: TBD
- F1 Score: TBD

**Comparison with Random Initialization**:
- F1 improvement: TBD

**Notes**:
- Pretrained embeddings provide better initialization
- Domain-specific embeddings (BioWordVec) may help more
- Observations: [Add your observations here]

---

## Summary Table

| Exp ID | Model | Key Change | Dev F1 | Test F1 | Notes |
|--------|-------|------------|--------|---------|-------|
| 1 | Baseline | No CRF | TBD | TBD | Baseline |
| 2 | BiLSTM-CRF | + CRF | TBD | TBD | Main model |
| 3a | BiLSTM-CRF | hidden=128 | TBD | TBD | |
| 3b | BiLSTM-CRF | hidden=256 | TBD | TBD | |
| 3c | BiLSTM-CRF | hidden=512 | TBD | TBD | |
| 4a | BiLSTM-CRF | lr=0.0001 | TBD | TBD | |
| 4b | BiLSTM-CRF | lr=0.001 | TBD | TBD | |
| 4c | BiLSTM-CRF | lr=0.01 | TBD | TBD | |
| 5 | BiLSTM-CRF | + char features | TBD | TBD | |
| 6 | BiLSTM-CRF | + pretrained emb | TBD | TBD | |

---

## Best Model

**Final Best Configuration**:
- Model: TBD
- Configuration: TBD
- Dev F1: TBD
- Test F1: TBD

**Model Saved At**: `artifacts/best_model.pt`

---

## Key Insights

1. **CRF Contribution**:
   - TBD

2. **Hyperparameter Sensitivity**:
   - TBD

3. **Character Features**:
   - TBD

4. **Pretrained Embeddings**:
   - TBD

5. **Error Patterns**:
   - TBD

---

## Future Experiments to Try

- [ ] Different batch sizes (16, 64)
- [ ] Different number of LSTM layers (1, 3)
- [ ] Different dropout rates (0.3, 0.7)
- [ ] Data augmentation techniques
- [ ] Ensemble methods
- [ ] Different optimization algorithms (SGD, AdamW)
- [ ] Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
- [ ] Label smoothing
- [ ] Focal loss for imbalanced classes
- [ ] Multi-task learning
- [ ] Self-attention layers
- [ ] ELMo or BERT contextualized embeddings

---

## Reproducibility Checklist

- [ ] Random seeds set (random, numpy, torch)
- [ ] Configuration files saved
- [ ] Model checkpoints saved
- [ ] Vocabulary files saved
- [ ] Predictions saved for analysis
- [ ] Environment documented (requirements.txt)
- [ ] Command history recorded
- [ ] Git commits for each experiment

---

*Last Updated*: [Date]
