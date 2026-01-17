# BiLSTM-CRF NER - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install torch numpy pandas scikit-learn PyYAML tqdm
```

### Step 2: Generate Sample Data
```bash
python -m src.data.preprocess
```
This creates 400 sample biomedical sentences in `data/processed/`

### Step 3: Train the Model
```bash
# Quick test (3 epochs, ~2-3 minutes)
python -m src.training.train --config config/config_test.yaml --model bilstm_crf

# Full training (30 epochs, ~10 minutes)
python -m src.training.train --config config/config.yaml --model bilstm_crf
```

### Step 4: Evaluate
```bash
python -m src.training.eval --config config/config.yaml --model bilstm_crf
```

### Step 5: Explore Results
```bash
# View predictions
cat reports/predictions.txt | head -50

# Check metrics
# (Printed during evaluation)
```

---

## 📋 Common Commands

### Verify Project Structure
```bash
python verify_structure.py
```

### Train Baseline Model
```bash
python -m src.training.train --config config/config.yaml --model baseline
```

### Train BiLSTM-CRF with Different Config
```bash
python -m src.training.train --config config/config_test.yaml --model bilstm_crf
```

### Evaluate with Custom Checkpoint
```bash
python -m src.training.eval --config config/config.yaml --model bilstm_crf --checkpoint artifacts/best_model.pt
```

### Using Shell Scripts
```bash
# Train
bash scripts/run_train.sh

# Train baseline
bash scripts/run_train.sh --baseline

# Evaluate
bash scripts/run_eval.sh
```

---

## 🎯 Expected Results

After training, you should see:

**Training Output**:
```
Epoch 1 | Batch 5/9 | Loss: 2.1543 | Time: 0.15s
...
Dev Precision:   0.8234
Dev Recall:      0.7891
Dev F1:          0.8058
🎉 New best F1 score! (0.8058)
```

**Test Evaluation**:
```
Test Precision: 0.8567
Test Recall:    0.8423
Test F1:        0.8494

Per-Entity Performance:
Chemical    0.8621    0.8534    0.8577
Disease     0.8512    0.8311    0.8410
```

---

## 🔧 Configuration

Edit `config/config.yaml` to change:

**Model Size**:
```yaml
model:
  embedding_dim: 100    # Try: 50, 100, 200
  hidden_size: 256      # Try: 128, 256, 512
  num_layers: 2         # Try: 1, 2, 3
```

**Training**:
```yaml
training:
  batch_size: 32        # Try: 16, 32, 64
  learning_rate: 0.001  # Try: 0.0001, 0.001, 0.01
  num_epochs: 30        # Try: 10, 30, 50
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `README.md` | Full documentation |
| `PROJECT_SUMMARY.md` | Complete implementation details |
| `config/config.yaml` | Main configuration |
| `src/models/bilstm_crf.py` | Main model |
| `src/models/layers.py` | CRF implementation |
| `src/training/train.py` | Training script |
| `reports/report_outline.md` | Academic report template |

---

## ❓ Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution:
pip install torch numpy pandas scikit-learn PyYAML tqdm
```

**Problem**: `FileNotFoundError: data/processed/train.txt`
```bash
# Solution:
python -m src.data.preprocess
```

**Problem**: Training is slow
```bash
# Solution: Use test config for quick verification
python -m src.training.train --config config/config_test.yaml --model bilstm_crf

# Or reduce batch size / epochs in config.yaml
```

**Problem**: Out of memory
```bash
# Solution: Edit config.yaml
# Reduce batch_size: 16 or 8
# Reduce hidden_size: 128
```

---

## 🎓 For Your NLP Course Report

1. **Run experiments**:
   ```bash
   # Baseline
   python -m src.training.train --model baseline

   # Main model
   python -m src.training.train --model bilstm_crf
   ```

2. **Record results** in `reports/experiments.md`

3. **Fill in** `reports/report_outline.md` with:
   - Your experimental results
   - Analysis of model performance
   - Comparison of baseline vs. BiLSTM-CRF
   - Error analysis from predictions.txt

4. **Generate figures** using `notebooks/exploration.ipynb`:
   - Sentence length distribution
   - Label distribution
   - CRF transition matrix
   - Learning curves

5. **Write report** using the outline as a template

---

## 🎉 Success Checklist

- [ ] Dependencies installed
- [ ] Data preprocessed (train/dev/test.txt exist)
- [ ] Baseline model trained
- [ ] BiLSTM-CRF model trained
- [ ] Models evaluated on test set
- [ ] BiLSTM-CRF outperforms baseline
- [ ] Experiments logged
- [ ] Predictions saved
- [ ] Ready to write report!

---

## 📚 Next Steps

**For a better grade**:
1. Try pretrained embeddings (Word2Vec/GloVe)
2. Enable character features (`use_char_features: true`)
3. Tune hyperparameters systematically
4. Add error analysis section with examples
5. Create visualizations in the notebook

**For deeper understanding**:
1. Read the CRF layer implementation (`src/models/layers.py`)
2. Understand the Viterbi algorithm
3. Compare transition matrices before/after training
4. Analyze what errors the CRF fixes vs. baseline

---

**Time Budget**:
- Setup & data prep: 5 minutes
- Training (quick test): 3 minutes
- Training (full): 10-15 minutes
- Evaluation: 1 minute
- Analysis: 15-30 minutes
- Report writing: 2-3 hours

**Total**: ~3-4 hours for complete project

---

Good luck with your NLP course project! 🎓
