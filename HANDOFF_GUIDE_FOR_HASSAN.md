# BiLSTM-CRF Named Entity Recognition Project
## Handoff Guide for Hassan Najdi

---

**From:** Yasser Hamdan
**Course:** NLP Course Project
**Date:** January 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Is Claude Code](#2-what-is-claude-code)
3. [Setting Up The Project](#3-setting-up-the-project)
4. [Running The Project](#4-running-the-project)
5. [Using Claude Code](#5-using-claude-code-with-this-project)
6. [Suggested Improvements](#6-suggested-improvements)
7. [Project Structure](#7-project-structure-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Quick Start Checklist](#9-quick-start-checklist)

---

## 1. Project Overview

### What Is This Project?

This is a **Named Entity Recognition (NER)** system for biomedical text. It automatically identifies and extracts:

- 💊 **Chemical entities** - drugs, compounds, chemical substances
- 🏥 **Disease entities** - medical conditions, symptoms, diseases

**Example Input:**
```
Aspirin can cause gastrointestinal bleeding and liver damage.
```

**Example Output:**
```
[Aspirin]                    -> Chemical
[gastrointestinal bleeding]  -> Disease
[liver damage]               -> Disease
```

### Technical Architecture

The model uses a **BiLSTM-CRF** architecture with advanced features:

| Component | Purpose | File |
|-----------|---------|------|
| BiLSTM | Captures context in both directions | `bilstm_crf.py` |
| CRF Layer | Ensures valid tag sequences | `layers.py` |
| Char CNN | Captures word morphology | `layers.py` |
| Self-Attention | Models long-range dependencies | `attention.py` |
| GloVe Embeddings | Pre-trained word representations | `embeddings.py` |

> **Note:** All components are implemented **from scratch** using PyTorch - no external NER libraries!

### Current Project Status

| Feature | Status | Notes |
|---------|--------|-------|
| BiLSTM-CRF Model | ✅ Complete | Fully implemented |
| CRF from scratch | ✅ Complete | Viterbi + Forward algorithm |
| Character CNN | ✅ Complete | Multi-kernel + Highway |
| Self-Attention | ✅ Complete | 4-head attention |
| Training Pipeline | ✅ Complete | Warmup, checkpoints |
| Evaluation | ✅ Complete | P/R/F1 metrics |
| Interactive Demo | ✅ Complete | CLI + Web (Gradio) |
| Unit Tests | ✅ Complete | 37+ tests passing |
| Documentation | ✅ Complete | README + docstrings |
| **Overall** | **100%** | **Ready for improvements** |

---

## 2. What Is Claude Code?

### Introduction

**Claude Code** is an AI-powered coding assistant that runs in your terminal. It can:

- Read and understand your entire codebase
- Write, edit, and refactor code
- Run terminal commands
- Debug issues
- Answer questions about the code
- Make improvements based on your requests

> Claude Code is like having an expert developer who understands your entire project and can make changes instantly. You just describe what you want in plain English!

### Installing Claude Code

#### Step 1: Install Node.js

Download and install from: https://nodejs.org/

Choose the **LTS (Long Term Support)** version.

#### Step 2: Install Claude Code

Open your terminal and run:

```bash
npm install -g @anthropic-ai/claude-code
```

#### Step 3: Authenticate

Run Claude Code for the first time:

```bash
claude
```

It will open a browser window for you to log in with your Anthropic account.

> ⚠️ You need an Anthropic account with API access. Sign up at https://console.anthropic.com/

#### Step 4: Verify Installation

After authentication, you should see:

```
Welcome to Claude Code!
Type your message or use /help for commands.
>
```

---

## 3. Setting Up The Project

### Step 1: Extract the ZIP File

Extract the project ZIP file to a location like:
```
C:\Users\Hassan\Projects\NLP_Project
```

### Step 2: Open Terminal in Project Directory

```bash
cd C:\Users\Hassan\Projects\NLP_Project
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

```bash
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your prompt.

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This will take several minutes as it downloads PyTorch and other large packages.

### Step 6: Verify Installation

```bash
python demo.py --text "Aspirin causes stomach bleeding"
```

If you see entity predictions, the installation is successful!

---

## 4. Running The Project

### Quick Reference Commands

| Task | Command |
|------|---------|
| Run unit tests | `python -m pytest tests/ -v` |
| Run model + metrics tests only | `python -m pytest tests/test_model.py tests/test_metrics.py -v` |
| Preprocess data | `python -m src.data.preprocess` |
| Train model | `python -m src.training.train` |
| Evaluate model | `python -m src.training.eval` |
| Interactive CLI demo | `python demo.py` |
| Demo with specific text | `python demo.py --text "your text"` |
| Web demo (Gradio) | `python app.py` |

### Testing the Model

#### Run Unit Tests

```bash
python -m pytest tests/test_model.py tests/test_metrics.py -v
```

Expected output: `37 passed, 1 skipped`

#### Run Interactive Demo

```bash
python demo.py
```

Then type sentences like:
- "Metformin is used to treat diabetes"
- "Ibuprofen may cause kidney damage"
- "Penicillin allergies can result in anaphylaxis"

#### Run Web Demo

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

---

## 5. Using Claude Code With This Project

### Starting Claude Code

Navigate to the project folder and start Claude Code:

```bash
cd C:\path\to\NLP_Project
claude
```

### Example Prompts for Claude Code

#### Understanding the Project

```
"Explain the architecture of this BiLSTM-CRF NER project. What are the main components and how do they work together?"
```

#### Running Tests

```
"Run the unit tests and tell me if there are any failures. If there are, fix them."
```

#### Training the Model

```
"Run the training pipeline with the current configuration and show me the results."
```

#### Evaluating Performance

```
"Evaluate the trained model on the test set and give me a detailed analysis of the performance metrics."
```

#### Running the Demo

```
"Test the demo with these sentences: 'Aspirin causes bleeding', 'Metformin treats diabetes', 'Penicillin can cause allergic reactions'"
```

---

## 6. Suggested Improvements

### High Priority Improvements

#### 1. Download Real BC5CDR Dataset

**Prompt for Claude Code:**
```
"Download the real BC5CDR dataset from the official source, parse it properly, and retrain the model. The current model was trained on sample data only."
```

**Why:** The model was trained on ~80 sample sentences. With the full dataset (~5000+ sentences), F1 score should reach 85-87%.

#### 2. Add GloVe Pre-trained Embeddings

**Prompt for Claude Code:**
```
"Download GloVe 100d embeddings and integrate them into the model. Update the config to use pretrained embeddings."
```

**Why:** Pre-trained embeddings provide better word representations, especially for rare words.

#### 3. Run Full Ablation Study

**Prompt for Claude Code:**
```
"Run the ablation study script to compare different model configurations. Generate a table and visualizations of the results."
```

**Why:** Shows the contribution of each component (CRF, Char CNN, Attention).

### Medium Priority Improvements

#### 4. Improve Error Analysis

**Prompt for Claude Code:**
```
"Run the error analysis tools and generate a detailed report showing: boundary errors, type errors, missed entities, and spurious predictions. Create visualizations."
```

#### 5. Add Cross-Validation

**Prompt for Claude Code:**
```
"Implement k-fold cross-validation for more robust evaluation. Report mean and standard deviation of F1 scores."
```

#### 6. Add Learning Rate Finder

**Prompt for Claude Code:**
```
"Implement a learning rate finder to automatically find the optimal learning rate for training."
```

### Low Priority / Nice-to-Have

#### 7. Add Model Ensembling

**Prompt for Claude Code:**
```
"Implement model ensembling by training multiple models with different random seeds and combining their predictions."
```

#### 8. Add BERT Integration (Advanced)

**Prompt for Claude Code:**
```
"Add an option to use BioBERT or PubMedBERT embeddings instead of GloVe for better biomedical text understanding."
```

#### 9. Create Jupyter Notebook Demo

**Prompt for Claude Code:**
```
"Create a Jupyter notebook that demonstrates the full pipeline: data loading, training, evaluation, and prediction with visualizations."
```

---

## 7. Project Structure Reference

```
NLP_Project/
│
├── config/
│   └── config.yaml          # All configuration options
│
├── src/
│   ├── data/
│   │   ├── bc5cdr_parser.py  # Dataset parser
│   │   ├── preprocess.py     # Data preprocessing
│   │   └── dataset.py        # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── bilstm_crf.py     # Main model
│   │   ├── layers.py         # CRF, CharCNN, Highway
│   │   └── attention.py      # Self-attention
│   │
│   ├── training/
│   │   ├── train.py          # Training pipeline
│   │   └── eval.py           # Evaluation script
│   │
│   └── utils/
│       ├── vocab.py          # Vocabularies
│       ├── metrics.py        # P/R/F1 metrics
│       ├── analysis.py       # Error analysis
│       └── visualization.py  # Plotting
│
├── tests/
│   ├── test_crf.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_vocab.py
│
├── artifacts/                # Saved models
├── data/                     # Dataset files
├── reports/                  # Output reports
│
├── demo.py                   # CLI demo
├── app.py                    # Gradio web demo
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 8. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Make sure venv is activated: `venv\Scripts\activate` |
| `CUDA out of memory` | Reduce batch size in `config/config.yaml` |
| `Model not found` | Run training first: `python -m src.training.train` |
| Tests failing | Ask Claude Code: "Fix the failing tests" |
| Gradio not working | Install it: `pip install gradio` |

### Getting Help from Claude Code

If you encounter any issue, just describe it to Claude Code:

```
"I'm getting this error: [paste error message]. Can you fix it?"
```

```
"The training is not working. Can you debug and tell me what's wrong?"
```

```
"Help me understand why the F1 score is so low and suggest improvements."
```

---

## 9. Quick Start Checklist

Follow these steps in order:

- [ ] **Step 1:** Extract the ZIP file
- [ ] **Step 2:** Install Node.js from https://nodejs.org/
- [ ] **Step 3:** Install Claude Code: `npm install -g @anthropic-ai/claude-code`
- [ ] **Step 4:** Open terminal in project folder
- [ ] **Step 5:** Create venv: `python -m venv venv`
- [ ] **Step 6:** Activate: `venv\Scripts\activate`
- [ ] **Step 7:** Install deps: `pip install -r requirements.txt`
- [ ] **Step 8:** Test demo: `python demo.py`
- [ ] **Step 9:** Start Claude Code: `claude`
- [ ] **Step 10:** Ask Claude to run tests and make improvements!

---

## Good luck, Hassan! 🚀

Feel free to contact me if you have any questions.

**-- Yasser Hamdan**

---

### How to Compile the LaTeX Version

If you want a nicely formatted PDF, you can compile the `HANDOFF_GUIDE_FOR_HASSAN.tex` file:

1. Go to https://www.overleaf.com/
2. Create a new project
3. Upload the `.tex` file
4. Click "Recompile"
5. Download the PDF
