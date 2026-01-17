# Academic Report Outline: Domain-Specific NER with BiLSTM-CRF

## Abstract (200-250 words)

- Brief overview of Named Entity Recognition (NER) and its importance
- Focus on domain-specific NER (biomedical domain)
- Summary of approach: BiLSTM-CRF architecture
- Key results: F1 scores, comparison with baseline
- Main contributions of the work

## 1. Introduction

### 1.1 Motivation
- Importance of NER in NLP
- Challenges in domain-specific NER (biomedical text)
- Why automated entity recognition is needed in biomedical literature
- Applications: drug discovery, clinical decision support, literature mining

### 1.2 Problem Statement
- Task definition: Identify and classify named entities (diseases, chemicals) in biomedical text
- BIO tagging scheme explanation
- Challenges: domain vocabulary, entity boundary detection, nested entities

### 1.3 Research Questions
- How effective is BiLSTM-CRF for biomedical NER?
- What is the contribution of the CRF layer compared to baseline BiLSTM?
- How do model hyperparameters affect performance?

### 1.4 Contributions
- Complete implementation of BiLSTM-CRF from scratch in PyTorch
- Comparison with baseline model
- Analysis of model performance on biomedical entities
- Educational codebase for understanding sequence labeling

## 2. Background and Related Work

### 2.1 Named Entity Recognition
- Definition and formalization as sequence labeling
- BIO/BILOU tagging schemes
- Evaluation metrics: precision, recall, F1
- Historical approaches: rule-based, CRF, HMM

### 2.2 Neural Sequence Labeling
- Early work: window-based approaches
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) networks
- Bidirectional LSTMs for sequence labeling

### 2.3 Conditional Random Fields (CRFs)
- CRF formulation for sequence labeling
- Transition parameters and inference
- Forward-backward algorithm
- Viterbi decoding

### 2.4 BiLSTM-CRF Architecture
- Lample et al. (2016): "Neural Architectures for Named Entity Recognition"
- Ma & Hovy (2016): "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"
- Key advantages: captures context + models label dependencies
- State-of-the-art before transformer era

### 2.5 Biomedical NER
- Domain characteristics: specialized terminology, ambiguity
- BC5CDR corpus and other biomedical NER datasets
- Previous work on biomedical entity recognition
- Comparison with general-domain NER

## 3. Dataset

### 3.1 BC5CDR Corpus
- Source and collection methodology
- Entity types: Disease, Chemical
- Size: number of abstracts, sentences, entities
- Train/dev/test split ratios

### 3.2 Data Preprocessing
- Text cleaning and tokenization
- BIO tag conversion
- Handling of special cases (e.g., multi-token entities)
- Vocabulary construction

### 3.3 Dataset Statistics
- Sentence length distribution
- Entity frequency and distribution
- Class balance analysis
- Examples of annotated sentences

### 3.4 Data Challenges
- Out-of-vocabulary words
- Long and complex entity names
- Ambiguous entity boundaries
- Inter-annotator agreement (if available)

## 4. Methodology

### 4.1 Model Architecture

#### 4.1.1 Baseline Model: BiLSTM Tagger
- Word embeddings layer
- Bidirectional LSTM encoder
- Linear projection to tag space
- Softmax + cross-entropy loss
- Independent token predictions

#### 4.1.2 Main Model: BiLSTM-CRF
- Word embeddings layer (random initialization)
- Bidirectional LSTM encoder
  - Number of layers
  - Hidden size
  - Dropout regularization
- Linear projection to emission scores
- CRF layer
  - Transition matrix
  - Forward algorithm for partition function
  - Viterbi decoding for inference
- Negative log-likelihood loss

#### 4.1.3 Optional Enhancements
- Character-level CNN features
- Pretrained word embeddings (Word2Vec, GloVe)
- Attention mechanisms

### 4.2 Training Procedure
- Optimization algorithm: Adam
- Learning rate and schedule
- Batch size
- Gradient clipping
- Early stopping criteria
- Regularization: dropout, weight decay

### 4.3 Evaluation Metrics
- Entity-level precision, recall, F1 (strict matching)
- Per-entity-type metrics
- Token-level accuracy
- Confusion analysis

## 5. Experiments

### 5.1 Experimental Setup
- Hardware: CPU/GPU specifications
- Software: PyTorch version, Python version
- Hyperparameter values
- Random seeds for reproducibility

### 5.2 Hyperparameter Tuning
- Parameters explored: embedding_dim, hidden_size, learning_rate, dropout
- Validation strategy
- Best configuration selection

### 5.3 Training Details
- Number of epochs to convergence
- Training time per epoch
- Memory requirements
- Convergence behavior (loss curves)

### 5.4 Baseline Experiments
- BiLSTM without CRF performance
- Ablation studies (e.g., removing bidirectionality)
- Comparison with simpler models

## 6. Results

### 6.1 Overall Performance

**Table 1: Main Results on Test Set**

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| BiLSTM (Baseline) | XX.X% | XX.X% | XX.X% |
| BiLSTM-CRF | XX.X% | XX.X% | XX.X% |

### 6.2 Per-Entity Performance

**Table 2: Results by Entity Type**

| Entity Type | Model | Precision | Recall | F1 |
|-------------|-------|-----------|--------|-----|
| Chemical | Baseline | XX.X% | XX.X% | XX.X% |
| Chemical | BiLSTM-CRF | XX.X% | XX.X% | XX.X% |
| Disease | Baseline | XX.X% | XX.X% | XX.X% |
| Disease | BiLSTM-CRF | XX.X% | XX.X% | XX.X% |

### 6.3 Learning Curves
- Training and validation loss over epochs
- F1 score on dev set over epochs
- Comparison of baseline vs. BiLSTM-CRF convergence

### 6.4 Impact of CRF Layer
- Performance gain from adding CRF
- Examples where CRF helps (transition constraints)
- Analysis of learned transition matrix

### 6.5 Effect of Hyperparameters
- Impact of hidden size
- Impact of number of layers
- Impact of dropout rate
- Impact of learning rate

## 7. Discussion

### 7.1 Key Findings
- BiLSTM-CRF outperforms baseline by X%
- CRF particularly helps with label sequence coherence
- Both entity types benefit from BiLSTM-CRF
- Model generalizes well to biomedical domain

### 7.2 Error Analysis

#### 7.2.1 Common Errors
- Boundary errors (partial matches)
- Type confusion (Disease vs. Chemical)
- Missed entities (false negatives)
- False alarms (false positives)

#### 7.2.2 Error Examples
- Specific cases where model fails
- Analysis of why these errors occur
- Patterns in errors

### 7.3 Qualitative Analysis
- Visualization of predictions
- Attention weights (if applicable)
- Transition probabilities learned by CRF
- Examples of correct predictions

### 7.4 Comparison with Prior Work
- How results compare to published baselines on BC5CDR
- State-of-the-art performance (transformer models)
- Where BiLSTM-CRF stands in 2024 landscape

### 7.5 Limitations
- Lack of pretrained embeddings (random initialization)
- No contextualized representations (ELMo, BERT)
- Character features not implemented/explored fully
- Limited training data
- Computational constraints

## 8. Conclusion

### 8.1 Summary
- Recap of problem and approach
- Main findings and results
- Validation of BiLSTM-CRF effectiveness

### 8.2 Achievements
- Successful implementation of BiLSTM-CRF
- Strong performance on biomedical NER
- Clean, educational codebase

### 8.3 Future Work
- Integration of pretrained embeddings (BioWordVec, BioBERT)
- Character-level features (char-CNN, char-LSTM)
- Attention mechanisms
- Multi-task learning (joint disease-chemical recognition)
- Transfer learning from general domain
- Application to other biomedical NER datasets
- Deployment considerations (speed, memory)
- Handling of nested entities

## References

### Key Papers to Cite

1. **BiLSTM-CRF Architecture**
   - Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. NAACL-HLT.
   - Ma, X., & Hovy, E. (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. ACL.

2. **CRF Foundation**
   - Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields. ICML.

3. **LSTM**
   - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

4. **BC5CDR Dataset**
   - Li, J., Sun, Y., Johnson, R. J., Sciaky, D., Wei, C. H., Leaman, R., ... & Lu, Z. (2016). BioCreative V CDR task corpus. Database.

5. **Biomedical NER**
   - Relevant domain-specific papers
   - Survey papers on biomedical NER

6. **Deep Learning for NLP**
   - Goldberg, Y. (2017). Neural Network Methods for Natural Language Processing. Synthesis Lectures on HLT.

## Appendices

### Appendix A: Hyperparameter Details
- Complete table of all hyperparameter values tested
- Full configuration files

### Appendix B: Additional Results
- Detailed per-tag confusion matrices
- More learning curves
- Statistical significance tests

### Appendix C: Implementation Details
- Key code snippets (CRF forward-backward, Viterbi)
- Model architecture diagrams
- Data flow diagrams

### Appendix D: Dataset Examples
- Representative annotated examples from train/dev/test
- Challenging examples

### Appendix E: Reproducibility
- Exact commands to reproduce results
- Environment specifications
- Random seeds used
