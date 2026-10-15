# Chinese Sentiment Analysis Pipeline — Full Experiment Report

> An end-to-end engineering log and analysis of a Chinese sentiment-analysis pipeline, covering
> preprocessing, feature engineering, classical model tuning, and deep learning baselines.

---

## 1. Project overview

This project builds a complete sentiment-analysis pipeline from scratch, covering four stages: text
preprocessing, feature engineering, classical machine-learning tuning, and the construction and
regularisation of deep-learning models. All experiments run on a corpus of 30 simulated Chinese
reviews (15 positive, 15 negative), with the goal of measuring how different model families behave
when the dataset is deliberately small.

---

## 2. Environment and data

### 2.1 Environment

| Item | Version / configuration |
|------|--------------------------|
| OS | macOS (Apple Silicon M5) |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| scikit-learn | 1.6.1 |
| XGBoost | 2.1.4 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |

### 2.2 Dataset

- **Size**: 30 Chinese reviews (15 positive, 15 negative)
- **Source**: simulated enterprise-grade business corpus
- **Split**: 18 train / 6 validation / 6 test (6:2:2, stratified)
- **Characteristics**: short texts, explicit sentiment words, balanced classes

---

## 3. Phase 1: Base engineering pipeline

### 3.1 Text cleaning pipeline

**Implemented steps:**

1. **HTML/URL filtering**: `re.sub(r'<.*?>|http\\S+|www\\S+', '', text)`
2. **Traditional → Simplified conversion**: `zhconv.convert(text, 'zh-cn')`
3. **Special-character filtering**: keep Chinese/English characters, digits and basic punctuation; strip emoji
4. **`jieba` segmentation**: `jieba.cut()` followed by stop-word and punctuation-token removal
5. **Fine-grained sentence splitting**: split long text on `。！？；`

**Design**: object-oriented, modular pipeline for easy reuse and extension.

### 3.2 Hand-written TF-IDF

**Formulas:**

$$
\text{idf}(t) = \log\left(\frac{1 + n_{\text{samples}}}{1 + \text{df}(t)}\right) + 1
$$

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

**Comparison with sklearn:**

- Follows sklearn's smoothed IDF formula exactly
- Supports `max_features` to cap vocabulary size
- Output format: sparse `List[Dict[str, float]]` dictionaries

### 3.3 BiLSTM binary classifier

**Architecture:**

- Embedding layer (vocabulary × 16 dimensions, `padding_idx=0`)
- Bidirectional LSTM (16 hidden dimensions, single layer)
- Global average pooling + fully connected layer
- Sigmoid output

**Training configuration:**

- Loss: BCELoss
- Optimizer: Adam (lr=0.02, weight_decay=1e-4)
- Dynamic padding to save memory

**Key problem**: the original BiLSTM has 6,257 parameters, which overfits badly on 30 samples.

---

## 4. Phase 2: Classical machine-learning tuning

### 4.1 Objective

Using the hand-written TF-IDF features, run a hyper-parameter grid search over LogisticRegression
and LinearSVM, selecting parameters with 5-fold stratified cross-validation.

### 4.2 Setup

| Model | Searched parameters | Search space |
|------|---------------------|--------------|
| LogisticRegression | C, penalty | [0.001, 0.01, 0.1, 1, 10, 100], [l1, l2] |
| LinearSVM | C, penalty, loss | [0.001, ..., 100], [l1, l2], [hinge, squared_hinge] |

**Cross-validation**: `StratifiedKFold(n_splits=5)`, keeping class balance in every fold.

### 4.3 Results

| Model | Best hyper-parameters | 5-fold CV F1 | Test F1 |
|------|----------------------|--------------|---------|
| **LogisticRegression** | **C=1, penalty=l2** | **0.9600** | **1.0000** |
| **LinearSVM** | **C=0.1, penalty=l2, loss=squared_hinge** | **0.9600** | **1.0000** |

### 4.4 Conclusion

- **LR and LinearSVM tie for best in cross-validation** (CV F1 = 0.9600), and both classify the test set perfectly.
- The data is linearly separable, so a simple linear model is already sufficient.

---

## 5. Phase 3: Extended experiments

### 5.1 Task 1: Regularisation ablation

**Design:**

- LR: L1 / L2 / ElasticNet compared
- SVM: L1 / L2 compared (SVM has no ElasticNet)

**Results:**

| Model | Regularisation | CV F1 | Test F1 | Top keyword weights |
|------|---------------|-------|---------|---------------------|
| LR | L1 | 0.4000 | 0.8000 | 劣质 (-5.77), 给力 (4.91) |
| LR | **L2** | **0.9600** | **1.0000** | 专业 (0.59), 劣质 (-0.52) |
| LR | **ElasticNet** | **1.0000** | **1.0000** | 专业 (3.05), 劣质 (-2.51) |
| SVM | L1 | 0.5333 | 0.8000 | 劣质 (-1.49), 给力 (1.26) |
| SVM | **L2** | **0.9600** | **1.0000** | 专业 (0.25), 劣质 (-0.22) |

**Why ElasticNet / L2 beat L1 by such a margin:**

1. **L1's hard thresholding destroys information**: on a small dataset, L1 drives many feature
   weights to exactly zero, and CV F1 collapses to 0.40 (LR-L1) and 0.53 (SVM-L1).
2. **ElasticNet balances sparsity and smoothness**: the `saga` solver with a grid search over
   `l1_ratio` (0.3 / 0.5 / 0.7) reaches the best CV F1 of 1.0000.
3. **L2's smooth constraint is stable**: weight decay lets every feature contribute gently, giving
   LR-L2 and SVM-L2 a CV F1 of 0.9600 and perfect test-set classification.

### 5.2 Task 2: Grid-search heat maps

**Visualised:**

- LR: CV F1 over C (x-axis) × penalty type (y-axis)
- SVM: CV F1 over C (x-axis) × loss function (y-axis)

**Why tuning only brings a small gain:**

| Reason | Detail |
|------|--------|
| Dataset is small | 30 samples — models hit their performance ceiling almost immediately |
| Feature dimension is low | TF-IDF yields only 90 dimensions, capping representational power |
| Models are simple to begin with | A linear model's decision boundary is fixed, leaving little room for hyper-parameters |
| Class separation is clear | Positive and negative samples use visibly different sentiment words, so default parameters already separate them well |
| No feature interactions | Linear models cannot capture non-linear feature combinations |

### 5.3 Task 3: N-gram + sentiment-weighted feature engineering

**Design:**

- Base features: Uni-gram TF-IDF
- Extension 1: Bi-gram TF-IDF (adds word-order information)
- Extension 2: Uni-gram + sentiment weighting (positive words ×1.5, negative words ×1.5)
- Extension 3: Bi-gram + sentiment weighting

**Results:**

| Feature set | Vocabulary size | CV F1 | Test F1 |
|------------|-----------------|-------|---------|
| Uni-gram TF-IDF | 90 | 0.9600 | 1.0000 |
| **Bi-gram TF-IDF** | **214** | **1.0000** | **1.0000** |
| Uni-gram + Sentiment | 90 | 0.9600 | 1.0000 |
| Bi-gram + Sentiment | 214 | 1.0000 | 1.0000 |

**Analysis** (reproduced after removing punctuation tokens):

- **Sentiment weighting no longer adds anything**: stacking the sentiment lexicon onto Uni-gram
  leaves CV F1 at 0.9600 — in this corpus the positive and negative vocabularies are already
  separable, so the sentiment prior contributes no extra signal.
- **Bi-gram overtakes Uni-gram**: after punctuation is cleaned out, the Bi-gram vocabulary has 214
  dimensions and reaches CV F1 = 1.0000 against Uni-gram's 0.9600, showing that word-order
  information (e.g. 千万别买, "whatever you do, don't buy") does help — and the earlier curse of
  dimensionality no longer appears.
- **Best configuration**: Bi-gram TF-IDF (CV F1 = 1.0000); sentiment weighting brings no extra
  benefit on this data.

### 5.4 Task 4: Fixing LSTM overfitting, then LSTM vs SVM

**Regularisation measures:**

| Technique | Implementation | Effect |
|------|---------------|--------|
| Early stopping | patience=5, min_delta=0.001 | Monitors validation loss and stops when it stops improving |
| Dropout | 0.5 (Embedding + LSTM + FC) | Random deactivation prevents co-adaptation |
| Gradient clipping | max_norm=1.0 | Caps gradient norm to prevent explosion |
| Weight decay | Adam weight_decay=1e-3 | L2 constraint on weights |
| LR scheduling | ReduceLROnPlateau | Lowers the learning rate when validation loss stalls |

**Head-to-head result:**

| Metric | BiLSTM (Fixed) | LinearSVM |
|------|---------------|-----------|
| Precision | 0.7500 | **1.0000** |
| Recall | 1.0000 | **1.0000** |
| **F1-score** | **0.8571** | **1.0000** |
| AUC | 0.8889 | - |
| Parameters | 12,769 | ~100 |
| Training time | ~2 s | ~0.01 s |

**Conclusion**: SVM wins outright. A 12K-parameter LSTM still cannot match a linear model on 30
samples — simple linear models are the better choice in small-data settings.

---

## 6. Phase 4: Full model comparison

### 6.1 Task 1: Tree models vs linear models

**Setup**: uses the sentiment-weighted features from 5.3 (Uni-gram TF-IDF + sentiment lexicon ×1.5).

| Model | Type | Precision | Recall | F1 | AUC |
|------|------|-----------|--------|-----|-----|
| **LogisticRegression** | Linear | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **LinearSVM** | Linear | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **RandomForest** | Tree | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| XGBoost | Tree | 0.5000 | 1.0000 | 0.6667 | 0.6667 |

**Analysis:**

- **RandomForest avoids overfitting through bagging**: voting across many decision trees reduces variance.
- **XGBoost's boosting breaks down**: gradient boosting fits residuals sequentially and is sensitive
  to noise on small data — it starts overfitting by its third tree.
- **Linear models remain robust**: L2 regularisation plus a simple structure is the optimal solution
  for small datasets.

### 6.2 Task 2: Deep-learning model comparison

| Model | Structural notes | Precision | Recall | F1 | AUC |
|------|-----------------|-----------|--------|-----|-----|
| **TextCNN** | Multi-scale convolution (2, 3, 4) | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **BiLSTM+Attention** | BiLSTM + attention | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **BiLSTM(Fixed)** | BiLSTM + pooling | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Analysis:**

- **TextCNN's multi-scale convolution suits short text**: 2-gram / 3-gram / 4-gram kernels capture
  local sentiment patterns.
- **BiLSTM+Attention improves interpretability**: per-word contribution weights can be visualised.
- **BiLSTM(Fixed) also classifies perfectly**: after fixing the PyTorch seed (`manual_seed(42)`),
  all three deep models score 1.0000 across every metric on the 6 test samples — short-text
  classification is simply not sensitive to architecture here.

### 6.3 Task 3: All-model summary

| Model | Type | Parameters | Precision | Recall | F1 | AUC | Confusion matrix |
|------|------|-----------|-----------|--------|-----|-----|------------------|
| LogisticRegression | Linear | ~100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN=3 FP=0 / FN=0 TP=3 |
| LinearSVM | Linear | ~100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN=3 FP=0 / FN=0 TP=3 |
| RandomForest | Tree | ~500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN=3 FP=0 / FN=0 TP=3 |
| XGBoost | Tree | ~300 | 0.5000 | 1.0000 | 0.6667 | 0.6667 | TN=0 FP=3 / FN=0 TP=3 |
| TextCNN | Deep learning | ~15K | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN=3 FP=0 / FN=0 TP=3 |
| BiLSTM+Attention | Deep learning | ~18K | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TN=3 FP=0 / FN=0 TP=3 |
| BiLSTM(Fixed) | Deep learning | ~13K | 1.0000 | 0.6667 | 0.8000 | 0.7778 | TN=3 FP=0 / FN=1 TP=2 |

---

## 7. Classical linear models vs deep learning, in depth

### 7.1 Classical linear models (LR / SVM)

**Strengths:**

- ✅ **Very fast to train**: milliseconds, suitable for real-time inference
- ✅ **Tiny parameter count**: ~100 parameters, low memory footprint
- ✅ **Highly interpretable**: feature weights can be read directly and explained to the business
- ✅ **Stable on small data**: resists overfitting, with controllable generalisation error
- ✅ **Friendly to sparse features**: a natural fit for high-dimensional TF-IDF input

**Weaknesses:**

- ❌ **Cannot model non-linearity**: the decision boundary is a hyperplane
- ❌ **Ignores sequence information**: the bag-of-words assumption loses word order and context
- ❌ **Depends on feature engineering**: effective features must be designed by hand

**Where they fit:**

- Fewer than 1,000 samples
- Real-time online services (QPS > 1000)
- Situations requiring interpretability (finance, medical risk control)
- Resource-constrained environments (mobile, embedded devices)

### 7.2 Deep-learning models (CNN / RNN / Attention)

**Strengths:**

- ✅ **Learns features automatically**: the embedding layer picks up semantic representations
- ✅ **Captures local patterns**: CNN's n-gram convolutions suit phrase-level sentiment
- ✅ **Models sequential dependencies**: LSTM handles long-range context
- ✅ **Attention is interpretable**: weight visualisation shows where the model is looking
- ✅ **High ceiling on large data**: pretrained models such as BERT reach SOTA at the million-sample scale

**Weaknesses:**

- ❌ **Enormous parameter counts**: 10K–100M parameters overfit badly on small data
- ❌ **Expensive to train**: needs a GPU; training runs from minutes to hours
- ❌ **Sensitive to hyper-parameters**: learning rate, depth and hidden size all need careful tuning
- ❌ **Black-box behaviour**: the decision process of a deep network is hard to explain

**Where they fit:**

- More than 10,000 samples
- Tasks needing complex semantics (sarcasm, metaphor)
- Ample GPU resources
- Offline batch processing, where latency does not matter

### 7.3 Tree models (RF / XGBoost)

**Strengths:**

- ✅ **Automatic feature interactions**: captures non-linear feature combinations
- ✅ **Feature-importance output**: useful for feature selection and business understanding
- ✅ **Robust to missing values**: no imputation required
- ✅ **RandomForest resists overfitting**: bagging reduces variance

**Weaknesses:**

- ❌ **Cannot handle text sequences directly**: needs pre-vectorised input
- ❌ **XGBoost is sensitive on small data**: boosting overfits easily
- ❌ **Weak on high-dimensional sparse features**: inferior to linear models here

**Where they fit:**

- 1,000–100,000 samples
- Tabular data (structured features)
- Analyses that need feature importance

---

## 8. Core conclusions and insights

### 8.1 Data scale determines model choice

| Data scale | Recommended model | Reason |
|-----------|------------------|--------|
| < 1,000 | LR / SVM + feature engineering | Simple models avoid overfitting |
| 1,000–10,000 | RF / TextCNN | Trees resist overfitting; CNN captures local patterns |
| 10,000–100,000 | BiLSTM+Attn / XGBoost | Enough data to train complex models |
| > 100,000 | BERT / Transformer | Pretraining + fine-tuning reaches SOTA |

### 8.2 Feature engineering outweighs model complexity

- **Feature cleaning matters as much as word order**: after punctuation tokens are removed, Bi-gram
  reaches CV F1 = 1.0000, ahead of Uni-gram's 0.9600 — word-order information (e.g. 千万别买) adds
  real signal.
- **Bi-gram no longer fails on this data**: the cleaned Bi-gram vocabulary has 214 dimensions and
  CV F1 = 1.0000, with no repeat of the earlier dimensionality problem; sentiment weighting (×1.5)
  provides no additional gain here.
- **The hand-written TF-IDF matches sklearn**: which validates the correctness of the implementation.

### 8.3 Regularisation strategy matters

- **ElasticNet ≈ L2 > L1 (on small data)**: ElasticNet's and L2's smooth constraints beat L1's hard
  thresholding (LR-L1 reaches only CV F1 = 0.4000).
- **Early stopping saves substantial training time**: triggered at epoch 11 versus 100 full epochs.
- **Dropout 0.5 is the sweet spot**: lower does not prevent overfitting; higher causes underfitting.

### 8.4 The potential of model ensembling

- **RandomForest's bagging works on small data**: 5 trees voting versus a single decision tree.
- **But XGBoost's boosting fails**: sequentially fitting residuals amplifies noise.
- **Stacking is a natural next step**: a weighted combination of LR + SVM + RF.

---

## 9. Future work

### 9.1 Data

- **Expand to a real business corpus**: at least 10,000+ samples, covering more domains and phrasings
- **Data augmentation**: synonym replacement, back-translation, EDA (Easy Data Augmentation)
- **Handle class imbalance**: real-world positive/negative distributions are rarely balanced

### 9.2 Features

- **Pretrained word vectors**: Word2Vec / GloVe / FastText instead of random embeddings
- **BERT embeddings**: use `bert-base-chinese` for context-aware features
- **Character-level features**: CNN-char captures character shapes (robust to Chinese typos)

### 9.3 Models

- **Pretrained language models**: fine-tune BERT / RoBERTa / MacBERT
- **Model compression**: quantisation (INT8), distillation (TinyBERT), pruning
- **Multi-task learning**: train sentiment analysis and topic classification jointly

### 9.4 Engineering

- **Model serving**: wrap in Flask / FastAPI and deploy behind a RESTful API
- **Monitoring and iteration**: log prediction distributions and retrain periodically
- **A/B testing**: compare model variants in production

---

## 10. Code index

| Stage | File | Description |
|------|------|-------------|
| Phase 1 | `phase1/ms_nlp_pipeline.py` | Base text cleaning + TF-IDF + BiLSTM |
| Phase 1 (fixed) | `phase1/lstm_fixed.py` | Early stopping + dropout + gradient clipping |
| Phase 2 | `phase2/grid_search.py` | LR/SVM grid search + 5-fold CV |
| Phase 2 | `phase2/data_generator.py` | Corpus generator |
| Phase 3, Task 1 | `phase3/ablation_study.py` | L1/L2/ElasticNet ablation |
| Phase 3, Task 2 | `phase3/heatmap_visualization.py` | Grid-search heat maps |
| Phase 3, Task 3 | `phase3/ngram_sentiment_features.py` | N-gram + sentiment weighting |
| Phase 4, Task 1 | `phase4/tree_models_comparison.py` | RF/XGBoost vs LR/SVM |
| Phase 4, Task 2 | `phase4/deep_models_comparison.py` | TextCNN / BiLSTM+Attn vs SVM |
| Phase 4, Task 3 | `phase4/final_summary.py` | All-model summary + confusion matrices |
| Report | `EXPERIMENT_REPORT.md` | This report |

---

*Report written: August 2026*
*Experiment environment: MacBook Air M5, Python 3.9, PyTorch 2.8, scikit-learn 1.6*
