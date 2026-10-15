# Chinese Sentiment Analysis Pipeline

An end-to-end NLP pipeline for Chinese-language sentiment classification, built from scratch —
text cleaning, a hand-written TF-IDF vectoriser, classical model tuning, feature engineering, and
deep learning baselines, ending in a head-to-head comparison of seven models.

The project was written to answer a specific question: **on a small, clean, short-text corpus, how
much model complexity is actually worth paying for?** Every stage is instrumented, and the
conclusions in [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) are drawn from the recorded runs
rather than assumed.

## Pipeline

```
raw text ─► cleaning ─► hand-written TF-IDF ─► feature engineering ─► model comparison
```

| Stage | What it covers |
| --- | --- |
| 1. Preprocessing | HTML/URL stripping, Traditional→Simplified conversion, punctuation and emoji filtering, `jieba` segmentation, stop-word removal, fine-grained sentence splitting |
| 2. Vectorisation | TF-IDF implemented from scratch (sklearn's smoothed IDF formula), sparse `List[Dict[str, float]]` output, `max_features` vocabulary cap |
| 3. Classical tuning | `LogisticRegression` and `LinearSVM` hyper-parameter grid search with 5-fold stratified cross-validation |
| 4. Feature engineering | Uni-gram vs Bi-gram TF-IDF, sentiment-lexicon re-weighting, L1/L2/ElasticNet regularisation ablation |
| 5. Model comparison | BiLSTM (baseline and regularised), TextCNN, BiLSTM + Attention, RandomForest, XGBoost, LR, SVM — all scored on the same split |

## Repository layout

```
ms-nlp-pipeline-task/
├── phase1/
│   ├── ms_nlp_pipeline.py          # cleaning + hand-written TF-IDF + BiLSTM baseline
│   └── lstm_fixed.py               # early stopping, dropout, gradient clipping, dual pooling
├── phase2/
│   ├── grid_search.py              # LR / SVM grid search + 5-fold stratified CV
│   └── data_generator.py           # corpus generator
├── phase3/
│   ├── ablation_study.py           # L1 / L2 / ElasticNet ablation, keyword weights
│   ├── heatmap_visualization.py    # CV F1 heat maps over the search space
│   ├── ngram_sentiment_features.py # N-gram + sentiment-weighted features
│   └── grid_search_heatmap.png
├── phase4/
│   ├── tree_models_comparison.py   # RandomForest / XGBoost vs linear models
│   ├── deep_models_comparison.py   # TextCNN / BiLSTM+Attention / BiLSTM(Fixed)
│   └── final_summary.py            # all seven models, metrics + confusion matrices
├── EXPERIMENT_REPORT.md            # full experiment log and analysis
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt

python phase1/ms_nlp_pipeline.py          # cleaning + TF-IDF + BiLSTM baseline
python phase1/lstm_fixed.py               # regularised BiLSTM
python phase2/grid_search.py              # LR / SVM tuning
python phase3/ablation_study.py           # regularisation ablation
python phase3/heatmap_visualization.py    # search-space heat maps
python phase3/ngram_sentiment_features.py # N-gram + sentiment weighting
python phase4/tree_models_comparison.py   # tree vs linear models
python phase4/deep_models_comparison.py   # deep model comparison
python phase4/final_summary.py            # aggregate metrics + confusion matrices
```

The corpus is 30 simulated Chinese reviews (15 positive, 15 negative), split 6:2:2 into
train / validation / test with stratified sampling. PyTorch runs are seeded (`manual_seed(42)`)
so the reported numbers reproduce.

## Findings

- **Small data does not reward complexity.** On 30 samples a ~100-parameter linear model reaches
  F1 = 1.0000 on the test set, while a 12K-parameter BiLSTM plateaus around 0.80–0.86. The
  bottleneck is the amount of evidence, not the capacity of the model.
- **How you regularise matters more than which model you pick.** L1 zeroed out so many weights that
  CV F1 collapsed to 0.40 (LR) and 0.53 (SVM); L2 and ElasticNet held at 0.96 and 1.00.
- **Boosting and bagging diverge sharply on tiny corpora.** RandomForest's bagging stayed stable at
  F1 = 1.0000, while XGBoost's sequential residual fitting began overfitting by its third tree and
  fell to 0.6667.
- **Bi-gram features helped once punctuation tokens were removed** — vocabulary 214, CV F1 = 1.0000
  vs 0.9600 for Uni-gram — while sentiment-lexicon re-weighting added nothing on this corpus,
  because the positive and negative vocabularies were already separable.
- **A hand-written TF-IDF reproduces the library implementation** closely enough that downstream
  results are indistinguishable, which was the point of writing it.

See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for the full tables, the model-by-model
comparison, and the reasoning behind each decision.

## Environment

- Python 3.9+
- PyTorch 2.x, scikit-learn 1.6+, XGBoost
- jieba, zhconv, NumPy, pandas, matplotlib, seaborn
