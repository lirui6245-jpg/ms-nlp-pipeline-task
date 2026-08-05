# NLP 基础工程闭环任务

文本清洗 → 手写TF-IDF → BiLSTM / ML Tuning → 特征工程 → 全模型对比

## 项目结构

```
ms-nlp-pipeline-task/
├── task1/
│   ├── ms_nlp_pipeline.py      # Phase 1: 文本清洗 + 手写TF-IDF + BiLSTM基线
│   └── lstm_fixed.py           # Phase 1修复: EarlyStopping + Dropout + 梯度裁剪 + 双池化
├── task2/
│   ├── grid_search.py           # Phase 2: 超参网格搜索 + 5折分层CV
│   └── data_generator.py        # 数据生成脚本
├── task3/
│   ├── ablation_study.py        # Phase 3-1: L1/L2/ElasticNet消融实验
│   ├── heatmap_visualization.py # Phase 3-2: 网格搜索热力图可视化
│   ├── ngram_sentiment_features.py # Phase 3-3: N-gram + 情感加权
│   └── grid_search_heatmap.png  # 热力图输出
├── task4/
│   ├── tree_models_comparison.py  # Phase 4-1: RF/XGBoost vs LR/SVM
│   ├── deep_models_comparison.py  # Phase 4-2: TextCNN / BiLSTM+Attention / BiLSTM(Fixed)
│   └── final_summary.py         # Phase 4-3: 全模型汇总 + 混淆矩阵
├── EXPERIMENT_REPORT.md         # 完整实验报告
└── requirements.txt
```

## 快速启动

```bash
pip3 install -r requirements.txt

# Phase 1: 基础Pipeline
python3 task1/ms_nlp_pipeline.py

# Phase 1: 修复版LSTM
python3 task1/lstm_fixed.py

# Phase 2: 网格搜索
python3 task2/grid_search.py

# Phase 3: 消融实验
python3 task3/ablation_study.py

# Phase 3: 热力图
python3 task3/heatmap_visualization.py

# Phase 3: N-gram特征
python3 task3/ngram_sentiment_features.py

# Phase 4: 树模型对比
python3 task4/tree_models_comparison.py

# Phase 4: 深度学习对比
python3 task4/deep_models_comparison.py

# Phase 4: 全模型汇总
python3 task4/final_summary.py
```

## 各阶段说明

### Phase 1: 文本预处理与深度学习基线
- 工业级文本清洗（繁简转换、HTML/URL过滤、结巴分词、停用词剔除、细粒度分句）
- 手写TF-IDF（严格遵循sklearn平滑IDF公式）
- BiLSTM二分类（Embedding → BiLSTM → 全局平均池化 → FC）

### Phase 1修复: LSTM过拟合修复
- EarlyStopping (patience=5)
- Dropout 0.5（Embedding/LSTM/FC全链路）
- 梯度裁剪（max_norm=1.0）
- 双池化（平均池化 + 最大池化）
- 学习率衰减（ReduceLROnPlateau）
- 权重衰减（weight_decay=1e-3）

### Phase 2: 传统机器学习调优
- 基于手写TF-IDF特征
- LogisticRegression / LinearSVM 网格搜索
- 5折分层交叉验证（StratifiedKFold）
- 最优参数：LR(C=10,l2), SVM(C=10,l2,hinge)

### Phase 3: 扩展实验
- **消融实验**: L1/L2/ElasticNet正则对比，输出关键词权重
- **热力图**: C值 vs Penalty/Loss 的CV F1热力图可视化
- **特征工程**: N-gram(1,2) + 情感加权（正面/负面词×1.5）

### Phase 4: 全面模型对比
- **树模型**: RandomForest、XGBoost vs 线性模型
- **深度学习**: TextCNN、BiLSTM+Attention、BiLSTM(Fixed)
- **汇总**: 全部7个模型的Precision/Recall/F1/AUC + 混淆矩阵

## 核心结论

| 数据规模 | 推荐模型 | 原因 |
|---------|---------|------|
| < 1,000条 | LR / SVM + 特征工程 | 简单模型避免过拟合 |
| 1,000-10,000条 | RF / TextCNN | 树模型抗过拟合，CNN捕获局部模式 |
| 10,000-100,000条 | BiLSTM+Attn / XGBoost | 足够数据训练复杂模型 |
| > 100,000条 | BERT / Transformer | 预训练+微调达到SOTA |

**关键发现**:
- 小数据集（30条）下线性模型最优，LSTM严重过拟合
- 情感加权将CV F1从0.9333提升到1.0000
- Bi-gram在小数据上因维度膨胀导致过拟合
- RF通过bagging抗过拟合，XGBoost的boosting在小数据上失效

## 环境

- Python 3.9+
- PyTorch 2.x
- scikit-learn 1.6+
- xgboost, jieba, zhconv, matplotlib, seaborn
