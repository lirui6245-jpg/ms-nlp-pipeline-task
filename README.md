# NLP 基础工程闭环任务 (Text Cleaning -> TF-IDF -> BiLSTM / ML Tuning)

本项目为微软实习期间交付的自然语言处理（NLP）基础工程闭环任务实现。代码完全采用面向对象与模块化管道（Pipeline）设计，实现了工业级预处理、统计特征向量化、深度学习序列分类以及传统机器学习模型调优的整合。

## 项目结构

    ms-nlp-pipeline-task/
    ├── task1/
    │   └── ms_nlp_pipeline.py      # 第一阶段：文本清洗 + 手写TF-IDF + BiLSTM二分类
    ├── task2/
    │   ├── grid_search.py           # 第二阶段：基于手写TF-IDF的超参网格搜索 + 5折CV
    │   └── data_generator.py        # 数据生成脚本（扩充语料以支持5折交叉验证）
    ├── README.md
    └── requirements.txt

## 1. 快速启动 (Quick Start)

### 环境安装
    pip3 install -r requirements.txt

### 运行第一阶段
    python3 task1/ms_nlp_pipeline.py

### 运行第二阶段
    python3 task2/grid_search.py

## 2. 第一阶段：文本预处理与深度学习基线

**文件**: task1/ms_nlp_pipeline.py

### 实现内容
1. **工业级文本清洗管道**
   - 繁简转换（zhconv）
   - 特殊符号/表情过滤（正则）
   - 结巴分词 + 停用词剔除
   - 细粒度分句

2. **手写 TF-IDF 特征提取**
   - 严格遵循 sklearn 平滑 IDF 公式：idf(t) = log((1 + n_samples) / (1 + df(t))) + 1
   - 支持 max_features 限制词表规模

3. **BiLSTM 情感二分类模型**
   - 双向 LSTM 捕获上下文特征
   - 动态 Padding 节省内存
   - BCELoss + Adam 优化器

## 3. 第二阶段：传统机器学习模型调优

**文件**: task2/grid_search.py

### 任务目标
基于第一阶段手写的 TF-IDF 特征，对 LogisticRegression 和 LinearSVM 进行超参网格搜索，通过 5 折分层交叉验证选出最优参数组合。

### 实现内容
1. **特征矩阵转换**
   - 将手写 TF-IDF 的字典稀疏表示转换为 numpy 稠密矩阵，适配 sklearn 接口

2. **超参网格搜索（GridSearchCV）**
   - **LogisticRegression**: 搜索正则化系数 C、惩罚项 penalty（l1/l2）
   - **LinearSVM**: 搜索 C、penalty、loss（hinge/squared_hinge），并处理参数兼容性问题（l1 仅支持 squared_hinge + dual=False）

3. **5 折分层交叉验证（StratifiedKFold）**
   - 保证每折训练/验证集中正负样本比例一致
   - 评估指标：F1-score（关注正类召回）

4. **模型评估与对比**
   - 输出每组模型的最优超参与 CV 分数
   - 在独立测试集上进行最终评估

### 实验结果

| 模型 | 最优超参 | 5折CV F1 | 测试集 F1 |
|------|---------|---------|----------|
| LogisticRegression | C=10, penalty=l2 | 0.8667 | 1.0000 |
| LinearSVM | C=10, penalty=l2, loss=hinge | **0.9333** | 1.0000 |

**结论**: LinearSVM 在交叉验证中表现更优（CV F1: 0.9333 vs 0.8667），测试集上两个模型均达到完美分类。该结果验证了手写 TF-IDF 特征对传统线性分类器的有效性。

## 4. 技术要点

- **分层抽样**: 二分类任务中，数据划分与交叉验证均使用 stratify=y 或 StratifiedKFold，确保类别平衡
- **参数兼容性**: LinearSVM 的 l1 惩罚与 hinge loss 不兼容，通过分组参数网格（list of dicts）避免无效组合
- **数据扩充**: 为支持 5 折交叉验证的最小样本要求，将模拟语料从 10 条扩充至 30 条（正负类各 15 条）
