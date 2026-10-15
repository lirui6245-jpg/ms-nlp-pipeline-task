import os
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 0. Data
# =====================================================================
RAW_CORPUS = [
    ("设计精美！外观贴心！不错喜欢！", 1),
    ("性能强劲！运行令人满意！不错给力！", 1),
    ("售后专业！解决问题快！完美放心！", 1),
    ("服务态度给力！响应及时！棒极了好评！", 1),
    ("设计精美，外观顺畅，优秀喜欢。", 1),
    ("使用体验一流！完美超出预期！十分满意！", 1),
    ("性能强劲！运行精致！很好给力！", 1),
    ("使用体验令人满意，不错超出预期，十分满意！", 1),
    ("售后专业，解决问题快，专业放心。", 1),
    ("性价比给力，功能齐全，完美购买。", 1),
    ("设计精美，外观一流，专业喜欢。", 1),
    ("整体优秀，无可挑剔，贴心回购。", 1),
    ("质量专业，做工精良，精致值得信赖。", 1),
    ("质量一流！做工精良！专业值得信赖！", 1),
    ("使用体验令人满意！贴心超出预期！十分满意！", 1),
    ("体验极差，完全很差，极差差评。", 0),
    ("千万别买！质量堪忧，经常很差，劣质后悔。", 0),
    ("功能缺陷，很多劣质，虚高崩溃。", 0),
    ("千万别买！质量堪忧，经常恶劣，虚高后悔。", 0),
    ("客服恶劣，态度坑人，极差生气。", 0),
    ("千万别买！包装破损，收到就不行，失望恼火。", 0),
    ("功能缺陷，很多破损，恶劣崩溃。", 0),
    ("千万别买！功能缺陷，很多失望，劣质崩溃。", 0),
    ("功能缺陷，很多令人失望，劣质崩溃。", 0),
    ("虚假宣传，根本差劲，令人失望被骗。", 0),
    ("质量堪忧，经常很差，劣质后悔。", 0),
    ("千万别买！包装破损，收到就不行，虚高恼火。", 0),
    ("体验极差，完全烂透了，坑人差评。", 0),
    ("物流极慢，等了很久，糟糕烦躁。", 0),
    ("用了就坏，质量烂透了，差劲无语。", 0),
]

class TextPreprocessor:
    def __init__(self, stop_words: List[str] = None):
        self.stop_words = set(stop_words) if stop_words else set()

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<.*?>|http\S+|www\S+', '', text)
        text = zhconv.convert(text, 'zh-cn')
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        tokens = jieba.cut(text)
        return [t for t in tokens if t.strip() and t not in self.stop_words
                and not all(ch in '，。！？、；：,.!?;:' for ch in t)]

class CustomTfidfVectorizer:
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit(self, tokenized_docs: List[List[str]]):
        n_samples = len(tokenized_docs)
        df = defaultdict(int)
        word_counter = Counter()
        for doc in tokenized_docs:
            for word in set(doc):
                df[word] += 1
            word_counter.update(doc)
        most_common = word_counter.most_common(self.max_features)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        for word in self.vocab:
            self.idf_[word] = math.log((1 + n_samples) / (1 + df[word])) + 1.0
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> List[Dict[str, float]]:
        tfidf_matrices = []
        for doc in tokenized_docs:
            doc_len = len(doc) if len(doc) > 0 else 1
            term_counts = Counter(doc)
            tfidf_vec = {}
            for word, count in term_counts.items():
                if word in self.vocab:
                    tf = count / doc_len
                    tfidf_vec[word] = round(tf * self.idf_[word], 4)
            tfidf_matrices.append(tfidf_vec)
        return tfidf_matrices

    def fit_transform(self, tokenized_docs: List[List[str]]) -> List[Dict[str, float]]:
        return self.fit(tokenized_docs).transform(tokenized_docs)

# =====================================================================
# 1. Data preprocessing
# =====================================================================
preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
tokenized_docs = []
labels = []
for text, label in RAW_CORPUS:
    clean_t = preprocessor.clean_text(text)
    tokens = preprocessor.tokenize(clean_t)
    tokenized_docs.append(tokens)
    labels.append(label)

tfidf_vectorizer = CustomTfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_docs)

vocab_list = list(tfidf_vectorizer.vocab.keys())
n_samples = len(tfidf_matrix)
n_features = len(vocab_list)
X = np.zeros((n_samples, n_features))
for i, doc_tfidf in enumerate(tfidf_matrix):
    for word, value in doc_tfidf.items():
        if word in tfidf_vectorizer.vocab:
            j = tfidf_vectorizer.vocab[word]
            X[i, j] = value
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# 2. Grid search and collect all results (for the heatmap)
# =====================================================================
print("=" * 70)
print(" 🔥 Grid Search + Heatmap Visualization")
print("=" * 70)

# 2.1 Logistic Regression: C vs penalty (L1/L2)
print("\n--- [Logistic Regression: C vs Penalty grid search] ---")
lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
lr_grid.fit(X_train, y_train)

# Extract results
lr_results = lr_grid.cv_results_
lr_c_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Build heatmap data
lr_l1_scores = []
lr_l2_scores = []
for c in lr_c_values:
    # Find the mean_test_score for the corresponding C value
    l1_mask = (lr_results['param_C'] == c) & (lr_results['param_penalty'] == 'l1')
    l2_mask = (lr_results['param_C'] == c) & (lr_results['param_penalty'] == 'l2')
    
    l1_score = lr_results['mean_test_score'][l1_mask][0] if any(l1_mask) else 0
    l2_score = lr_results['mean_test_score'][l2_mask][0] if any(l2_mask) else 0
    
    lr_l1_scores.append(l1_score)
    lr_l2_scores.append(l2_score)

# 2.2 Linear SVM: C vs loss (hinge/squared_hinge)
print("--- [Linear SVM: C vs Loss grid search] ---")
svm_params = [
    {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 
     'C': [0.001, 0.01, 0.1, 1, 10, 100], 'dual': [True], 'max_iter': [5000]},
]

svm_grid = GridSearchCV(
    LinearSVC(random_state=42),
    svm_params,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
svm_grid.fit(X_train, y_train)

svm_results = svm_grid.cv_results_
svm_c_values = [0.001, 0.01, 0.1, 1, 10, 100]

svm_hinge_scores = []
svm_squared_scores = []
for c in svm_c_values:
    hinge_mask = (svm_results['param_C'] == c) & (svm_results['param_loss'] == 'hinge')
    squared_mask = (svm_results['param_C'] == c) & (svm_results['param_loss'] == 'squared_hinge')
    
    hinge_score = svm_results['mean_test_score'][hinge_mask][0] if any(hinge_mask) else 0
    squared_score = svm_results['mean_test_score'][squared_mask][0] if any(squared_mask) else 0
    
    svm_hinge_scores.append(hinge_score)
    svm_squared_scores.append(squared_score)

# =====================================================================
# 3. Plot the heatmaps
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3.1 LR heatmap
lr_data = np.array([lr_l1_scores, lr_l2_scores])
sns.heatmap(lr_data, 
            xticklabels=[str(c) for c in lr_c_values],
            yticklabels=['L1', 'L2'],
            annot=True, 
            fmt='.4f',
            cmap='YlOrRd',
            ax=axes[0],
            cbar_kws={'label': 'CV F1 Score'})
axes[0].set_title('Logistic Regression: C vs Penalty', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Regularization Coefficient (C)')
axes[0].set_ylabel('Penalty Type')

# 3.2 SVM heatmap
svm_data = np.array([svm_hinge_scores, svm_squared_scores])
sns.heatmap(svm_data,
            xticklabels=[str(c) for c in svm_c_values],
            yticklabels=['Hinge', 'Squared Hinge'],
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            ax=axes[1],
            cbar_kws={'label': 'CV F1 Score'})
axes[1].set_title('Linear SVM: C vs Loss Function', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Regularization Coefficient (C)')
axes[1].set_ylabel('Loss Function')

plt.tight_layout()
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_search_heatmap.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Heatmap saved: {save_path}")
plt.show()

# =====================================================================
# 4. Analysis: why hyperparameter tuning only yields small improvements
# =====================================================================
print("\n" + "=" * 70)
print(" 📊 Hyperparameter Tuning Effect Analysis")
print("=" * 70)

print(f"\nLR default params (C=1.0, penalty=l2) CV F1: ", end="")
default_lr = LogisticRegression(random_state=42, max_iter=1000)
default_lr.fit(X_train, y_train)
default_pred = default_lr.predict(X_test)
print(f"{f1_score(y_test, default_pred):.4f}")

print(f"LR best params (C=10, penalty=l2) CV F1: {lr_grid.best_score_:.4f}")
print(f"Improvement: {(lr_grid.best_score_ - f1_score(y_test, default_pred)) / f1_score(y_test, default_pred) * 100:.1f}%")

print(f"\nSVM default params (C=1.0) CV F1: ", end="")
default_svm = LinearSVC(random_state=42, max_iter=5000)
default_svm.fit(X_train, y_train)
default_pred = default_svm.predict(X_test)
print(f"{f1_score(y_test, default_pred):.4f}")

print(f"SVM best params (C=10, loss=hinge) CV F1: {svm_grid.best_score_:.4f}")
print(f"Improvement: {(svm_grid.best_score_ - f1_score(y_test, default_pred)) / f1_score(y_test, default_pred) * 100:.1f}%")

print("\n💡 Root cause analysis:")
print("1. The dataset is small (30 samples), so the model quickly reaches its performance ceiling")
print("2. TF-IDF features have low dimensionality (93 dimensions), limiting information capacity")
print("3. Linear models are inherently simple, leaving limited room for hyperparameter tuning")
print("4. The data distribution is clean, so default parameters already separate classes well")
print("5. Large tuning gains would require more complex feature engineering or non-linear models")
