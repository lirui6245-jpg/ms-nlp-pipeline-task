import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import numpy as np

# =====================================================================
# 0. Simulated enterprise-grade business corpus (expanded to 30 samples)
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

# =====================================================================
# 1. Text cleaning pipeline
# =====================================================================
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


# =====================================================================
# 2. Hand-written TF-IDF feature extractor
# =====================================================================
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
# 3. Ablation study: L1 / L2 / Elastic Net regularization comparison
# =====================================================================
def main():
    print("=" * 70)
    print(" 🔬 Ablation Study: L1 / L2 / Elastic Net Regularization")
    print("=" * 70)

    # 3.1 Data preprocessing
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []

    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    # 3.2 TF-IDF feature extraction
    print("\n--- [Step 1: Hand-written TF-IDF feature extraction] ---")
    tfidf_vectorizer = CustomTfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_docs)
    print(f" -> Vocabulary size: {len(tfidf_vectorizer.vocab)}")

    # Convert to a numpy matrix
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

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =====================================================================
    # 3.3 Logistic Regression ablation study
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 📊 Logistic Regression: L1 / L2 / Elastic Net ablation study")
    print("=" * 70)

    lr_configs = [
        {"name": "L1 Regularization", "params": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear'], 'max_iter': [1000]}},
        {"name": "L2 Regularization", "params": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear'], 'max_iter': [1000]}},
        {"name": "ElasticNet", "params": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.3, 0.5, 0.7], 'max_iter': [2000]}},
    ]

    lr_results = []
    for config in lr_configs:
        print(f"\n--- Training: {config['name']} ---")
        grid = GridSearchCV(
            LogisticRegression(random_state=42),
            config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        
        y_pred = grid.best_estimator_.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        print(f"✅ Best parameters: {grid.best_params_}")
        print(f"✅ CV F1: {grid.best_score_:.4f} | Test F1: {test_f1:.4f}")
        
        lr_results.append({
            'name': config['name'],
            'best_params': grid.best_params_,
            'cv_f1': grid.best_score_,
            'test_f1': test_f1,
            'model': grid.best_estimator_
        })

    # =====================================================================
    # 3.4 Linear SVM ablation study (L1 / L2)
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 📊 Linear SVM: L1 / L2 ablation study")
    print("=" * 70)

    svm_configs = [
        {"name": "L1 Regularization", "params": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'max_iter': [5000]}},
        {"name": "L2 Regularization", "params": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'dual': [True], 'max_iter': [5000]}},
    ]

    svm_results = []
    for config in svm_configs:
        print(f"\n--- Training: {config['name']} ---")
        grid = GridSearchCV(
            LinearSVC(random_state=42),
            config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        
        y_pred = grid.best_estimator_.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        print(f"✅ Best parameters: {grid.best_params_}")
        print(f"✅ CV F1: {grid.best_score_:.4f} | Test F1: {test_f1:.4f}")
        
        svm_results.append({
            'name': config['name'],
            'best_params': grid.best_params_,
            'cv_f1': grid.best_score_,
            'test_f1': test_f1,
            'model': grid.best_estimator_
        })

    # =====================================================================
    # 3.5 Output keyword weight comparison
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🔑 Keyword weight comparison (Top 10)")
    print("=" * 70)

    def print_top_weights(model, model_name, vocab_list):
        if hasattr(model, 'coef_'):
            weights = model.coef_[0]
            word_weights = [(vocab_list[i], weights[i]) for i in range(len(vocab_list))]
            word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n{model_name}:")
            for word, weight in word_weights[:10]:
                print(f"  {word:<12} {weight:>8.4f}")

    for r in lr_results:
        print_top_weights(r['model'], f"LR-{r['name']}", vocab_list)

    for r in svm_results:
        print_top_weights(r['model'], f"SVM-{r['name']}", vocab_list)

    # =====================================================================
    # 3.6 Result summary
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🏆 Ablation study result summary")
    print("=" * 70)
    print(f"{'Model':<25} {'CV F1':<12} {'Test F1':<12}")
    print("-" * 50)
    for r in lr_results:
        print(f"{'LR-' + r['name']:<25} {r['cv_f1']:<12.4f} {r['test_f1']:<12.4f}")
    for r in svm_results:
        print(f"{'SVM-' + r['name']:<25} {r['cv_f1']:<12.4f} {r['test_f1']:<12.4f}")

    print("\n" + "=" * 70)
    print(" 🎉 Ablation study complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
