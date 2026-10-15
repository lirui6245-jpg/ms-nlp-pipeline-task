import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

# Import the needed utilities from sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =====================================================================
# 0. Simulated enterprise-grade business corpus (1=positive review, 0=negative review)
# Expanded to 30 samples to support 5-fold stratified cross validation
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
# 1. Text cleaning pipeline (carried over from Phase 1)
# =====================================================================
class TextPreprocessor:
    """Industrial-grade text preprocessor: handles Traditional/Simplified conversion, noise filtering and tokenization"""
    def __init__(self, stop_words: List[str] = None):
        self.stop_words = set(stop_words) if stop_words else set()

    def clean_text(self, text: str) -> str:
        # Remove HTML tags and URLs
        text = re.sub(r'<.*?>|http\S+|www\S+', '', text)
        # Traditional to Simplified Chinese
        text = zhconv.convert(text, 'zh-cn')
        # Regex filter: keep Chinese/English letters, digits and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：]', '', text)
        # Collapse extra whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        """Jieba tokenization with stopword and punctuation removal"""
        tokens = jieba.cut(text)
        return [t for t in tokens if t.strip() and t not in self.stop_words
                and not all(ch in '，。！？、；：,.!?;:' for ch in t)]


# =====================================================================
# 2. Hand-written TF-IDF feature extractor (carried over from Phase 1)
# =====================================================================
class CustomTfidfVectorizer:
    """
    Hand-written TF-IDF implementation, strictly following sklearn's smoothed IDF formula:
    idf(t) = log((1 + n_samples) / (1 + df(t))) + 1
    """
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
# 3. Phase 2: hyperparameter grid search and cross validation on hand-written TF-IDF features
# =====================================================================
def main():
    print("=" * 70)
    print(" 💼 Microsoft Internship: Phase 2 - Grid Search + Cross Validation")
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

    # 3.2 Hand-written TF-IDF feature extraction
    print("\n--- [Step 1: Hand-written TF-IDF feature extraction] ---")
    tfidf_vectorizer = CustomTfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_docs)
    print(f" -> Vocabulary size: {len(tfidf_vectorizer.vocab)}")
    print(f" -> Number of samples: {len(tfidf_matrix)}")

    # 3.3 Convert dictionary-format TF-IDF into a dense numpy matrix
    # Note: sklearn models need matrix input, so the sparse dict representation is converted to a standard matrix
    print("\n--- [Step 2: Feature matrix format conversion] ---")
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
    print(f" -> Feature matrix shape: {X.shape}")
    print(f" -> Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 3.4 Split train/test sets (stratified sampling to keep the class ratio consistent)
    print("\n--- [Step 3: Stratified train/test split (8:2)] ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f" -> Train set size: {len(X_train)} (positive: {sum(y_train==1)}, negative: {sum(y_train==0)})")
    print(f" -> Test set size: {len(X_test)} (positive: {sum(y_test==1)}, negative: {sum(y_test==0)})")

    # 3.5 Define the 5-fold stratified cross-validation strategy
    print("\n--- [Step 4: Set up 5-fold stratified cross validation] ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f" -> Cross-validation folds: 5")

    # 3.6 Logistic Regression grid search
    print("\n" + "=" * 70)
    print(" 🔍 Starting grid search: Logistic Regression")
    print("=" * 70)

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
        verbose=1
    )

    lr_grid.fit(X_train, y_train)
    print(f"\n✅ Best parameters: {lr_grid.best_params_}")
    print(f"✅ Best CV F1 score: {lr_grid.best_score_:.4f}")

    y_pred_lr = lr_grid.best_estimator_.predict(X_test)
    print(f"\n📊 Logistic Regression test set report:")
    print(classification_report(y_test, y_pred_lr, digits=4))

    # 3.7 Linear SVM grid search
    print("\n" + "=" * 70)
    print(" 🔍 Starting grid search: Linear SVM")
    print("=" * 70)

    svm_params = [
        {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'dual': [True], 'max_iter': [5000]},
        {'penalty': ['l1'], 'loss': ['squared_hinge'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'dual': [False], 'max_iter': [5000]}
    ]

    svm_grid = GridSearchCV(
        LinearSVC(random_state=42),
        svm_params,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        svm_grid.fit(X_train, y_train)
    print(f"\n✅ Best parameters: {svm_grid.best_params_}")
    print(f"✅ Best CV F1 score: {svm_grid.best_score_:.4f}")

    y_pred_svm = svm_grid.best_estimator_.predict(X_test)
    print(f"\n📊 Linear SVM test set report:")
    print(classification_report(y_test, y_pred_svm, digits=4))

    # 3.8 Result comparison
    print("\n" + "=" * 70)
    print(" 🏆 Final comparison results")
    print("=" * 70)
    print(f"{'Model':<20} {'Best CV F1':<15} {'Test F1':<15}")
    print("-" * 50)
    print(f"{'LogisticRegression':<20} {lr_grid.best_score_:<15.4f} {f1_score(y_test, y_pred_lr):<15.4f}")
    print(f"{'LinearSVM':<20} {svm_grid.best_score_:<15.4f} {f1_score(y_test, y_pred_svm):<15.4f}")

    print("\n" + "=" * 70)
    print(" 🎉 Phase 2 task complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
