import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import numpy as np

# =====================================================================
# 0. Data (using the best features: Uni-gram + Sentiment)
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

class SentimentWeighter:
    def __init__(self):
        self.positive_words = {'优秀', '完美', '给力', '精致', '顺畅', '专业', '贴心', '一流', '令人满意', '好评', '喜欢', '值得信赖', '回购', '舒适', '丰富'}
        self.negative_words = {'极差', '劣质', '差劲', '失望', '恶劣', '堪忧', '破损', '崩溃', '后悔', '恼火', '烦躁', '无语', '烂透', '坑人', '假货', '差评'}
        self.sentiment_boost = 1.5

    def apply_weight(self, tfidf_vec: Dict[str, float]) -> Dict[str, float]:
        weighted = {}
        for word, score in tfidf_vec.items():
            unigrams = word.split('_')
            sentiment_factor = 1.0
            for ug in unigrams:
                if ug in self.positive_words or ug in self.negative_words:
                    sentiment_factor = self.sentiment_boost
                    break
            weighted[word] = round(score * sentiment_factor, 4)
        return weighted

# =====================================================================
# 1. Data preprocessing
# =====================================================================
def prepare_data():
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []

    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    # TF-IDF + Sentiment
    vectorizer = CustomTfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

    weighter = SentimentWeighter()
    tfidf_matrix = [weighter.apply_weight(vec) for vec in tfidf_matrix]

    # Convert to numpy
    vocab_list = list(vectorizer.vocab.keys())
    n_samples = len(tfidf_matrix)
    n_features = len(vocab_list)
    X = np.zeros((n_samples, n_features))
    for i, doc_tfidf in enumerate(tfidf_matrix):
        for word, value in doc_tfidf.items():
            if word in vectorizer.vocab:
                j = vectorizer.vocab[word]
                X[i, j] = value
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# 2. Model training and evaluation
# =====================================================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    # Normalize decision_function to [0,1] for AUC
    if not hasattr(model, 'predict_proba'):
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f" 📊 {model_name}")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"\nConfusion matrix:")
    print(cm)
    
    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'cm': cm
    }

def main():
    print("=" * 70)
    print(" 🌲 Phase 4 Task 1: Tree Models vs Linear Models")
    print("=" * 70)

    X_train, X_test, y_train, y_test = prepare_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    # 2.1 Logistic Regression (best parameters)
    print("\n--- [Training LogisticRegression] ---")
    lr = LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=42)
    results.append(evaluate_model(lr, X_train, X_test, y_train, y_test, "LogisticRegression"))

    # 2.2 Linear SVM (best parameters)
    print("\n--- [Training LinearSVM] ---")
    svm = LinearSVC(C=10, penalty='l2', loss='hinge', max_iter=5000, random_state=42)
    results.append(evaluate_model(svm, X_train, X_test, y_train, y_test, "LinearSVM"))

    # 2.3 Random Forest
    print("\n--- [Training RandomForest] ---")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=cv, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"✅ RF best parameters: {rf_grid.best_params_}")
    results.append(evaluate_model(rf_grid.best_estimator_, X_train, X_test, y_train, y_test, "RandomForest"))

    # 2.4 XGBoost
    print("\n--- [Training XGBoost] ---")
    xgb_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100],
        'subsample': [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='logloss'), 
                            xgb_params, cv=cv, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    print(f"✅ XGBoost best parameters: {xgb_grid.best_params_}")
    results.append(evaluate_model(xgb_grid.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost"))

    # =====================================================================
    # 3. Summary table
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🏆 Model Performance Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")

    print("\n" + "=" * 70)
    print(" 🎉 Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
