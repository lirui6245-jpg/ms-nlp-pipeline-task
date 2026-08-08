import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import zhconv
import jieba

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import numpy as np

# =====================================================================
# 0. 数据
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
# 1. 文本清洗管道
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
# 2. 手写 TF-IDF 支持 N-gram
# =====================================================================
class CustomTfidfVectorizerNGram:
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 1)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """生成 n-gram 特征"""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams

    def fit(self, tokenized_docs: List[List[str]]):
        n_samples = len(tokenized_docs)
        df = defaultdict(int)
        word_counter = Counter()
        
        for doc in tokenized_docs:
            ngrams = self._get_ngrams(doc)
            for ngram in set(ngrams):
                df[ngram] += 1
            word_counter.update(ngrams)

        most_common = word_counter.most_common(self.max_features)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

        for word in self.vocab:
            self.idf_[word] = math.log((1 + n_samples) / (1 + df[word])) + 1.0
        return self

    def transform(self, tokenized_docs: List[List[str]]) -> List[Dict[str, float]]:
        tfidf_matrices = []
        for doc in tokenized_docs:
            ngrams = self._get_ngrams(doc)
            doc_len = len(ngrams) if len(ngrams) > 0 else 1
            term_counts = Counter(ngrams)
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
# 3. 情感加权模块
# =====================================================================
class SentimentWeighter:
    """基于简单情感词典的 TF-IDF 加权"""
    def __init__(self):
        # 正面情感词
        self.positive_words = {
            '优秀', '完美', '给力', '精致', '顺畅', '专业', '贴心', '一流', 
            '令人满意', '好评', '喜欢', '值得信赖', '回购', '舒适', '丰富'
        }
        # 负面情感词
        self.negative_words = {
            '极差', '劣质', '差劲', '失望', '恶劣', '堪忧', '破损', '崩溃',
            '后悔', '恼火', '烦躁', '无语', '烂透', '坑人', '假货', '差评'
        }
        self.sentiment_boost = 1.5  # 情感词加权系数

    def apply_weight(self, tfidf_vec: Dict[str, float]) -> Dict[str, float]:
        """对 TF-IDF 向量应用情感加权"""
        weighted = {}
        for word, score in tfidf_vec.items():
            # 检查 unigram（去掉 n-gram 的下划线）
            unigrams = word.split('_')
            sentiment_factor = 1.0
            
            for ug in unigrams:
                if ug in self.positive_words:
                    sentiment_factor = self.sentiment_boost
                    break
                elif ug in self.negative_words:
                    sentiment_factor = self.sentiment_boost
                    break
            
            weighted[word] = round(score * sentiment_factor, 4)
        return weighted


# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    print("=" * 70)
    print(" 🧪 N-gram + Sentiment Weighting Feature Experiment")
    print("=" * 70)

    # 4.1 数据预处理
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []

    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    # 4.2 对比实验设置
    experiments = [
        {"name": "Uni-gram TF-IDF", "ngram": (1, 1), "sentiment": False},
        {"name": "Bi-gram TF-IDF", "ngram": (1, 2), "sentiment": False},
        {"name": "Uni-gram + Sentiment", "ngram": (1, 1), "sentiment": True},
        {"name": "Bi-gram + Sentiment", "ngram": (1, 2), "sentiment": True},
    ]

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for exp in experiments:
        print(f"\n{'='*70}")
        print(f" 🔍 实验: {exp['name']}")
        print(f"{'='*70}")

        # 特征提取
        vectorizer = CustomTfidfVectorizerNGram(max_features=500, ngram_range=exp['ngram'])
        tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
        print(f" -> 词表大小: {len(vectorizer.vocab)}")

        # 情感加权
        if exp['sentiment']:
            weighter = SentimentWeighter()
            tfidf_matrix = [weighter.apply_weight(vec) for vec in tfidf_matrix]
            print(f" -> 已应用情感加权 (boost={weighter.sentiment_boost})")

        # 转换为 numpy
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

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练最优模型 (LinearSVM with L2)
        svm = LinearSVC(C=10, penalty='l2', loss='hinge', max_iter=5000, random_state=42)
        svm.fit(X_train, y_train)
        
        # 交叉验证
        svm_cv = GridSearchCV(
            LinearSVC(random_state=42, max_iter=5000),
            {'C': [10], 'penalty': ['l2'], 'loss': ['hinge'], 'dual': [True]},
            cv=cv, scoring='f1', n_jobs=-1
        )
        svm_cv.fit(X_train, y_train)
        
        y_pred = svm.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)
        
        print(f"✅ CV F1: {svm_cv.best_score_:.4f}")
        print(f"✅ 测试集 F1: {test_f1:.4f}")
        
        results.append({
            'name': exp['name'],
            'vocab_size': len(vectorizer.vocab),
            'cv_f1': svm_cv.best_score_,
            'test_f1': test_f1
        })

    # 4.3 结果对比
    print("\n" + "=" * 70)
    print(" 🏆 特征工程实验结果对比")
    print("=" * 70)
    print(f"{'特征方案':<25} {'词表大小':<12} {'CV F1':<12} {'测试集 F1':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<25} {r['vocab_size']:<12} {r['cv_f1']:<12.4f} {r['test_f1']:<12.4f}")

    # 4.4 分析
    print("\n💡 分析:")
    print("1. Bi-gram 引入词序信息，可能捕获更多语义")
    print("2. 情感加权强化情感词特征，提升区分度")
    print("3. 小数据集下特征扩充效果有限，需更大语料验证")

    print("\n" + "=" * 70)
    print(" 🎉 实验完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
