import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score, precision_score, 
                            recall_score, roc_auc_score, confusion_matrix)
import xgboost as xgb

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
# 1. 准备数据
# =====================================================================
def prepare_tfidf_data():
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []
    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    vectorizer = CustomTfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
    weighter = SentimentWeighter()
    tfidf_matrix = [weighter.apply_weight(vec) for vec in tfidf_matrix]

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

def prepare_seq_data():
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []
    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for doc in tokenized_docs:
        for word in doc:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    sequences = []
    for doc in tokenized_docs:
        seq = [word2idx.get(w, word2idx["<UNK>"]) for w in doc]
        sequences.append(seq)

    # 分层划分训练/验证/测试 (6:2:2)，保证各子集均含正负样本
    indices = list(range(len(sequences)))
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    return (train_idx, val_idx, test_idx, sequences, labels, word2idx)

# =====================================================================
# 2. 深度学习模型定义
# =====================================================================
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences) if sequences else 1
    padded = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq)
    return padded, torch.tensor(labels).unsqueeze(1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_classes=1, filter_sizes=[2,3,4], num_filters=32, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled.append(pooled_out)
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x))

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=16, num_layers=2, dropout=0.5):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout_fc(x)
        return self.sigmoid(self.fc2(x))

class BiLSTMFixed(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=16, dropout_rate=0.5, num_layers=2):
        super(BiLSTMFixed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = avg_pool + max_pool
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout_fc(x)
        return self.sigmoid(self.fc2(x))

# =====================================================================
# 3. 训练与评估
# =====================================================================
def train_torch_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                val_preds.extend((preds > 0.5).float().numpy())
                val_true.extend(batch_y.numpy())
        
        val_f1 = f1_score(val_true, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return model

def eval_torch_model(model, test_loader):
    model.eval()
    test_preds, test_probs, test_true = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            probs = model(batch_x)
            preds = (probs > 0.5).float()
            test_preds.extend(preds.numpy())
            test_probs.extend(probs.numpy())
            test_true.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_true = np.array(test_true).flatten()
    
    return {
        'precision': precision_score(test_true, test_preds),
        'recall': recall_score(test_true, test_preds),
        'f1': f1_score(test_true, test_preds),
        'auc': roc_auc_score(test_true, test_probs),
        'cm': confusion_matrix(test_true, test_preds)
    }

def eval_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    if not hasattr(model, 'predict_proba'):
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
    
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'cm': confusion_matrix(y_test, y_pred)
    }

# =====================================================================
# 4. 主流程
# =====================================================================
def main():
    print("=" * 70)
    print(" 📊 Phase 4 Task 3: All Models Summary & Confusion Matrices")
    print("=" * 70)

    # 准备数据
    X_train, X_test, y_train, y_test = prepare_tfidf_data()
    train_idx, val_idx, test_idx, sequences, labels, word2idx = prepare_seq_data()

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_seqs = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_loader = DataLoader(TextDataset(train_seqs, train_labels), batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TextDataset(val_seqs, val_labels), batch_size=4, collate_fn=collate_fn)
    test_loader = DataLoader(TextDataset(test_seqs, test_labels), batch_size=4, collate_fn=collate_fn)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # =====================================================================
    # 4.1 传统机器学习模型
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🤖 训练传统机器学习模型")
    print("=" * 70)

    # LR
    print("\n--- [LogisticRegression] ---")
    lr = LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=42)
    lr.fit(X_train, y_train)
    r = eval_sklearn_model(lr, X_test, y_test)
    r['model'] = 'LogisticRegression'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # SVM
    print("\n--- [LinearSVM] ---")
    svm = LinearSVC(C=10, penalty='l2', loss='hinge', max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)
    r = eval_sklearn_model(svm, X_test, y_test)
    r['model'] = 'LinearSVM'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # RF
    print("\n--- [RandomForest] ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    r = eval_sklearn_model(rf, X_test, y_test)
    r['model'] = 'RandomForest'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # XGBoost
    print("\n--- [XGBoost] ---")
    xgb_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=50, subsample=0.8, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    r = eval_sklearn_model(xgb_model, X_test, y_test)
    r['model'] = 'XGBoost'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # =====================================================================
    # 4.2 深度学习模型
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🧠 训练深度学习模型")
    print("=" * 70)
    criterion = nn.BCELoss()
    vocab_size = len(word2idx)

    # TextCNN
    print("\n--- [TextCNN] ---")
    cnn = TextCNN(vocab_size, embed_dim=32, num_filters=32, dropout=0.5)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01, weight_decay=1e-3)
    cnn = train_torch_model(cnn, train_loader, val_loader, criterion, optimizer)
    r = eval_torch_model(cnn, test_loader)
    r['model'] = 'TextCNN'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # BiLSTM+Attention
    print("\n--- [BiLSTM+Attention] ---")
    lstm_attn = BiLSTMAttention(vocab_size, embed_dim=16, hidden_dim=16, num_layers=2, dropout=0.5)
    optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=0.01, weight_decay=1e-3)
    lstm_attn = train_torch_model(lstm_attn, train_loader, val_loader, criterion, optimizer)
    r = eval_torch_model(lstm_attn, test_loader)
    r['model'] = 'BiLSTM+Attention'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # BiLSTM(Fixed)
    print("\n--- [BiLSTM(Fixed)] ---")
    lstm_fixed = BiLSTMFixed(vocab_size, embed_dim=16, hidden_dim=16, dropout_rate=0.5, num_layers=2)
    optimizer = torch.optim.Adam(lstm_fixed.parameters(), lr=0.01, weight_decay=1e-3)
    lstm_fixed = train_torch_model(lstm_fixed, train_loader, val_loader, criterion, optimizer)
    r = eval_torch_model(lstm_fixed, test_loader)
    r['model'] = 'BiLSTM(Fixed)'
    results.append(r)
    print(f"✅ F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # =====================================================================
    # 5. 汇总表 + 混淆矩阵
    # =====================================================================
    print("\n" + "=" * 70)
    print(" 🏆 全部模型性能汇总表")
    print("=" * 70)
    print(f"{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")

    print("\n" + "=" * 70)
    print(" 📋 各模型混淆矩阵")
    print("=" * 70)
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  TN={r['cm'][0][0]:<3} FP={r['cm'][0][1]:<3}")
        print(f"  FN={r['cm'][1][0]:<3} TP={r['cm'][1][1]:<3}")

    print("\n" + "=" * 70)
    print(" 🎉 任务三完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
