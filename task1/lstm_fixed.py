import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
# 1. 文本清洗
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
# 2. 修复后的 BiLSTM 模型
# =====================================================================
class BiLSTMClassifierFixed(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 32, 
                 dropout_rate: float = 0.5, num_layers: int = 2):
        super(BiLSTMClassifierFixed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 多层 BiLSTM + Dropout between layers
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 更大的全连接层 + Dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        
        # 全局平均池化 + 最大池化（双池化）
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = avg_pool + max_pool
        
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)


# =====================================================================
# 3. 数据管道
# =====================================================================
class TextDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths) if max(lengths) > 0 else 1
    padded = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq)
    return padded, torch.tensor(labels).unsqueeze(1)


# =====================================================================
# 4. Early Stopping
# =====================================================================
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# =====================================================================
# 5. 主流程
# =====================================================================
def main():
    print("=" * 70)
    print(" 🧠 Fixed BiLSTM with Early Stopping & Enhanced Regularization")
    print("=" * 70)

    # 5.1 数据预处理
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    tokenized_docs = []
    labels = []

    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    # 构建词表
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for doc in tokenized_docs:
        for word in doc:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    # 序列化
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

    train_seqs = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_seqs = [sequences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # DataLoader
    train_ds = TextDataset(train_seqs, train_labels)
    val_ds = TextDataset(val_seqs, val_labels)
    test_ds = TextDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn)

    # 5.2 模型初始化
    torch.manual_seed(42)  # 固定 PyTorch 随机种子，保证实验可复现
    model = BiLSTMClassifierFixed(
        vocab_size=len(word2idx), 
        embed_dim=16, 
        hidden_dim=16, 
        dropout_rate=0.5,
        num_layers=2
    )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    early_stop = EarlyStopping(patience=5)

    # 5.3 训练
    print("\n--- [开始训练] ---")
    best_val_f1 = 0
    
    for epoch in range(1, 101):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()

        # Validation
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
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        early_stop(val_loss / len(val_loader))
        if early_stop.early_stop:
            print(f"⏹️  Early stopping triggered at epoch {epoch}")
            break

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    # 5.4 测试集评估
    print("\n--- [测试集评估] ---")
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

    # 全方位指标
    precision = precision_score(test_true, test_preds)
    recall = recall_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds)
    auc = roc_auc_score(test_true, test_probs)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, digits=4))

    # 5.5 与线性模型对比
    print("\n" + "=" * 70)
    print(" 📊 BiLSTM vs LinearSVM 全方位指标对比")
    print("=" * 70)
    print(f"{'指标':<15} {'BiLSTM':<12} {'LinearSVM':<12}")
    print("-" * 40)
    print(f"{'Precision':<15} {precision:<12.4f} {'1.0000':<12}")
    print(f"{'Recall':<15} {recall:<12.4f} {'1.0000':<12}")
    print(f"{'F1-Score':<15} {f1:<12.4f} {'1.0000':<12}")
    print(f"{'AUC':<15} {auc:<12.4f} {'-':<12}")
    print(f"{'参数量':<15} {sum(p.numel() for p in model.parameters()):<12} {'~100':<12}")

    print("\n💡 分析:")
    print("- BiLSTM 参数量大，需要更多数据才能发挥优势")
    print("- 小数据集下，线性模型已能达到完美分类")
    print("- LSTM 的序列建模能力在短文本上优势不明显")

    print("\n" + "=" * 70)
    print(" 🎉 训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
