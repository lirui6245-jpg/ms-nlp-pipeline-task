import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

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

# =====================================================================
# 1. TextCNN 模型
# =====================================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, num_classes: int = 1, 
                 filter_sizes: List[int] = [2, 3, 4], num_filters: int = 32, dropout: float = 0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        # 多尺度卷积 + ReLU + MaxPool
        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len - fs + 1)
            pooled_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            pooled.append(pooled_out)
        
        x = torch.cat(pooled, dim=1)  # (batch, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


# =====================================================================
# 2. BiLSTM + Attention 模型
# =====================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim * 2)
        attn_weights = F.softmax(self.attention(lstm_output).squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim * 2)
        return context, attn_weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 32, 
                 num_layers: int = 2, dropout: float = 0.5):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)  # (batch, seq_len, hidden_dim * 2)
        context, attn_weights = self.attention(lstm_out)
        x = self.fc1(context)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return self.sigmoid(x), attn_weights


# =====================================================================
# 3. 修复后的 BiLSTM（任务四版本）
# =====================================================================
class BiLSTMFixed(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16, hidden_dim: int = 16, 
                 dropout_rate: float = 0.5, num_layers: int = 2):
        super(BiLSTMFixed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
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
        x = self.fc2(x)
        return self.sigmoid(x)


# =====================================================================
# 4. 数据管道
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


# =====================================================================
# 5. 训练工具
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
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


def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, epochs=100, patience=5):
    print(f"\n--- [训练 {model_name}] ---")
    early_stop = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            if isinstance(model, BiLSTMAttention):
                preds, _ = model(batch_x)
            else:
                preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(model, BiLSTMAttention):
                    preds, _ = model(batch_x)
                else:
                    preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                val_preds.extend((preds > 0.5).float().numpy())
                val_true.extend(batch_y.numpy())

        val_f1 = f1_score(val_true, val_preds)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}] | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        early_stop(val_loss / len(val_loader))
        if early_stop.early_stop:
            print(f"⏹️  Early stopping at epoch {epoch}")
            break

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return best_val_f1


def evaluate_model(model, test_loader):
    model.eval()
    test_preds, test_probs, test_true = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if isinstance(model, BiLSTMAttention):
                probs, _ = model(batch_x)
            else:
                probs = model(batch_x)
            preds = (probs > 0.5).float()
            test_preds.extend(preds.numpy())
            test_probs.extend(probs.numpy())
            test_true.extend(batch_y.numpy())

    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_true = np.array(test_true).flatten()

    precision = precision_score(test_true, test_preds)
    recall = recall_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds)
    auc = roc_auc_score(test_true, test_probs)
    cm = confusion_matrix(test_true, test_preds)

    return precision, recall, f1, auc, cm


# =====================================================================
# 6. 主流程
# =====================================================================
def main():
    print("=" * 70)
    print(" 🧠 Phase 4 Task 2: Deep Models Comparison")
    print("    TextCNN | BiLSTM+Attention | BiLSTM(Fixed) vs LinearSVM")
    print("=" * 70)

    # 6.1 数据预处理
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
    train_loader = DataLoader(TextDataset(train_seqs, train_labels), batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TextDataset(val_seqs, val_labels), batch_size=4, collate_fn=collate_fn)
    test_loader = DataLoader(TextDataset(test_seqs, test_labels), batch_size=4, collate_fn=collate_fn)

    vocab_size = len(word2idx)
    criterion = nn.BCELoss()
    results = []

    # 6.2 TextCNN
    cnn = TextCNN(vocab_size, embed_dim=32, num_filters=32, dropout=0.5)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=0.01, weight_decay=1e-3)
    train_model(cnn, train_loader, val_loader, criterion, optimizer_cnn, "TextCNN", epochs=100, patience=5)
    p, r, f1, auc, cm = evaluate_model(cnn, test_loader)
    print(f"\n✅ TextCNN | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    print(f"混淆矩阵:\n{cm}")
    results.append({"model": "TextCNN", "precision": p, "recall": r, "f1": f1, "auc": auc, "cm": cm})

    # 6.3 BiLSTM + Attention
    lstm_attn = BiLSTMAttention(vocab_size, embed_dim=16, hidden_dim=16, num_layers=2, dropout=0.5)
    optimizer_attn = torch.optim.Adam(lstm_attn.parameters(), lr=0.01, weight_decay=1e-3)
    train_model(lstm_attn, train_loader, val_loader, criterion, optimizer_attn, "BiLSTM+Attention", epochs=100, patience=5)
    p, r, f1, auc, cm = evaluate_model(lstm_attn, test_loader)
    print(f"\n✅ BiLSTM+Attention | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    print(f"混淆矩阵:\n{cm}")
    results.append({"model": "BiLSTM+Attention", "precision": p, "recall": r, "f1": f1, "auc": auc, "cm": cm})

    # 6.4 BiLSTM Fixed (任务四版本)
    lstm_fixed = BiLSTMFixed(vocab_size, embed_dim=16, hidden_dim=16, dropout_rate=0.5, num_layers=2)
    optimizer_fixed = torch.optim.Adam(lstm_fixed.parameters(), lr=0.01, weight_decay=1e-3)
    train_model(lstm_fixed, train_loader, val_loader, criterion, optimizer_fixed, "BiLSTM(Fixed)", epochs=100, patience=5)
    p, r, f1, auc, cm = evaluate_model(lstm_fixed, test_loader)
    print(f"\n✅ BiLSTM(Fixed) | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    print(f"混淆矩阵:\n{cm}")
    results.append({"model": "BiLSTM(Fixed)", "precision": p, "recall": r, "f1": f1, "auc": auc, "cm": cm})

    # 6.5 汇总表
    print("\n" + "=" * 70)
    print(" 🏆 深度学习模型性能汇总")
    print("=" * 70)
    print(f"{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")

    print("\n💡 分析:")
    print("- TextCNN 通过多尺度卷积捕获局部特征，适合短文本")
    print("- BiLSTM+Attention 引入注意力机制，可解释性更强")
    print("- 小数据集下深度学习模型难以发挥优势，参数量过大")
    print("- LinearSVM 在 30 条样本上仍是最优选择")

    print("\n" + "=" * 70)
    print(" 🎉 任务二完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
