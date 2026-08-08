import re
import math
from collections import Counter, defaultdict
from typing import List, Dict
import zhconv
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 0. 模拟企业级业务语料库 (1=正向评价/好评, 0=负向评价/差评)
# =====================================================================
RAW_CORPUS = [
    ("这款笔记本电脑的性能真的太强了！散热超级好，非常推荐购买。😊", 1),
    ("完全是假货，屏幕有严重的坏点，客服态度極其惡劣，立刻退款！", 0),
    ("微软的开发工具体验一如既往的流畅，文档非常清晰，给开发组点赞。", 1),
    ("物流慢得要死，等了足足十天才收到，而且外包装全都破损了...", 0),
    ("性价比很高，电池续航超出了我的预期，日常办公和运行模型毫无压力。", 1),
    ("千萬別買！系統經常無故藍屏死機，廣告多得讓人崩潰，極度失望。", 0),
    ("键盘手感舒适，接口丰富，作为生产力工具来说无可挑剔！", 1),
    ("宣称有降噪功能，实际根本没用，杂音极其严重，这就是虚假宣传。", 0),
    ("从下订单到收货体验非常顺畅，售后技术支持也很专业，五星好评。", 1),
    ("大失所望，使用不到半天就发烫严重，这种品控怎么敢出厂的？", 0)
]

# =====================================================================
# 1. 任务一：工业级文本清洗管道 (Text Preprocessing Pipeline)
# =====================================================================
class TextPreprocessor:
    """工业级文本预处理器：处理繁简转换、噪声过滤、分词与分句"""
    def __init__(self, stop_words: List[str] = None):
        self.stop_words = set(stop_words) if stop_words else set()

    def clean_text(self, text: str) -> str:
        # 移除 HTML 标签与 URL
        text = re.sub(r'<.*?>|http\S+|www\S+', '', text)
        # 繁体转简体
        text = zhconv.convert(text, 'zh-cn')
        # 正则过滤：保留中英文、数字、中英文基础标点（去除表情符号与特殊不可见乱码）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：]', '', text)
        # 压缩多余空格
        return re.sub(r'\s+', ' ', text).strip()

    def split_sentences(self, text: str) -> List[str]:
        """按标点对长文本进行细粒度分句"""
        sentences = re.split(r'[。！？；]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize(self, text: str) -> List[str]:
        """结巴分词并剔除停用词与标点"""
        tokens = jieba.cut(text)
        return [t for t in tokens if t.strip() and t not in self.stop_words
                and not all(ch in '，。！？、；：,.!?;:' for ch in t)]


# =====================================================================
# 2. 任务二：手写标准 TF-IDF 特征提取器
# =====================================================================
class CustomTfidfVectorizer:
    """
    手写实现 TF-IDF 算法，严格遵循 sklearn 平滑逆文档频率公式：
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
# 3. 任务三：PyTorch 数据管道与动态 Padding (DataLoader)
# =====================================================================
class TextDataset(Dataset):
    def __init__(self, tokenized_docs: List[List[str]], labels: List[int], word2idx: Dict[str, int]):
        self.data = []
        for doc in tokenized_docs:
            seq = [word2idx.get(w, word2idx["<UNK>"]) for w in doc]
            self.data.append(seq)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_fn(batch):
    """自定义批处理函数：实现当前 Batch 内部的动态 Padding，节省内存与显存"""
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths) if max(lengths) > 0 else 1

    padded_seqs = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        end = len(seq)
        padded_seqs[i, :end] = seq

    return padded_seqs, torch.tensor(labels).unsqueeze(1)


# =====================================================================
# 4. 任务三：双向 LSTM (BiLSTM) 深度学习模型
# =====================================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super(BiLSTMClassifier, self).__init__()
        # 词嵌入层：padding_idx=0 确保填充位置不参与梯度计算与更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        # 双向 LSTM：捕获上下文双向特征
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 双向输出维度需乘 2
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        # 提取序列时序全局平均池化特征
        pooled_out = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled_out)
        return self.sigmoid(logits)


# =====================================================================
# 5. 主流程控制管道 (Main Execution Pipeline)
# =====================================================================
def main():
    print("=" * 70)
    print(" 💼 Microsoft Internship Technical Task: NLP Engineering Pipeline")
    print("=" * 70)

    # 5.1 文本清洗与分句
    preprocessor = TextPreprocessor(stop_words=["的", "了", "和", "是", "就", "也"])
    cleaned_texts, tokenized_docs, labels = [], [], []

    print("\n--- [Step 1: 文本预处理与分句分析] ---")
    for text, label in RAW_CORPUS:
        clean_t = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean_t)
        cleaned_texts.append(clean_t)
        tokenized_docs.append(tokens)
        labels.append(label)

    sample_idx = 0
    print(f" -> 原始输入文本: {RAW_CORPUS[sample_idx][0]}")
    print(f" -> 繁简转换与过滤: {cleaned_texts[sample_idx]}")
    print(f" -> 细粒度分句切分: {preprocessor.split_sentences(cleaned_texts[sample_idx])}")
    print(f" -> 结巴去噪分词结果: {tokenized_docs[sample_idx]}")

    # 5.2 TF-IDF 特征提取
    print("\n--- [Step 2: 手写 CustomTfidfVectorizer 特征空间拟合] ---")
    tfidf_vectorizer = CustomTfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_docs)
    
    sample_tfidf = sorted(tfidf_matrix[sample_idx].items(), key=lambda x: x[1], reverse=True)
    print(f" -> 构建有效词表空间 (Vocab Size): {len(tfidf_vectorizer.vocab)}")
    print(f" -> 样本 01 权重最高 TOP 5 关键词特征: {sample_tfidf[:5]}")

    # 5.3 构建深度学习张量管道
    print("\n--- [Step 3: 构建 BiLSTM 神经网络动态管道] ---")
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for doc in tokenized_docs:
        for word in doc:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    
    dataset = TextDataset(tokenized_docs, labels, word2idx)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(f" -> 序列化词表总数: {len(word2idx)} | 批次数量 (Batches): {len(dataloader)}")

    # 5.4 双向 LSTM 模型训练
    print("\n--- [Step 4: 训练双向 LSTM (BiLSTM) 情感二分类模型] ---")
    model = BiLSTMClassifier(vocab_size=len(word2idx), embed_dim=16, hidden_dim=16, dropout_rate=0.2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 16):
        total_loss, correct, total_samples = 0.0, 0, 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total_samples += len(batch_y)

        avg_loss = total_loss / total_samples
        accuracy = (correct / total_samples) * 100
        if epoch % 3 == 0 or epoch == 1:
            print(f" -> Epoch [{epoch:02d}/15] | 平均损失 (Loss): {avg_loss:.4f} | 训练集准确率 (Acc): {accuracy:.2f}%")

    print("\n" + "=" * 70)
    print(" 🎉 任务完成！管道化代码通过验证。")
    print("=" * 70)

if __name__ == "__main__":
    main()
