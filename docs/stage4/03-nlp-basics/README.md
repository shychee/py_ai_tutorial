# 模块M03: 自然语言处理基础

**阶段**: Stage 4 - 深度学习
**预计学习时间**: 3-4小时（理论）+ 3-4小时（实践）
**难度**: ⭐⭐⭐⭐ 中高等

---

## 📚 学习目标

完成本模块后，你将能够：

- ✅ 理解词嵌入技术（Word2Vec、GloVe、FastText）的原理与应用
- ✅ 掌握循环神经网络（RNN、LSTM、GRU）处理序列数据的机制
- ✅ 深入理解Transformer架构（Self-Attention、Multi-Head Attention）
- ✅ 熟悉预训练模型（BERT、GPT、T5）的原理与微调方法
- ✅ 能够完成文本分类、命名实体识别、机器翻译等NLP任务
- ✅ 掌握使用Hugging Face Transformers库进行模型微调

---

## 🎯 核心知识点

### 1. 词嵌入 (Word Embeddings)

#### 1.1 为什么需要词嵌入？

**传统表示方法的问题**：

**One-Hot编码**:
```python
vocab = ["king", "queen", "man", "woman", "apple"]
"king"  = [1, 0, 0, 0, 0]
"queen" = [0, 1, 0, 0, 0]
```

**缺点**：
- 维度灾难（词汇量10万 → 10万维向量）
- 无法表示词语之间的语义关系
- 词向量正交（余弦相似度=0）

**词嵌入的优势**:
```python
# 词嵌入将词映射到低维稠密向量（如300维）
"king"  = [0.50, 0.33, ..., -0.21]  # 300维
"queen" = [0.48, 0.31, ..., -0.19]  # 语义相近

# 可以进行向量运算
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

#### 1.2 Word2Vec

**两种训练方式**：

**1) CBOW (Continuous Bag of Words)**:
```
上下文: [the, cat, on, the] → 预测中心词: "sat"
```

**2) Skip-gram**:
```
中心词: "sat" → 预测上下文: [the, cat, on, the]
```

**网络结构**（Skip-gram）:
```
输入层(one-hot) → 隐藏层(embedding) → 输出层(softmax)
    10000维           300维             10000维
```

**训练技巧**：
- **负采样 (Negative Sampling)**: 不计算所有10000个词的softmax，只计算1个正样本+k个负样本（k=5-20）
- **层次Softmax (Hierarchical Softmax)**: 使用二叉树结构，复杂度从O(V)降到O(log V)

**代码示例** (使用Gensim):
```python
from gensim.models import Word2Vec

# 训练Word2Vec模型
sentences = [["I", "love", "NLP"], ["Deep", "learning", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# 查询相似词
similar_words = model.wv.most_similar("love", topn=5)
print(similar_words)

# 向量运算
result = model.wv.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=1
)
print(result)  # 输出: [('queen', 0.87)]
```

#### 1.3 GloVe (Global Vectors)

**核心思想**: 结合全局统计信息（词共现矩阵）与局部上下文窗口。

**训练目标**:
```
J = Σ f(X_ij) · (w_i^T · w_j + b_i + b_j - log(X_ij))²
```
其中：
- `X_ij`: 词i和词j的共现次数
- `w_i, w_j`: 词向量
- `f(X_ij)`: 权重函数（减少高频词影响）

**GloVe vs Word2Vec**:
| 特性 | Word2Vec | GloVe |
|------|----------|-------|
| 训练方式 | 局部上下文窗口 | 全局共现矩阵 |
| 训练速度 | 较快 | 较慢（需统计共现矩阵） |
| 性能 | 略低 | 略高 |
| 适用场景 | 大规模语料 | 中小规模语料 |

#### 1.4 FastText

**核心创新**: 考虑**子词信息 (Subword Information)**

**示例**:
```
Word2Vec: "apple" → [0.2, 0.3, ...]
FastText:  "apple" → <ap, app, ppl, ple, le> 的平均
```

**优势**:
- 处理**未登录词 (OOV, Out-of-Vocabulary)**:
  ```python
  # Word2Vec无法处理
  "apples" (未见过) → ❌ 无向量

  # FastText可以组合子词
  "apples" → <ap, app, ppl, ple, les, es> → ✅ 有向量
  ```
- 对**形态丰富的语言**（如德语、土耳其语）效果更好

**代码示例**:
```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5, min_count=1)

# 处理未登录词
oov_vector = model.wv['unknownword']  # 可以生成向量
```

---

### 2. 循环神经网络 (RNN)

#### 2.1 为什么需要RNN？

**问题**: 传统神经网络无法处理变长序列。

**RNN的优势**:
- 共享参数（不同时间步使用相同权重）
- 保持历史信息（隐藏状态记忆）
- 可处理任意长度序列

**RNN结构**:
```
     y_t (输出)
      ↑
     h_t (隐藏状态)
    ↗  ↖
  h_{t-1}  x_t (输入)
```

**数学公式**:
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

**代码示例** (PyTorch):
```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后时间步
        return out
```

#### 2.2 RNN的问题：梯度消失/爆炸

**梯度消失**:
```
∂L/∂h_1 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_2/∂h_1
         = ∂L/∂h_T · W^{T-1}
```

如果 `W < 1`，连乘T次后梯度→0（长期依赖消失）
如果 `W > 1`，连乘T次后梯度→∞（梯度爆炸）

**解决方案**:
- **梯度裁剪 (Gradient Clipping)**: 限制梯度最大值
- **更好的激活函数**: ReLU替代tanh
- **门控机制**: LSTM、GRU

#### 2.3 LSTM (Long Short-Term Memory)

**核心思想**: 引入**记忆细胞 (Cell State)** 和**三个门控单元**。

**LSTM结构**:
```
    输入门     遗忘门     输出门
     i_t       f_t       o_t
      ↓         ↓         ↓
  ┌────────────────────────┐
  │   Cell State (C_t)     │
  └────────────────────────┘
```

**数学公式**:
```python
# 遗忘门: 决定丢弃多少旧记忆
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

# 输入门: 决定添加多少新信息
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# 更新记忆细胞
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

# 输出门: 决定输出多少信息
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**直观理解**:
- **遗忘门**: "忘记不重要的信息"
- **输入门**: "记住新的重要信息"
- **输出门**: "输出当前需要的信息"

**代码示例**:
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 取最后隐藏状态
        out = self.fc(hidden[-1])  # (batch, num_classes)
        return out
```

#### 2.4 GRU (Gated Recurrent Unit)

**简化版LSTM**: 合并记忆细胞和隐藏状态，只有2个门。

**数学公式**:
```python
# 重置门
r_t = σ(W_r · [h_{t-1}, x_t])

# 更新门
z_t = σ(W_z · [h_{t-1}, x_t])

# 候选隐藏状态
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])

# 最终隐藏状态
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**GRU vs LSTM**:
| 特性 | LSTM | GRU |
|------|------|-----|
| 参数量 | 更多（4个门） | 更少（2个门） |
| 训练速度 | 较慢 | 较快 |
| 性能 | 略高（大数据集） | 略低 |
| 适用场景 | 复杂任务 | 简单任务、资源受限 |

---

### 3. Transformer架构

#### 3.1 为什么需要Transformer？

**RNN/LSTM的局限**:
- ❌ 无法并行化（必须按时间步顺序计算）
- ❌ 长期依赖问题（虽然LSTM缓解了，但未彻底解决）
- ❌ 训练慢（尤其是长序列）

**Transformer的优势**:
- ✅ 完全并行化（所有位置同时计算）
- ✅ 长距离依赖直接建模（Self-Attention）
- ✅ 可扩展性强（适合大规模预训练）

#### 3.2 Self-Attention机制

**核心思想**: 计算序列中每个词与其他所有词的关联程度。

**计算步骤**:

**1) 生成Q, K, V矩阵**:
```
Query  = X · W_Q  # (seq_len, d_model) × (d_model, d_k)
Key    = X · W_K
Value  = X · W_V
```

**2) 计算注意力分数**:
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**直观理解**:
```
输入句子: "The cat sat on the mat"

对于单词"cat":
- Q_cat 与所有 K 计算相似度
- 得到注意力分数: [0.1, 0.6, 0.2, 0.05, 0.05]
               (The, cat, sat, on,  the)
- 对 V 加权求和得到 cat 的新表示
```

**可视化**:
```
     The   cat   sat   on    the   mat
The  0.5   0.2   0.1   0.1   0.05  0.05
cat  0.1   0.6   0.2   0.05  0.05  0.0
sat  0.1   0.3   0.4   0.1   0.05  0.05
...
```

**代码实现**:
```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```

#### 3.3 Multi-Head Attention

**核心思想**: 多个注意力头并行学习不同的特征子空间。

**公式**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

其中 head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

**优势**:
- 不同头关注不同的语义信息
  - Head 1: 语法关系（主谓宾）
  - Head 2: 语义关系（同义词）
  - Head 3: 位置关系（相邻词）

**示例**（8个头）:
```
Head 1: "cat" 关注 "sat" (动作关系)
Head 2: "cat" 关注 "the" (修饰关系)
Head 3: "cat" 关注 "mat" (位置关系)
...
```

**代码实现**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 分割成多个头
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力（并行计算所有头）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 合并多个头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(output)
```

#### 3.4 位置编码 (Positional Encoding)

**问题**: Self-Attention对词序不敏感（"cat sat" 和 "sat cat" 结果相同）

**解决方案**: 添加位置信息

**正弦位置编码**:
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**代码实现**:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

#### 3.5 完整Transformer架构

**Encoder-Decoder结构**:
```
输入序列 → Encoder (N×) → Decoder (N×) → 输出序列
```

**Encoder Block**:
```
输入
  ↓
Multi-Head Attention → Add & Norm
  ↓
Feed Forward → Add & Norm
  ↓
输出
```

**Decoder Block**:
```
输入
  ↓
Masked Multi-Head Attention → Add & Norm
  ↓
Cross-Attention (with Encoder) → Add & Norm
  ↓
Feed Forward → Add & Norm
  ↓
输出
```

**关键组件**:
- **残差连接 (Residual Connection)**: 缓解梯度消失
- **Layer Normalization**: 稳定训练
- **Feed Forward Network**: 2层全连接 + ReLU
- **Masked Attention**: Decoder中防止看到未来信息

---

### 4. 预训练模型

#### 4.1 预训练 + 微调范式

**传统方法**（从零训练）:
```
标注数据(少) → 训练模型 → 性能一般
```

**预训练 + 微调**:
```
大规模无标注数据 → 预训练 → 通用语言模型
         ↓
  小规模标注数据 → 微调 → 特定任务模型 (高性能)
```

#### 4.2 BERT (Bidirectional Encoder Representations from Transformers)

**核心思想**: 双向编码器，使用**Masked Language Model**预训练。

**预训练任务**:

**1) Masked Language Model (MLM)**:
```
输入: "The [MASK] sat on the mat"
目标: 预测 [MASK] = "cat"
```

**2) Next Sentence Prediction (NSP)**:
```
输入: [CLS] Sentence A [SEP] Sentence B [SEP]
目标: 判断B是否是A的下一句
```

**模型架构**:
```
输入: [CLS] token1 token2 ... tokenN [SEP]
  ↓
Transformer Encoder (12层 or 24层)
  ↓
输出: [CLS]表示 + token表示
```

**BERT家族**:
| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|---------|---------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |
| RoBERTa | 24 | 1024 | 16 | 355M (优化版BERT) |
| ALBERT | 12 | 768 | 12 | 12M (参数共享) |

**应用场景**:
- 文本分类
- 命名实体识别 (NER)
- 问答系统
- 语义相似度

**微调示例** (使用Hugging Face):
```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备数据
texts = ["I love NLP", "This is terrible"]
labels = [1, 0]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs['labels'] = torch.tensor(labels)

# 微调
trainer = Trainer(model=model, train_dataset=inputs)
trainer.train()
```

#### 4.3 GPT (Generative Pre-trained Transformer)

**核心思想**: 单向解码器，使用**自回归语言模型**预训练。

**预训练任务**: **Causal Language Modeling**
```
输入: "The cat sat"
目标: 预测下一个词 "on"
```

**模型架构**:
```
输入: token1 token2 ... tokenN
  ↓
Transformer Decoder (使用Masked Self-Attention)
  ↓
输出: 预测下一个token的概率分布
```

**GPT演进**:
| 模型 | 发布年份 | 层数 | 参数量 | 关键特性 |
|------|---------|------|--------|---------|
| GPT-1 | 2018 | 12 | 117M | 首次提出预训练+微调 |
| GPT-2 | 2019 | 48 | 1.5B | Zero-shot学习能力 |
| GPT-3 | 2020 | 96 | 175B | Few-shot in-context learning |
| GPT-4 | 2023 | ? | >1T | 多模态、更强推理能力 |

**GPT vs BERT**:
| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 注意力方向 | 双向 | 单向（因果） |
| 预训练任务 | MLM + NSP | 自回归LM |
| 适用任务 | 理解类（分类、NER） | 生成类（文本生成、对话） |

#### 4.4 T5 (Text-to-Text Transfer Transformer)

**核心思想**: 将所有NLP任务统一为**文本到文本**转换。

**任务统一格式**:
```python
# 翻译
"translate English to German: Hello" → "Hallo"

# 分类
"sentiment: This movie is great!" → "positive"

# 摘要
"summarize: [长文本]" → "简短摘要"

# 问答
"question: What is NLP? context: ..." → "答案"
```

**优势**:
- 统一的输入输出格式
- 一个模型处理多个任务
- 可以轻松添加新任务

**T5家族**:
- T5-Small: 60M参数
- T5-Base: 220M参数
- T5-Large: 770M参数
- T5-3B: 3B参数
- T5-11B: 11B参数

---

### 5. 下游任务与微调

#### 5.1 文本分类 (Text Classification)

**任务**: 将文本分配到预定义类别

**应用场景**:
- 情感分析（正面/负面）
- 垃圾邮件检测
- 新闻分类

**微调策略**:
```python
# 使用[CLS] token的表示进行分类
[CLS] token1 token2 ... [SEP]
  ↓
BERT Encoder
  ↓
[CLS]表示 → Linear → Softmax → 类别概率
```

#### 5.2 命名实体识别 (NER)

**任务**: 识别文本中的实体（人名、地名、组织名等）

**标注格式** (BIO):
```
I      B-PER  (Begin-Person)
love   O      (Outside)
New    B-LOC  (Begin-Location)
York   I-LOC  (Inside-Location)
```

**微调策略**:
```python
# 对每个token进行分类
token1 token2 ... tokenN
  ↓         ↓         ↓
BERT Encoder
  ↓         ↓         ↓
Linear   Linear    Linear
  ↓         ↓         ↓
B-PER     O        B-LOC
```

**代码示例**:
```python
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_list)  # B-PER, I-PER, B-LOC, ...
)
```

#### 5.3 机器翻译 (Machine Translation)

**任务**: 将源语言文本翻译成目标语言

**Transformer架构**:
```
源语言输入 → Encoder → Decoder → 目标语言输出
```

**训练策略**:
- **Teacher Forcing**: 训练时使用真实目标作为Decoder输入
- **Beam Search**: 推理时保留top-k候选翻译

**代码示例**:
```python
from transformers import MarianMTModel, MarianTokenizer

# 加载预训练翻译模型
model_name = "Helsinki-NLP/opus-mt-en-zh"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 翻译
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
translated = model.generate(**inputs)
print(tokenizer.decode(translated[0], skip_special_tokens=True))
# 输出: "你好，你好吗？"
```

---

## 🛠️ 实践环节

### 任务1: 词嵌入可视化

**目标**: 使用t-SNE可视化Word2Vec词向量，观察语义聚类

**关键代码** (`notebooks/stage4/04-rnn-text-classification.ipynb` 第2节):
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 提取词向量
words = ["king", "queen", "man", "woman", "apple", "orange"]
vectors = [model.wv[word] for word in words]

# 降维到2D
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.show()
```

**预期结果**: "king"-"queen" 距离近，"apple"-"orange" 距离近

### 任务2: 使用LSTM进行情感分类

**目标**: 在IMDB电影评论数据集上训练LSTM分类器

**步骤**:
1. 加载IMDB数据集
2. 文本预处理（分词、截断/填充）
3. 构建LSTM模型
4. 训练并评估

**预期结果**: 测试集准确率 > 85%

### 任务3: 微调BERT进行文本分类

**目标**: 使用Hugging Face Transformers微调BERT

**步骤**:
1. 加载预训练BERT模型
2. 准备数据集（tokenization）
3. 定义训练参数
4. 使用Trainer API微调
5. 评估性能

**预期结果**:
- 5 epochs内验证集准确率 > 90%
- 对比LSTM: BERT准确率提升5-10%

### 任务4: 可视化Attention权重

**目标**: 理解Transformer如何关注不同词语

**代码**:
```python
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 提取第一层第一个头的注意力权重
attention = outputs.attentions[0][0, 0].detach().numpy()

# 可视化
plt.imshow(attention, cmap='viridis')
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.show()
```

---

## 📖 扩展阅读

### 经典论文

1. **Attention Is All You Need** (Transformer, 2017)
   - 链接: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   - 阅读时间: 2小时

2. **BERT: Pre-training of Deep Bidirectional Transformers** (2018)
   - 链接: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
   - 阅读时间: 1.5小时

3. **Language Models are Few-Shot Learners** (GPT-3, 2020)
   - 链接: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
   - 阅读时间: 2小时

### 在线资源

- **CS224N (Stanford)**: [http://web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)
- **Hugging Face Course**: [https://huggingface.co/course/](https://huggingface.co/course/)
- **The Illustrated Transformer**: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

### 实战项目推荐

完成本模块后，建议尝试以下项目：

- 🚀 **[P06: Transformer翻译系统](../projects/p06-transformer-translation/README.md)** - 双框架实现（推荐）
- 🚀 **[P07: 预训练模型信息提取](../projects/p07-pretrained-info-extraction/README.md)** - BERT微调

---

## ❓ 常见问题 (FAQ)

### Q1: LSTM和Transformer如何选择？

**A**: 根据任务和数据规模选择：
- **小数据集 (<10k样本)**: LSTM/GRU (参数少，不易过拟合)
- **大数据集 (>100k样本)**: Transformer (性能更好)
- **实时推理**: LSTM (推理速度快)
- **批量推理**: Transformer (可并行)

### Q2: 如何处理未登录词(OOV)？

**A**: 三种策略：
1. **FastText**: 使用子词信息
2. **WordPiece/BPE**: 分词算法（BERT使用）
3. **`<UNK>` token**: 替换为特殊标记

### Q3: 为什么BERT不能生成文本？

**A**: BERT是**双向编码器**，训练时可以看到未来词，无法用于自回归生成。生成任务需要使用**单向解码器**（如GPT）。

### Q4: 微调BERT时如何避免过拟合？

**A**: 5个技巧：
1. 使用较小学习率 (2e-5 - 5e-5)
2. 添加Dropout (0.1 - 0.3)
3. 冻结部分层（只微调后几层）
4. 使用Early Stopping
5. 数据增强（回译、同义词替换）

### Q5: Transformer训练很慢怎么办？

**A**: 优化策略：
1. **减少序列长度**: 512 → 128（如果任务允许）
2. **使用梯度累积**: 模拟大batch size
3. **混合精度训练**: FP16代替FP32
4. **使用预训练模型**: 避免从零训练
5. **模型蒸馏**: 用小模型学习大模型

---

## ✅ 学习检查清单

完成本模块后，你应该能够：

- [ ] 解释Word2Vec的Skip-gram和CBOW训练方式
- [ ] 说明LSTM如何解决RNN的梯度消失问题
- [ ] 手动计算Self-Attention的输出（给定Q, K, V）
- [ ] 解释Multi-Head Attention的优势
- [ ] 区分BERT和GPT的架构与预训练任务
- [ ] 使用Hugging Face Transformers微调预训练模型
- [ ] 可视化并解释Attention权重
- [ ] 比较不同NLP模型在特定任务上的性能

---

## ⏭️ 下一步

完成本模块后，你可以：

1. **回顾总结**: 复习[模块M01](../01-dl-basics/README.md)和[模块M02](../02-cv-basics/README.md)
2. **实战项目**: 从[项目列表](../projects/)中选择NLP项目开始实践
3. **深入研究**: 阅读Transformer/BERT/GPT原论文
4. **进阶学习**: 进入[阶段5: AIGC与大模型](../../stage5/index.md)

---

**准备好了吗？打开 [04-rnn-text-classification.ipynb](../../notebooks/stage4/04-rnn-text-classification.ipynb) 开始动手实践！** 🚀
