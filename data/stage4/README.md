# Stage 4 数据集与预训练模型

**阶段**: Stage 4 - 深度学习
**更新日期**: 2025-11-17

---

## 📦 数据集概览

Stage 4 包含深度学习训练所需的标准数据集和预训练模型权重。

### 必需数据集 (Required)

| 数据集 | 大小 | 类型 | 用途 | 自动下载 |
|-------|------|------|------|---------|
| **MNIST** | ~11 MB | 图像分类 | 手写数字识别(0-9) | ✅ PyTorch |
| **CIFAR-10** | ~170 MB | 图像分类 | 10类物体识别 | ✅ PyTorch |
| **IMDB** | ~80 MB | 文本分类 | 电影评论情感分析 | ✅ HuggingFace |

### 可选数据集 (Optional)

| 数据集 | 大小 | 类型 | 用途 | 下载方式 |
|-------|------|------|------|---------|
| **CIFAR-100** | ~170 MB | 图像分类 | 100类物体识别 | PyTorch |
| **ImageNet Sample** | ~1 GB | 图像分类 | 迁移学习训练 | 手动下载 |
| **COCO Sample** | ~500 MB | 目标检测 | YOLOv8训练 | 手动下载 |

---

## 🤖 预训练模型

### 必需模型 (Required)

| 模型 | 大小 | 框架 | 用途 | 自动下载 |
|------|------|------|------|---------|
| **ResNet-50** | ~100 MB | PyTorch | CNN迁移学习 | ✅ 首次使用时 |

### 可选模型 (Optional)

| 模型 | 大小 | 框架 | 用途 | 下载方式 |
|------|------|------|------|---------|
| **BERT-base-uncased** | ~440 MB | PyTorch/TF | NLP预训练 | HuggingFace |
| **YOLOv8n** | ~6 MB | PyTorch | 目标检测 | Ultralytics |

---

## 🚀 快速开始

### 方式1: 自动下载（推荐）

运行下载脚本自动获取所有必需数据集：

```bash
# 从项目根目录运行
python scripts/data/download-stage4.py

# 仅下载数据集（跳过模型）
python scripts/data/download-stage4.py --skip-models

# 下载指定数据集
python scripts/data/download-stage4.py --dataset DS-S4-MNIST

# 使用国内镜像加速（开发中）
python scripts/data/download-stage4.py --mirror
```

### 方式2: 在代码中自动下载

大多数数据集在首次使用时会自动下载：

```python
import torchvision

# MNIST 自动下载
train_dataset = torchvision.datasets.MNIST(
    root='./data/stage4/mnist',
    train=True,
    download=True  # 首次运行时自动下载
)

# CIFAR-10 自动下载
cifar_dataset = torchvision.datasets.CIFAR10(
    root='./data/stage4/cifar10',
    train=True,
    download=True
)
```

### 方式3: 离线数据包

如果网络受限，可以使用预打包的离线数据：

```bash
# 下载离线包（假设从服务器获取）
# offline/stage4-data.tar.gz (~2GB, 包含MNIST/CIFAR-10/IMDB)
# offline/stage4-models.tar.gz (~500MB, 包含ResNet-50权重)

# 解压到数据目录
tar -xzf offline/stage4-data.tar.gz -C data/stage4/
tar -xzf offline/stage4-models.tar.gz -C data/models/
```

---

## 📁 目录结构

下载完成后，目录结构如下：

```
data/stage4/
├── mnist/                          # MNIST 数据集
│   ├── MNIST/
│   │   └── raw/
│   │       ├── train-images-idx3-ubyte.gz
│   │       ├── train-labels-idx1-ubyte.gz
│   │       ├── t10k-images-idx3-ubyte.gz
│   │       └── t10k-labels-idx1-ubyte.gz
│   └── processed/
│       ├── training.pt
│       └── test.pt
│
├── cifar10/                        # CIFAR-10 数据集
│   └── cifar-10-batches-py/
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── test_batch
│       ├── batches.meta
│       └── readme.html
│
├── cifar100/                       # CIFAR-100 (可选)
│   └── cifar-100-python/
│       ├── train
│       ├── test
│       └── meta
│
├── imdb/                           # IMDB 电影评论
│   └── imdb/
│       └── 1.0.0/
│           ├── train/
│           ├── test/
│           └── unsupervised/
│
├── imagenet-sample/                # ImageNet 样本 (可选)
│   └── imagenette2/
│       ├── train/
│       └── val/
│
└── coco-sample/                    # COCO 样本 (可选)
    └── val2017/
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        └── ...

data/models/                        # 预训练模型权重
├── resnet50_pytorch.pth           # ResNet-50 PyTorch权重
├── bert-base-uncased/             # BERT模型 (可选)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
└── yolov8n.pt                     # YOLOv8 Nano (可选)
```

---

## 📊 数据集详细说明

### 1. MNIST 手写数字数据集

**简介**: 60,000张训练图像 + 10,000张测试图像，28×28灰度图像

**类别**: 数字 0-9

**使用示例**:
```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data/stage4/mnist',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**用于**:
- `notebooks/stage4/02-pytorch-basics.ipynb`
- 神经网络基础教学

---

### 2. CIFAR-10 图像分类数据集

**简介**: 60,000张32×32彩色图像，10个类别

**类别**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**使用示例**:
```python
import torchvision

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/stage4/cifar10',
    train=True,
    download=True,
    transform=transform_train
)
```

**用于**:
- `notebooks/stage4/03-cnn-image-classification.ipynb`
- CNN基础与迁移学习

---

### 3. IMDB 电影评论数据集

**简介**: 50,000条电影评论，二分类情感分析

**类别**: Positive (正面) / Negative (负面)

**使用示例**:
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('imdb', cache_dir='./data/stage4/imdb')

# 查看样本
print(dataset['train'][0])
# {'text': '...', 'label': 1}  # 1=positive, 0=negative
```

**用于**:
- `notebooks/stage4/04-rnn-text-classification.ipynb`
- RNN/LSTM/BERT文本分类

---

### 4. CIFAR-100 图像分类数据集 (可选)

**简介**: 60,000张32×32彩色图像，100个细粒度类别

**类别**: 100个类别，分为20个超类

**使用示例**:
```python
trainset = torchvision.datasets.CIFAR100(
    root='./data/stage4/cifar100',
    train=True,
    download=True,
    transform=transform
)
```

**用于**:
- CNN进阶实验
- 细粒度分类任务

---

### 5. ImageNet Sample (可选)

**简介**: ImageNet-1K的精简版本，包含10个类别

**大小**: ~1GB

**下载方式**:
```bash
# 使用脚本下载
python scripts/data/download-stage4.py --dataset DS-S4-IMAGENET-SAMPLE

# 或手动下载
wget https://github.com/fastai/imagenette/releases/download/v0.3/imagenette2.tgz
tar -xzf imagenette2.tgz -C data/stage4/imagenet-sample/
```

**用于**:
- 迁移学习实战
- `docs/stage4/projects/p01-industrial-vision/`

---

### 6. COCO Sample (可选)

**简介**: COCO 2017验证集样本

**大小**: ~500MB (1000张图像)

**下载方式**:
```bash
# 手动下载
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/stage4/coco-sample/
```

**用于**:
- 目标检测任务
- `docs/stage4/projects/p02-yolov11-realtime/`

---

## 🔍 预训练模型详细说明

### 1. ResNet-50 (PyTorch)

**简介**: 在ImageNet-1K上预训练的ResNet-50模型

**参数量**: ~25M

**使用示例**:
```python
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(pretrained=True)

# 冻结卷积层，仅训练最后的全连接层
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层用于自己的任务
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10类分类
```

**用于**:
- `notebooks/stage4/03-cnn-image-classification.ipynb`
- 迁移学习教程

---

### 2. BERT-base-uncased (可选)

**简介**: 12层Transformer编码器，110M参数

**使用示例**:
```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./data/models/bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, cache_dir='./data/models/bert-base-uncased')

# 使用
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

**用于**:
- `notebooks/stage4/04-rnn-text-classification.ipynb`
- NLP预训练模型微调

---

### 3. YOLOv8n (可选)

**简介**: YOLOv8 Nano版本，轻量级目标检测模型

**参数量**: ~3M

**使用示例**:
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
model.train(data='coco.yaml', epochs=100)

# 推理
results = model('path/to/image.jpg')
```

**用于**:
- `docs/stage4/projects/p02-yolov11-realtime/`
- 实时目标检测

---

## ⚙️ 环境要求

### Python 依赖

```bash
# 必需
pip install torch torchvision

# 可选（根据需要安装）
pip install transformers datasets
pip install ultralytics  # YOLOv8
```

### 硬件要求

| 数据集/模型 | CPU | GPU | 内存 | 硬盘 |
|------------|-----|-----|------|------|
| MNIST/CIFAR-10 | ✅ | 可选 | 4GB | 200MB |
| IMDB | ✅ | 可选 | 8GB | 100MB |
| ResNet-50 迁移学习 | ✅ | 推荐 | 8GB | 100MB |
| BERT 微调 | ❌ | **必需** | 16GB | 500MB |
| YOLOv8 训练 | ❌ | **必需** | 16GB | 1GB |

**GPU 推荐**:
- 入门: NVIDIA GTX 1060 (6GB VRAM)
- 推荐: NVIDIA RTX 3060 (12GB VRAM)
- 专业: NVIDIA A100 (40GB VRAM)

---

## 🔧 故障排查

### Q1: 下载速度慢或失败

**解决方案**:
1. 使用国内镜像（开发中）
2. 使用代理: `export https_proxy=http://127.0.0.1:7890`
3. 下载离线包: 参见"方式3: 离线数据包"

### Q2: PyTorch 数据集损坏

**解决方案**:
```bash
# 删除缓存并重新下载
rm -rf data/stage4/mnist/
python -c "import torchvision; torchvision.datasets.MNIST(root='./data/stage4/mnist', download=True)"
```

### Q3: HuggingFace 下载失败

**解决方案**:
```bash
# 设置镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用离线模式
export TRANSFORMERS_OFFLINE=1
```

### Q4: GPU 内存不足

**解决方案**:
1. 减小 batch_size
2. 使用混合精度训练: `torch.cuda.amp`
3. 使用梯度累积
4. 使用 CPU（速度较慢）

---

## 📚 参考资料

### 数据集来源

- **MNIST**: [Yann LeCun's Website](http://yann.lecun.com/exdb/mnist/)
- **CIFAR-10/100**: [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
- **IMDB**: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
- **ImageNet**: [ImageNet Official](https://image-net.org/)
- **COCO**: [COCO Dataset](https://cocodataset.org/)

### 预训练模型来源

- **ResNet**: [PyTorch Hub](https://pytorch.org/hub/pytorch_vision_resnet/)
- **BERT**: [HuggingFace Hub](https://huggingface.co/bert-base-uncased)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)

---

## 📝 更新日志

### 2025-11-17
- ✅ 创建 Stage 4 数据集下载脚本
- ✅ 支持 MNIST、CIFAR-10、IMDB 自动下载
- ✅ 支持 ResNet-50、BERT、YOLOv8 模型下载
- ✅ 添加离线数据包支持（计划中）

---

## 🆘 获取帮助

遇到问题？

1. 查看 [故障排查](#故障排查) 部分
2. 运行验证脚本: `python scripts/data/verify.py --stage 4`
3. 查看 [跨平台故障恢复清单](../../docs/cross-platform/troubleshooting.md)
4. 提交 Issue: [GitHub Issues](https://github.com/yourusername/py_ai_tutorial/issues)

---

**上一阶段**: [Stage 3 数据集](../stage3/README.md)
**返回**: [项目根目录](../../README.md)
