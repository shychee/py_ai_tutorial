# Python AI 教程：从机器学习到大模型

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/shychee/py_ai_tutorial?include_prereleases)](https://github.com/shychee/py_ai_tutorial/releases)
[![GitHub stars](https://img.shields.io/github/stars/shychee/py_ai_tutorial?style=social)](https://github.com/shychee/py_ai_tutorial/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/shychee/py_ai_tutorial?style=social)](https://github.com/shychee/py_ai_tutorial/network/members)

[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue?logo=markdown&logoColor=white)](https://shychee.github.io/py_ai_tutorial/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](notebooks/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Ready-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Ready-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Ready-blue?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

[![macOS](https://img.shields.io/badge/macOS-Intel%20%7C%20Apple%20Silicon-black?logo=apple&logoColor=white)](docs/cross-platform/setup-macos-intel.md)
[![Linux](https://img.shields.io/badge/Linux-Ubuntu%20%7C%20CentOS-orange?logo=linux&logoColor=white)](docs/cross-platform/setup-linux.md)
[![Windows](https://img.shields.io/badge/Windows-Native%20%7C%20WSL2-blue?logo=windows&logoColor=white)](docs/cross-platform/setup-windows-native.md)

</div>

<div align="center">

> 为具备 3-5 年 Python 后端经验的工程师打造的 AI 学习路径，从传统机器学习到生成式 AI 的渐进式教程体系。

</div>

---

## 🎯 项目简介

本教程采用**实战驱动**的方式，帮助有一定Python基础的后端工程师系统学习人工智能技术栈。课程覆盖：

- **阶段3：机器学习与数据挖掘** - NumPy/Pandas数据分析 + scikit-learn传统ML
- **阶段4：深度学习** - PyTorch/TensorFlow深度学习 + CV/NLP应用
- **阶段5：AIGC与大模型** - LangChain/RAG/Agent + 对话系统开发

### 核心特点

✅ **渐进式学习路径** - 阶段3→4→5，明确前置依赖，循序渐进

✅ **跨平台支持** - macOS (Intel/M系列)、Linux、Windows (原生/WSL2)、云端GPU

✅ **实战项目丰富** - 17个行业项目（医疗、电商、金融、CV、NLP、LLM等）

✅ **代码质量标准** - PEP 8、类型注解、文档字符串、单元测试

✅ **多框架对比** - PyTorch vs TensorFlow核心项目双实现

✅ **中文优先** - 文档、注释中文为主，技术术语中英对照

✅ **现代化工具链** - 使用uv包管理器 + pyproject.toml标准配置

---

## 🚀 快速开始

### 环境要求

- Python ≥3.9（推荐3.11+）
- 操作系统：macOS 10.15+、Linux (Ubuntu 20.04+)、Windows 10/11
- 磁盘空间：至少10GB（含数据与模型缓存）
- 内存：8GB+（16GB推荐）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/shychee/py_ai_tutorial.git
cd py_ai_tutorial
```

#### 2. 安装uv包管理器

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境（Python 3.11）
uv venv --python 3.11

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装阶段3依赖（传统机器学习，CPU即可）
uv pip install -e ".[stage3]"

# 或安装完整依赖（包含阶段3/4/5 + 文档 + 开发工具）
uv pip install -e ".[all]"
```

#### 4. 获取数据集

P01-P03 的数据集已包含在仓库中，clone 后即可使用。

P04 使用 Kaggle 真实数据集，需手动下载：

```bash
# 方式1: 使用 Kaggle CLI
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/stage3/ --unzip

# 方式2: 浏览器下载
# 访问 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 检查数据集状态
python scripts/data/download-stage3.py
```

#### 5. 运行首个项目

**Jupyter Notebook方式**（推荐新手）:
```bash
# 安装Jupyter Lab
uv pip install jupyterlab

# 启动Jupyter Lab
jupyter lab

# 在浏览器中打开 projects/stage3/p01-healthcare/notebooks/analysis.ipynb
# 按顺序运行每个cell
```

**Python脚本方式**（推荐有经验者）:
```bash
# 进入项目目录
cd projects/stage3/p01-healthcare

# 运行分析脚本
python src/analyze.py --config configs/default.yaml

# 查看输出
ls outputs/
```

**预期结果**:
- 输出指标落在合理范围内（总销售额 12M-13M，平均订单金额 200-300）
- 生成可视化图表（outputs/figures/）
- 生成实验报告（outputs/reports/report.md）

---

## 📚 学习路径

> **目录说明**: `notebooks/stage3/` 包含基础教学 notebook（NumPy、Pandas、可视化等），`projects/stage3/` 包含项目实战代码和 notebook。建议先学基础，再做项目。

### 阶段3：机器学习与数据挖掘（P1优先级，MVP核心）

**学习目标**: 掌握数据分析与传统ML算法，能完成端到端机器学习项目

**模块**:
- M01: 科学计算库（NumPy/Pandas/Matplotlib）
- M02: Pandas项目实战
- M03: AI数学基础
- M04: 机器学习进阶（分类/回归/聚类/集成）

**项目作业**（9个）:
1. 朝阳医院指标搭建及销售数据汇总
2. 服装零售销售数据分析（优衣库4P分析）
3. 银行电话营销活动分析（分类模型）
4. 电信客户流失预测（特征工程 + 随机森林）
5. 零售超市经营分析（SWOT分析）
6. 滴滴出行运营数据异常分析
7. 淘宝百万级用户行为分析（年度复盘）
8. 航空公司客户价值分析（K-means聚类）
9. 信用贷款前审批项目（风控模型）

**时间预估**: 4-6小时（理论2-3h + 实践2-3h）

### 阶段4：深度学习（P2优先级）

**学习目标**: 掌握深度学习框架（PyTorch/TensorFlow），能完成CV/NLP迁移学习项目

**模块**:
- M01: 深度学习基础理论（神经网络/反向传播/优化器）
- M02: 计算机视觉基础（CNN/目标检测/图像分割）
- M03: 自然语言处理基础（RNN/Transformer/预训练模型）

**项目作业**（7个，核心4个双框架实现）:
1. 工业视觉检测（TensorFlow，TF Lite部署）
2. 基于YOLOv11的视频实时检测系统 ⭐双框架
3. OCR票据识别（PaddlePaddle）
4. 自动驾驶场景图像分割 ⭐双框架
5. 医学影像分析（PyTorch + MONAI）
6. 基于Transformer的翻译系统 ⭐双框架
7. 基于预训练模型的信息提取系统

**时间预估**: 6-10小时（GPU版本更快）

### 阶段5：AIGC与大模型（P3优先级）

**学习目标**: 掌握LLM应用开发（Prompt/RAG/Agent），能完成端到端对话系统

**模块**:
- M01: AIGC与LLM概述（GPT原理/主流LLM对比）
- M02: 大模型开发（DeepSeek/RAG/Agent/Fine-tuning）

**项目作业**（1个综合项目）:
1. 基于LLM的对话系统（RAG + Agent）

**时间预估**: 1-2天（端到端）

---

## 🖥️ 跨平台支持

我们提供针对不同操作系统的详细配置指引：

| 操作系统 | 文档链接 | 特殊说明 |
|---------|---------|---------|
| macOS Intel (x86_64) | [setup-macos-intel.md](docs/cross-platform/setup-macos-intel.md) | Homebrew安装，CPU环境 |
| macOS Apple Silicon (arm64) | [setup-macos-arm64.md](docs/cross-platform/setup-macos-arm64.md) | MPS GPU加速支持 |
| Linux (Ubuntu/CentOS) | [setup-linux.md](docs/cross-platform/setup-linux.md) | CUDA GPU支持 |
| Windows 原生 | [setup-windows-native.md](docs/cross-platform/setup-windows-native.md) | PowerShell脚本 |
| Windows WSL2 | [setup-windows-wsl2.md](docs/cross-platform/setup-windows-wsl2.md) | 推荐，接近Linux体验 |
| 云端 GPU | [setup-cloud-gpu.md](docs/cross-platform/setup-cloud-gpu.md) | Colab/AWS/自托管 |

**遇到问题？** 查看[故障恢复清单](docs/cross-platform/troubleshooting.md)（≥5条常见问题与解决方案）

---

## 📖 项目结构

```
py_ai_tutorial/
├── docs/                      # 教程文档（Markdown）
│   ├── stage3/                # 阶段3：机器学习与数据挖掘
│   │   ├── 01-scientific-computing/
│   │   ├── 02-pandas-practice/
│   │   ├── 03-ml-basics/
│   │   ├── 04-ml-advanced/
│   │   └── projects/          # 9个小项目
│   ├── stage4/                # 阶段4：深度学习
│   │   ├── 01-dl-basics/
│   │   ├── 02-cv-basics/
│   │   ├── 03-nlp-basics/
│   │   └── projects/          # 7个小项目
│   ├── stage5/                # 阶段5：AIGC与大模型
│   │   ├── 01-aigc-llm-intro/
│   │   ├── 02-llm-dev/
│   │   └── projects/          # 1个综合项目
│   └── cross-platform/        # 跨平台配置指引
├── notebooks/                 # Jupyter Notebook（教学版）
│   ├── stage3/
│   ├── stage4/
│   └── stage5/
├── scripts/                   # 工具脚本
│   ├── data/                  # 数据下载与验证
│   ├── env/                   # 环境检测与配置
│   └── evaluation/            # 评估脚本
├── data/                      # 数据集（运行后生成）
│   ├── stage3/
│   ├── stage4/
│   └── stage5/
├── templates/                 # 项目模板
│   └── project-template/
├── tests/                     # 单元测试
├── mkdocs.yml                 # MkDocs配置
├── pyproject.toml             # 项目配置与依赖（uv标准）
└── README.md                  # 本文件
```

---

## 🧪 测试

运行所有测试：

```bash
# 单元测试
pytest tests/

# Notebook测试（需要nbval）
pytest --nbval notebooks/

# 数据校验
python scripts/data/verify.py --stage 3
```

---

## 🛠️ 开发

### 安装开发依赖

```bash
uv pip install -e ".[dev]"
```

### 代码格式化与检查

```bash
# 格式化代码（black）
black scripts/ tests/

# 代码检查（ruff）
ruff check scripts/ tests/

# 类型检查（mypy）
mypy scripts/
```

### 构建文档

```bash
# 安装文档依赖
uv pip install -e ".[docs]"

# 本地预览
mkdocs serve
# 访问 http://localhost:8000

# 构建静态站点
mkdocs build
# 输出到 site/ 目录
```

---

## 🤝 贡献指南

我们欢迎任何形式的贡献！请查看[贡献指南](docs/contributing.md)了解详情。

### 如何贡献

1. Fork本仓库
2. 创建Feature分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加某个很棒的功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8代码规范
- 使用类型注解（Type Hints）
- 编写文档字符串（Docstrings）
- 添加单元测试（覆盖率≥80%）
- 使用中文注释（技术术语除外）

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 📧 联系方式

- **GitHub Issues**: [提交问题](https://github.com/shychee/py_ai_tutorial/issues)
- **邮件**: shychee96@gmail.com
- **文档**: [在线文档](https://shychee.github.io/py_ai_tutorial/)

---

## 🙏 致谢

本教程参考了以下优秀资源：

- [scikit-learn文档](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [LangChain文档](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

**祝学习顺利！从数据分析到生成式AI，一步一个脚印。** 🚀
