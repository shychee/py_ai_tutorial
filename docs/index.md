# Python AI 教程：从机器学习到大模型

<div align="center">

**为具备 3-5 年 Python 后端经验的工程师打造的 AI 学习路径**

从传统机器学习到生成式 AI 的渐进式教程体系

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/shychee/py_ai_tutorial/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/shychee/py_ai_tutorial?style=social)](https://github.com/shychee/py_ai_tutorial/stargazers)

</div>

---

## 🎯 项目简介

本教程采用**实战驱动**的方式，帮助有一定Python基础的后端工程师系统学习人工智能技术栈。课程覆盖：

- **阶段3：机器学习与数据挖掘** - NumPy/Pandas数据分析 + scikit-learn传统ML
- **阶段4：深度学习** - PyTorch/TensorFlow深度学习 + CV/NLP应用
- **阶段5：AIGC与大模型** - LangChain/RAG/Agent + 对话系统开发

## ✨ 核心特点

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **渐进式学习路径**

    ---

    阶段3→4→5，明确前置依赖，循序渐进

- :material-laptop:{ .lg .middle } **跨平台支持**

    ---

    macOS (Intel/M系列)、Linux、Windows (原生/WSL2)、云端GPU

- :material-briefcase:{ .lg .middle } **实战项目丰富**

    ---

    17个行业项目（医疗、电商、金融、CV、NLP、LLM等）

- :material-code-braces:{ .lg .middle } **代码质量标准**

    ---

    PEP 8、类型注解、文档字符串、单元测试

- :material-compare:{ .lg .middle } **多框架对比**

    ---

    PyTorch vs TensorFlow核心项目双实现

- :material-translate:{ .lg .middle } **中文优先**

    ---

    文档、注释中文为主，技术术语中英对照

</div>

## 🚀 快速开始

### 环境要求

- Python ≥3.9（推荐3.11+）
- 操作系统：macOS 10.15+、Linux (Ubuntu 20.04+)、Windows 10/11
- 磁盘空间：至少10GB（含数据与模型缓存）
- 内存：8GB+（16GB推荐）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/shychee/py_ai_tutorial.git
cd py_ai_tutorial

# 2. 安装uv包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境
uv venv --python 3.11
source .venv/bin/activate  # macOS/Linux

# 4. 安装依赖（阶段3）
uv pip install -e ".[stage3]"

# 5. 下载数据集
python scripts/data/download-stage3.py
```

## 📚 学习路径

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
4. 通讯公司客户响应速度提升（RFM分析）
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

**项目作业**（7个，核心4个双框架实现）

**时间预估**: 6-10小时（GPU版本更快）

### 阶段5：AIGC与大模型（P3优先级）

**学习目标**: 掌握LLM应用开发（Prompt/RAG/Agent），能完成端到端对话系统

**时间预估**: 1-2天（端到端）

## 🖥️ 跨平台支持

我们提供针对不同操作系统的详细配置指引：

| 操作系统 | 特殊说明 |
|---------|---------|
| [macOS Intel (x86_64)](cross-platform/setup-macos-intel.md) | Homebrew安装，CPU环境 |
| [macOS Apple Silicon (arm64)](cross-platform/setup-macos-arm64.md) | MPS GPU加速支持 |
| [Linux (Ubuntu/CentOS)](cross-platform/setup-linux.md) | CUDA GPU支持 |
| [Windows 原生](cross-platform/setup-windows-native.md) | PowerShell脚本 |
| [Windows WSL2](cross-platform/setup-windows-wsl2.md) | 推荐，接近Linux体验 |
| [云端 GPU](cross-platform/setup-cloud-gpu.md) | Colab/AWS/自托管 |

**遇到问题？** 查看[故障恢复清单](cross-platform/troubleshooting.md)（≥5条常见问题与解决方案）

## 🤝 贡献指南

我们欢迎任何形式的贡献！

1. Fork本仓库
2. 创建Feature分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加某个很棒的功能'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📜 许可证

本项目采用 [MIT License](https://github.com/shychee/py_ai_tutorial/blob/main/LICENSE) 开源协议。

## 📧 联系方式

- **GitHub Issues**: [提交问题](https://github.com/shychee/py_ai_tutorial/issues)
- **邮件**: shychee96@gmail.com
- **文档**: [在线文档](https://shychee.github.io/py_ai_tutorial/)

---

**祝学习顺利！从数据分析到生成式AI，一步一个脚印。** 🚀
