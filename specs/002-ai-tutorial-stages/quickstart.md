# 快速开始指南

**Feature**: 阶段3-5教程与跨平台指引
**Last Updated**: 2025-11-05
**Estimated Time**: 30-60分钟（环境配置 + 首个项目运行）

---

## 概述

本指南帮助你在30-60分钟内完成以下任务：
1. **环境配置**: 安装Python、uv包管理器与项目依赖
2. **数据准备**: 下载并验证教程数据集
3. **首个项目**: 运行阶段3的首个项目（数据分析），验证环境正常

**适用人群**: 具备3-5年Python后端经验、基本命令行使用能力的学习者

**前置要求**:
- 操作系统: macOS (10.15+)、Linux (Ubuntu 20.04+/CentOS 8+)、Windows 10/11
- 磁盘空间: 至少10GB可用空间
- 网络: 可访问PyPI与GitHub（若受限，参见[离线模式](#offline-mode)）

---

## 第一步：检测系统环境

在开始之前，请确认你的操作系统与硬件配置：

### 自动检测（推荐）

```bash
# 克隆项目（若尚未克隆）
git clone https://github.com/yourusername/py_ai_tutorial.git
cd py_ai_tutorial

# 运行环境检测脚本
python3 scripts/env/detect-platform.py
```

**输出示例**:
```
检测到的环境信息:
  操作系统: macOS 14.0 (arm64)
  Python版本: 3.11.5
  CPU架构: arm64 (Apple Silicon)
  GPU: Apple M1 GPU (MPS支持)
  内存: 16 GB
  推荐配置文档: docs/cross-platform/setup-macos-arm64.md
```

### 手动检测

**macOS**:
```bash
# 检测架构
uname -m  # 输出 x86_64 (Intel) 或 arm64 (Apple Silicon)

# 检测Python版本
python3 --version  # 应为 3.9+ (推荐 3.11+)
```

**Linux**:
```bash
# 检测发行版
cat /etc/os-release

# 检测CUDA（若有GPU）
nvidia-smi  # 若有输出，说明有NVIDIA GPU
```

**Windows**:
```powershell
# 检测Windows版本
winver

# 检测WSL2（推荐）
wsl --status
```

---

## 第二步：安装Python与uv

### macOS

#### Intel芯片（x86_64）

```bash
# 1. 安装Homebrew（若未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装Python 3.11
brew install python@3.11

# 3. 验证安装
python3.11 --version
# 输出: Python 3.11.x

# 4. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 5. 重启终端或执行
source $HOME/.cargo/env

# 6. 验证uv
uv --version
# 输出: uv 0.x.x
```

#### Apple Silicon（arm64）

```bash
# 步骤同Intel，但Homebrew路径为 /opt/homebrew
brew install python@3.11
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

**常见问题**:
- **问题**: `uv`命令未找到
  - **解决**: 重启终端，或手动执行`source $HOME/.cargo/env`
- **问题**: Homebrew安装失败（网络问题）
  - **解决**: 使用国内镜像，参见[Homebrew镜像配置](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)

### Linux（Ubuntu 20.04+）

```bash
# 1. 更新包管理器
sudo apt update

# 2. 安装Python 3.11（若系统默认版本<3.11）
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-dev python3.11-venv

# 3. 验证安装
python3.11 --version

# 4. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv --version
```

**GPU支持（可选，阶段4/5需要）**:
```bash
# 检测NVIDIA驱动
nvidia-smi

# 若未安装，参考NVIDIA官方文档安装驱动与CUDA 11.8+
# https://developer.nvidia.com/cuda-downloads
```

### Windows

#### 推荐方式：WSL2（获得接近Linux的体验）

```powershell
# 1. 启用WSL2
wsl --install

# 2. 重启电脑

# 3. 安装Ubuntu（默认）
wsl --install -d Ubuntu-22.04

# 4. 进入WSL2环境
wsl

# 5. 在WSL2内按照Linux步骤安装Python与uv
sudo apt update
sudo apt install python3.11 python3.11-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### 原生Windows（备选）

```powershell
# 1. 下载Python 3.11 Windows安装包
# https://www.python.org/downloads/windows/

# 2. 安装时勾选"Add Python to PATH"

# 3. 验证安装
python --version

# 4. 安装uv（使用pipx）
python -m pip install pipx
pipx install uv

# 5. 验证uv
uv --version
```

**注意**: Windows原生环境可能遇到更多兼容性问题，**强烈建议使用WSL2**。

---

## 第三步：克隆项目与创建虚拟环境

```bash
# 1. 克隆项目（若未克隆）
git clone https://github.com/yourusername/py_ai_tutorial.git
cd py_ai_tutorial

# 2. 创建虚拟环境（使用uv）
uv venv --python 3.11

# 3. 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate

# Windows (WSL2):
source .venv/bin/activate

# Windows (原生, PowerShell):
.\.venv\Scripts\Activate.ps1

# 4. 验证虚拟环境激活
which python  # 应输出 .venv/bin/python
python --version  # 应输出 Python 3.11.x
```

---

## 第四步：安装项目依赖

### 阶段3依赖（数据分析 + 传统机器学习）

```bash
# 激活虚拟环境后
uv pip install -e ".[stage3]"

# 等待5-10分钟（首次安装需下载NumPy、Pandas、scikit-learn等）
# uv比pip快10-100倍，通常2-3分钟即可完成

# 验证安装
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

### 阶段4依赖（深度学习，CPU版）

```bash
# 安装阶段3依赖 + PyTorch CPU版
uv pip install -e ".[stage3,stage4-cpu]"

# 验证PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 输出应为: 2.1.x, False (CPU版本)
```

### 阶段4依赖（深度学习，GPU版）

**macOS (Apple Silicon, MPS支持)**:
```bash
uv pip install -e ".[stage3,stage4-mps]"

# 验证MPS
python -c "import torch; print(torch.backends.mps.is_available())"
# 输出: True
```

**Linux (CUDA 11.8)**:
```bash
# 确保已安装CUDA 11.8驱动
nvidia-smi

uv pip install -e ".[stage3,stage4-gpu]"

# 验证CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
# 输出: True, 11.8
```

### 阶段5依赖（LLM/AIGC）

```bash
uv pip install -e ".[stage3,stage5]"

# 验证HuggingFace transformers
python -c "import transformers; print(transformers.__version__)"
```

**依赖安装故障排查**:
- **问题**: 安装超时或失败
  - **解决**: 使用国内PyPI镜像：`uv pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple`
- **问题**: CUDA版本不匹配
  - **解决**: 根据`nvidia-smi`输出的CUDA版本，修改`pyproject.toml`中的torch版本
- **问题**: Apple Silicon安装TensorFlow失败
  - **解决**: 使用`tensorflow-macos`与`tensorflow-metal`（已在pyproject.toml配置）

---

## 第五步：下载数据集

### 自动下载（推荐）

```bash
# 下载阶段3所有数据集（约2GB）
python scripts/data/download-stage3.py

# 下载特定项目数据集
python scripts/data/download-stage3.py --project p01-healthcare

# 验证数据完整性
python scripts/data/verify.py --stage 3
```

**输出示例**:
```
[INFO] 开始下载阶段3数据集...
[INFO] 下载 healthcare_sales.csv (12.5 MB)... 完成
[INFO] 下载 ecommerce_orders.csv (45.3 MB)... 完成
...
[INFO] 所有数据集下载完成，总大小: 1.8 GB
[INFO] 运行校验...
[SUCCESS] 所有数据集SHA256校验通过
```

### 离线模式（网络受限）

```bash
# 1. 从百度网盘/阿里云盘下载离线包（提供链接）
# stage3-data.tar.gz (~1.5 GB)

# 2. 解压到data/目录
tar -xzf stage3-data.tar.gz -C data/

# 3. 验证完整性
python scripts/data/verify.py --stage 3 --offline

# 4. 设置离线模式环境变量
export OFFLINE_MODE=1  # macOS/Linux
# 或
$env:OFFLINE_MODE=1  # Windows PowerShell
```

---

## 第六步：运行首个项目

### 项目：朝阳医院指标搭建及销售数据汇总（阶段3-P01）

**项目目标**: 使用Pandas完成数据清洗、聚合分析与可视化，熟悉数据分析工作流。

#### 方式1: Jupyter Notebook（推荐新手）

```bash
# 1. 安装Jupyter Lab（若未安装）
uv pip install jupyterlab

# 2. 启动Jupyter Lab
jupyter lab

# 3. 在浏览器中打开
# http://localhost:8888

# 4. 导航到 notebooks/stage3/p01-healthcare.ipynb

# 5. 按顺序运行每个cell（Shift + Enter）

# 6. 查看输出（数据表、图表、指标）
```

**预期输出**:
- 数据加载成功，显示前5行
- 缺失值统计表
- 清洗后数据行数（应为~48000，原始50000去重后）
- 月度销售趋势图
- TOP 10产品柱状图
- 汇总指标表（总销售额、平均订单金额等）

#### 方式2: Python脚本（推荐有经验者）

```bash
# 1. 查看项目结构
cd docs/stage3/projects/p01-healthcare
tree -L 2

# 输出:
# .
# ├── README.md
# ├── pyproject.toml
# ├── data/ -> ../../../../data/stage3/
# ├── configs/
# │   └── default.yaml
# ├── src/
# │   ├── __init__.py
# │   ├── analyze.py
# │   └── utils/
# ├── outputs/ (运行后生成)
# └── tests/

# 2. 运行分析脚本
python src/analyze.py --config configs/default.yaml

# 3. 查看输出
ls outputs/
# 输出:
# experiment_info.json
# figures/
#   ├── sales_trend.png
#   └── top_products.png
# reports/
#   └── report.md
# processed_data.parquet
```

**预期运行时间**: 30-60秒（CPU）

#### 验证结果

```bash
# 1. 检查实验元数据
cat outputs/experiment_info.json

# 输出示例:
# {
#   "timestamp": "2025-11-05T10:30:00Z",
#   "python_version": "3.11.5",
#   "metrics": {
#     "total_sales": 12500000.5,
#     "avg_order_value": 250.0,
#     "data_coverage": 96.0
#   }
# }

# 2. 查看实验报告
cat outputs/reports/report.md

# 3. 查看图表（使用系统默认图片查看器）
open outputs/figures/sales_trend.png  # macOS
xdg-open outputs/figures/sales_trend.png  # Linux
start outputs/figures/sales_trend.png  # Windows
```

#### 对比预期指标

根据项目README中的"预期指标范围"：
- **总销售额**: 应在12,000,000 - 13,000,000之间 ✅
- **平均订单金额**: 应在200 - 300之间 ✅
- **数据覆盖率**: 应≥90% ✅

如果你的结果在范围内，**恭喜！环境配置成功**。

---

## 第七步：提交与评估（可选）

### 自我评估

```bash
# 运行项目自带的测试
cd docs/stage3/projects/p01-healthcare
pytest tests/

# 运行代码质量检查
black --check src/
mypy src/
```

### 自动评估（使用评估API）

```bash
# 1. 打包提交物
cd py_ai_tutorial
tar -czf submission.tar.gz docs/stage3/projects/p01-healthcare/

# 2. 提交到评估系统（若已部署）
curl -X POST http://localhost:8000/api/v1/evaluate \
  -F "project_id=stage3-p01-healthcare" \
  -F "learner_id=your_user_id" \
  -F "submission_file=@submission.tar.gz"

# 3. 查看评估结果
# 返回JSON，包含total_score、passed、feedback等
```

**评估维度**（参见项目README）:
- **代码质量** (30%): PEP 8、类型注解、文档字符串
- **数据处理正确性** (40%): 缺失值/异常值/重复项处理
- **分析深度** (20%): 多维度分析、有洞察的结论
- **可视化质量** (10%): 图表类型、标注完整性

**及格分**: 60分
**你的目标**: ≥80分（优秀）

---

## 常见问题（FAQ）

### Q1: 安装uv时提示"permission denied"

**A**: 不要使用`sudo`安装uv（避免权限问题），使用当前用户安装：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
若仍失败，使用pipx安装：
```bash
python3 -m pip install --user pipx
pipx install uv
```

### Q2: 虚拟环境激活后，`python`仍指向系统Python

**A**: 检查激活脚本是否正确执行：
```bash
# macOS/Linux:
source .venv/bin/activate

# 验证:
which python  # 应输出 /path/to/py_ai_tutorial/.venv/bin/python
```

### Q3: 数据下载失败（网络超时）

**A**:
1. 使用国内镜像源（若数据托管在GitHub Releases，可能被墙）
2. 下载离线数据包（百度网盘/阿里云盘链接见README）
3. 若在企业内网，联系管理员配置代理

### Q4: Jupyter Notebook无法打开

**A**:
1. 确保已安装`jupyterlab`：`uv pip install jupyterlab`
2. 检查端口占用：`jupyter lab --port=8889`
3. 若浏览器未自动打开，手动访问终端输出的URL

### Q5: 运行脚本时提示"ModuleNotFoundError"

**A**:
1. 确认虚拟环境已激活（终端提示符前有`(.venv)`）
2. 重新安装依赖：`uv pip install -e .`
3. 检查是否在项目根目录：`pwd`应输出`.../py_ai_tutorial`

### Q6: Apple Silicon运行PyTorch报错

**A**:
1. 确认安装的是ARM64版本：`python -c "import platform; print(platform.machine())"`应输出`arm64`
2. 使用MPS后端：`torch.backends.mps.is_available()`应返回`True`
3. 若仍失败，降级到CPU版本（性能略差但更稳定）

### Q7: 停滞超过30分钟，怎么办？

**A**: 根据"故障恢复清单"（docs/cross-platform/troubleshooting.md）排查，若仍无法解决：
1. **切换到云端GPU方案**（见下一节）
2. 在GitHub Issues提交问题（附上错误日志、环境信息）
3. 加入学习社群寻求帮助

---

## 云端GPU快速启动（逃生舱）

如果本地环境配置耗时超过30分钟，或CPU训练速度过慢（单epoch >30分钟），可切换到云端GPU：

### 选项A: Google Colab（免费，有GPU配额限制）

1. 访问 https://colab.research.google.com/
2. 上传Notebook：`File -> Upload notebook -> notebooks/stage4/p01-industrial-vision.ipynb`
3. 启用GPU：`Runtime -> Change runtime type -> GPU -> T4`
4. 运行第一个cell（安装依赖）
5. 运行后续cell（训练与评估）

**优点**: 免费、即开即用
**缺点**: 会话有时限（12小时）、GPU配额有限

### 选项B: 自托管云主机（推荐团队/企业）

1. 创建云主机实例（AWS EC2 g4dn.xlarge / 阿里云GPU实例）
2. 选择预配置镜像：`Deep Learning AMI (Ubuntu 20.04)`
3. SSH登录：`ssh -i key.pem ubuntu@<instance-ip>`
4. 克隆项目：`git clone https://github.com/yourusername/py_ai_tutorial.git`
5. 按照上述步骤安装依赖（已有CUDA，跳过驱动安装）
6. 运行GPU版本项目

**成本**: AWS g4dn.xlarge约$0.5/小时（按需），使用Spot实例更便宜

### 选项C: Kaggle Notebooks（免费，有GPU配额）

1. 访问 https://www.kaggle.com/code
2. 创建新Notebook，启用GPU
3. 上传数据集到Kaggle Datasets
4. 在Notebook中引用数据集并运行

**优点**: 免费、社区活跃
**缺点**: 数据上传限制、环境定制性较差

---

## 下一步

完成快速开始后，你可以：
1. **继续阶段3其余项目**（9选3，建议完成医疗、电商、金融各1个）
2. **学习模块理论**（回顾`docs/stage3/01-scientific-computing/`等模块）
3. **进阶到阶段4**（深度学习，需GPU或云端环境）
4. **加入学习社群**（交流经验、互相评审代码）
5. **阅读拓展资源**（`docs/stage3/README.md`中的论文与博客链接）

---

## 获取帮助

- **文档**: `docs/`目录下的模块与项目README
- **故障排查**: `docs/cross-platform/troubleshooting.md`
- **GitHub Issues**: https://github.com/yourusername/py_ai_tutorial/issues
- **学习社群**: [Discord/微信群链接]
- **邮件支持**: tutorial@example.com

**祝学习顺利！从数据分析到生成式AI，一步一个脚印。**
