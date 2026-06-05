# Research & Technical Decisions

**Feature**: 阶段3-5教程与跨平台指引
**Date**: 2025-11-05
**Purpose**: 解决技术选型未决项，研究最佳实践，确保设计决策有据可依

## Phase 0: Research Findings

### 1. 静态站点生成器选型

**Decision**: 选择 **MkDocs Material** 作为主要文档站点生成器

**Rationale**:
- **易用性**: Markdown原生支持，配置简单（单一mkdocs.yml），符合目标学员的技术背景
- **中文支持**: 内置中文搜索与字体优化，Material主题完全支持中文排版
- **功能完整**: 支持代码高亮、数学公式（KaTeX）、标签页、警告框等教学必需元素
- **维护活跃**: Material for MkDocs是最活跃的MkDocs主题，2025年仍在持续更新
- **离线能力**: 可生成纯静态HTML，无外部依赖，支持离线学习场景
- **可扩展**: 支持自定义插件（如Jupyter Notebook嵌入、评论系统等）

**Alternatives Considered**:
- **Docusaurus**: React技术栈，对后端工程师门槛略高；中文搜索需额外配置；但在交互性上更强（备选方案）
- **Sphinx**: Python生态标准，但配置复杂（reStructuredText语法学习曲线陡），对非文档专业人员不友好
- **VuePress**: Vue技术栈，国内社区活跃，中文支持好，但生态不如MkDocs Material成熟

**Implementation Notes**:
- 使用mkdocs-material主题（最新8.x版本）
- 集成mkdocs-jupyter插件，直接在文档中嵌入可交互Notebook
- 使用mkdocs-git-revision-date-localized-plugin显示文档更新时间
- 配置中文搜索（jieba分词）与本地化（日期格式、导航文案等）

---

### 2. 包管理工具uv的最佳实践

**Decision**: 采用 **uv** + **pyproject.toml** 作为标准工作流，提供传统pip兼容路径作为备选

**Rationale**:
- **性能**: uv使用Rust编写，依赖解析与安装速度比pip快10-100倍（尤其对大型ML库如PyTorch）
- **准确性**: 基于PubGrub算法的依赖解析，避免pip的依赖冲突与版本回溯问题
- **兼容性**: 完全兼容PyPI与pip，可无缝迁移现有项目
- **现代化**: 遵循PEP 621（pyproject.toml元数据标准），与ruff、mypy等工具集成良好
- **虚拟环境**: uv venv创建隔离环境，与venv/conda相似但更快
- **章程合规**: 明确符合章程原则八"现代化工具链"

**Implementation Notes**:
- 在跨平台指引中提供uv安装步骤（curl | sh，或pipx安装）
- pyproject.toml中明确依赖版本（使用~=指定兼容范围，如numpy~=1.26.0）
- 为兼容传统环境，提供`uv pip compile`生成的requirements.txt作为备选
- 在每个项目模板中包含.python-version文件（uv自动识别Python版本）
- 教程中演示：`uv venv` → `source .venv/bin/activate` → `uv pip install -e .`

**Troubleshooting Guidance**:
- 若uv安装失败（如企业防火墙），提供pipx或pip安装回退路径
- 若依赖解析失败，提供`--resolution=lowest-direct`等调试参数说明
- 对比uv与pip的性能基准（安装PyTorch+依赖：uv ~30s vs pip ~5min）

---

### 3. 跨平台环境差异处理策略

**Decision**: 采用 **分层适配策略** - 通用基线 + OS专属补充 + 云端GPU逃生舱

**Rationale**:
- **覆盖广度**: macOS（Intel/M系列）、Linux（主流发行版）、Windows（原生/WSL2）、云端GPU（Linux CUDA）
- **最小化重复**: 共同部分（Python安装、uv使用、项目克隆）抽取为通用基线文档
- **针对性强**: 每个OS的专属问题（如macOS Xcode CLI、Windows路径编码、WSL2文件系统性能）独立文档
- **容错设计**: 当本地环境遇阻>30分钟，提供云端GPU快速启动路径作为逃生舱

**Platform-Specific Issues & Solutions**:

#### macOS Intel (x86_64)
- **问题**: Homebrew安装路径（/usr/local vs /opt/homebrew）
- **解决**: 脚本自动检测`brew --prefix`，教程说明路径差异
- **问题**: 某些C扩展编译需要Xcode Command Line Tools
- **解决**: 提供`xcode-select --install`前置步骤

#### macOS Apple Silicon (arm64)
- **问题**: TensorFlow原生支持不完整，需tensorflow-macos与tensorflow-metal
- **解决**: pyproject.toml中使用平台标记（`tensorflow-macos; platform_machine=="arm64"`）
- **问题**: PyTorch的MPS后端（Metal Performance Shaders）与CUDA语义差异
- **解决**: 教程中标注MPS不支持的算子（如某些稀疏张量操作），提供CPU回退路径

#### Windows 原生
- **问题**: 路径分隔符（\ vs /）与POSIX命令不兼容
- **解决**: 脚本使用pathlib统一处理；提供PowerShell版本脚本
- **问题**: 编码问题（GBK vs UTF-8）
- **解决**: 所有文件明确指定`encoding="utf-8"`；教程说明设置`chcp 65001`

#### Windows WSL2（推荐）
- **优势**: 接近原生Linux体验，兼容性最好
- **问题**: 文件系统性能（/mnt/c下访问慢）
- **解决**: 建议在WSL2文件系统（~/projects）内操作，避免跨挂载点
- **问题**: GPU支持需WSL2 + CUDA on WSL
- **解决**: 提供CUDA on WSL安装指引与驱动版本检测脚本

#### Linux（Ubuntu/CentOS）
- **问题**: 系统包管理器差异（apt vs yum）
- **解决**: 优先使用uv管理Python依赖，减少对系统包依赖；必需系统包（如libgomp）提供两套安装命令
- **问题**: CUDA版本与驱动匹配
- **解决**: 提供CUDA 11.8/12.1两套依赖配置；脚本自动检测`nvidia-smi`输出并推荐版本

#### 云端GPU（逃生舱）
- **触发条件**: 本地环境配置耗时>30分钟，或CPU训练单epoch>30分钟
- **最小路径**:
  1. 选择云平台（AWS/GCP/阿里云/自托管Jupyter）
  2. 使用预配置镜像（CUDA 11.8 + PyTorch/TensorFlow）
  3. 上传项目代码（git clone或jupyter upload）
  4. 运行GPU版本Notebook
- **成本控制**: 建议使用抢占式实例（spot instance），单小时成本<$1
- **数据同步**: 提供rsync脚本或对象存储（S3）批量上传/下载方案

**Implementation Notes**:
- 创建`scripts/env/detect-platform.py`自动检测OS与硬件配置
- 每个平台文档包含"环境验证清单"（运行`python -c "import torch; print(torch.cuda.is_available())"`等）
- 故障恢复清单（troubleshooting.md）按错误类型索引（依赖冲突、CUDA错误、OOM等）

---

### 4. 数据集管理与离线支持

**Decision**: 采用 **本地优先 + 镜像备份 + 离线包** 三层策略

**Rationale**:
- **可达性**: 部分学习者网络受限（企业内网、教育网），需离线能力
- **可复现**: 固定数据集版本（SHA256校验），避免上游数据更新导致结果差异
- **存储效率**: 使用Parquet/HDF5压缩格式，相比CSV节省50-80%空间

**Data Acquisition Strategy**:

#### 阶段3数据集（传统ML）
- **来源**: UCI ML Repository、Kaggle开放数据、sklearn内置数据集
- **大小**: 单项目<500MB，总计~2GB
- **格式**: CSV（原始） + Parquet（优化后）
- **脚本**: `scripts/data/download-stage3.py`
  - 自动下载UCI/Kaggle数据（需API key或公开URL）
  - 转换为Parquet并生成SHA256校验文件
  - 支持断点续传（检测已下载文件）
- **离线包**: 提供预打包的`stage3-data.tar.gz`（~1.5GB），包含所有项目数据与校验文件

#### 阶段4数据集（深度学习）
- **来源**: ImageNet子集、COCO、MNIST/CIFAR、自制小样本数据
- **大小**: 单项目<2GB，总计~8GB（含预训练模型权重）
- **格式**: 图像（JPEG/PNG） + 标注（COCO JSON/YOLO txt）
- **脚本**: `scripts/data/download-stage4.py`
  - 使用torchvision.datasets自动下载MNIST/CIFAR
  - COCO/ImageNet子集从HuggingFace Hub下载（`huggingface-cli download`）
  - 预训练模型权重缓存到`~/.cache/torch/hub`或`~/.cache/huggingface`
- **离线包**: 提供`stage4-data.tar.gz`（~6GB）+ `stage4-models.tar.gz`（~3GB）
- **云端备选**: 若本地下载失败，提供S3/OSS公开bucket链接（境内加速）

#### 阶段5数据集（LLM/AIGC）
- **来源**: 自制对话数据、HuggingFace datasets（如SQuAD、GLUE子集）
- **大小**: 单项目<1GB，总计~3GB
- **格式**: JSONL（对话轮次） + Parquet（批量数据）
- **脚本**: `scripts/data/download-stage5.py`
  - 使用`datasets.load_dataset`下载HuggingFace数据
  - 支持代理设置（`HF_ENDPOINT`环境变量切换镜像源）
- **离线包**: 提供`stage5-data.tar.gz`（~2GB）
- **LLM模型**: 不包含大模型权重（>10GB），教程使用API调用（DeepSeek/OpenAI）或引导到HuggingFace下载

**Data Verification**:
- 所有数据包附带`checksums.txt`（SHA256）
- 脚本自动运行`scripts/data/verify.py`校验完整性
- 若校验失败，提示重新下载或使用离线包

**Offline Mode Setup**:
1. 下载离线包（百度网盘/阿里云盘/GitHub Releases）
2. 解压到项目data/目录
3. 运行`scripts/data/verify.py --offline`确认完整性
4. 设置环境变量`OFFLINE_MODE=1`跳过网络下载步骤

---

### 5. Jupyter Notebook与纯Python脚本的双轨制

**Decision**: 每个项目提供 **Notebook（教学版） + 脚本（生产版）** 两套实现

**Rationale**:
- **Notebook优势**: 可视化强（实时图表、中间结果）、交互式、适合教学演示
- **脚本优势**: 版本控制友好、CI/CD集成、批量执行、适合生产部署
- **目标学员**: 后端工程师更熟悉脚本化工作流，但需要Notebook的可视化能力辅助理解

**Implementation Strategy**:

#### Notebook（notebooks/目录）
- **结构**: 按阶段组织，每个项目一个.ipynb文件
- **风格**:
  - 每个cell包含注释说明（中文）
  - 关键输出保留（图表、指标）供离线查看
  - 使用魔法命令（%time、%matplotlib inline）增强交互性
- **测试**: 使用nbval插件自动测试Notebook（`pytest --nbval notebooks/`）
- **转换**: 可导出为HTML（`jupyter nbconvert --to html`）或Python脚本（`--to script`）

#### Python脚本（docs/stageX/projects/pXX/src/）
- **结构**: 标准包结构（src/models、src/utils、src/train.py、src/evaluate.py）
- **质量**: 遵循PEP 8、类型注解、文档字符串、单元测试
- **CLI**: 使用argparse或typer提供命令行接口（`python train.py --config configs/default.yaml`）
- **日志**: 使用logging模块，输出到outputs/logs/
- **配置**: 使用YAML配置文件（configs/），支持超参数扫描

#### 同步机制
- **源头**: Notebook作为教学原型，验证正确后抽取为脚本
- **工具**: 使用jupytext双向同步（.ipynb ↔ .py）
- **审查**: 脚本提交需Code Review，Notebook提交需运行验证

**Usage Guidance**:
- **学习路径**: 先学Notebook（理解原理与可视化） → 再学脚本（工程化实践）
- **项目作业**: 允许提交Notebook或脚本（两者评分标准一致）
- **生产部署**: 必须使用脚本版本（Notebook不适合生产环境）

---

### 6. 多框架对比教学的实现方式

**Decision**: **核心项目双实现（PyTorch + TensorFlow）+ 补充材料对比表**

**Rationale**:
- **PyTorch**: 学术界主流（灵活、动态图、调试友好），目标学员未来可能接触研究代码
- **TensorFlow**: 工业界广泛应用（TF Serving、TFLite、分布式训练），生产部署成熟
- **成本控制**: 全部项目双实现工作量大（17个项目 × 2 = 34份代码），采用"核心双实现 + 其余单实现 + 对比文档"策略

**Framework Coverage**:

#### 阶段3（传统ML）
- **主框架**: scikit-learn（无需对比，行业标准）
- **补充**: XGBoost、LightGBM（集成学习项目）

#### 阶段4（深度学习）
- **核心项目双实现**（4个）:
  1. CNN图像分类（CIFAR-10）
  2. 目标检测（YOLO）
  3. RNN文本分类
  4. Transformer翻译
- **PyTorch单实现**（2个）:
  1. 医学影像分割（使用PyTorch生态的monai库）
  2. OCR（使用PaddlePaddle，展示第三方框架）
- **TensorFlow单实现**（1个）:
  1. 工业视觉质检（使用TF Lite部署到边缘设备）

#### 阶段5（AIGC/LLM）
- **主框架**: HuggingFace transformers（统一接口，支持PyTorch/TF后端）
- **LLM SDK**: DeepSeek API、OpenAI SDK（API调用，无框架依赖）
- **训练**: PyTorch + LoRA（轻量微调）

**Comparison Documentation**:
- 创建`docs/framework-comparison.md`对比表:
  - API风格（命令式 vs 声明式）
  - 模型定义（nn.Module vs Keras Sequential）
  - 训练循环（手动 vs fit()）
  - 部署（TorchScript vs SavedModel）
  - 生态（torchvision vs tf.keras.applications）
- 在每个双实现项目中添加"框架差异说明"章节
- 提供"如何在两个框架间迁移代码"的指南

**Code Organization**:
- 双实现项目目录结构:
  ```
  projects/p01-cnn-classification/
  ├── pytorch/
  │   ├── train.py
  │   └── model.py
  ├── tensorflow/
  │   ├── train.py
  │   └── model.py
  ├── notebooks/
  │   ├── pytorch-version.ipynb
  │   └── tensorflow-version.ipynb
  └── README.md  # 框架对比说明
  ```

---

### 7. 评估指标与可复现性标准

**Decision**: 采用 **基准指标范围 + 随机种子控制 + 多次运行统计** 确保可复现

**Rationale**:
- **挑战**: 深度学习模型训练存在随机性（权重初始化、数据增强、dropout），完全确定性复现困难
- **平衡**: 不追求完全一致（不现实），而是定义合理波动范围（±5%）
- **目标**: 学员能验证自己的实现是否正确，而非纠结小数点后几位差异

**Reproducibility Strategy**:

#### 随机种子设置
- 所有项目在脚本/Notebook开头设置统一种子:
  ```python
  import random
  import numpy as np
  import torch

  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
- TensorFlow版本使用`tf.random.set_seed(SEED)`

#### 基准指标定义
- 每个项目在README.md中给出"预期指标范围":
  - 阶段3示例: "逻辑回归准确率应在0.82-0.88之间（5折交叉验证均值）"
  - 阶段4示例: "ResNet-18在CIFAR-10测试集准确率应在92-94%（训练50 epoch）"
  - 阶段5示例: "RAG问答系统ROUGE-L应在0.45-0.55（100条测试样本）"
- 提供"如何判断实验是否成功"的checklist

#### 评估脚本标准化
- 创建`scripts/evaluation/run-eval.py`统一评估流程:
  1. 加载测试数据（固定划分，不参与训练）
  2. 加载模型（从outputs/models/加载checkpoint）
  3. 运行推理（禁用dropout/数据增强）
  4. 计算指标（accuracy、precision、recall、F1、AUROC等）
  5. 生成报告（JSON格式，含时间戳与环境信息）
- 支持多次运行取平均（`--num-runs=5`）

#### 环境信息记录
- 每次实验在outputs/目录生成`experiment_info.json`:
  ```json
  {
    "timestamp": "2025-11-05T10:30:00Z",
    "python_version": "3.11.5",
    "framework": "pytorch",
    "framework_version": "2.1.0",
    "cuda_version": "11.8",
    "device": "cuda:0 (NVIDIA T4)",
    "seed": 42,
    "config": "configs/default.yaml",
    "metrics": {
      "accuracy": 0.934,
      "loss": 0.187
    }
  }
  ```

#### 故障排查指引
- 若学员指标超出范围，提供排查步骤:
  1. 检查随机种子是否正确设置
  2. 检查数据预处理是否一致（归一化、增强方式）
  3. 检查超参数是否匹配（学习率、batch size、epoch数）
  4. 检查框架版本是否一致（提供兼容版本列表）
  5. 若仍无法复现，在GitHub Issues提交实验报告

---

### 8. 时间预估与学习路径设计

**Decision**: 采用 **模块化里程碑 + 自适应路径** 设计，支持不同基础学员

**Rationale**:
- **目标学员差异**: 虽然都是3-5年Python后端经验，但数学基础、AI背景、可投入时间差异大
- **灵活性需求**: 允许跳过已掌握内容、深入感兴趣方向、调整学习节奏

**Time Estimation (per stage)**:

#### 阶段3（机器学习与数据挖掘）
- **理论学习**: 2-3小时
  - 科学计算库（NumPy/Pandas）: 1小时
  - AI数学基础（概率/线代/微积分速览）: 1小时
  - 机器学习概念（监督/无监督/评估）: 0.5-1小时
- **项目实战**: 2-3小时
  - 环境配置: 0.5小时
  - 完成3个小项目（9选3）: 每个0.5-1小时
- **总计**: 4-6小时（1-2天，每天3小时投入）

#### 阶段4（深度学习）
- **理论学习**: 3-4小时
  - 深度学习基础（神经网络/反向传播/优化器）: 1.5小时
  - 计算机视觉（CNN/目标检测/图像分割）: 1小时
  - NLP基础（RNN/Transformer/预训练模型）: 1-1.5小时
- **项目实战**: 3-6小时
  - GPU环境配置（本地或云端）: 0.5-1小时
  - 完成2个小项目（7选2，推荐CV+NLP各1）: 每个1-2小时（GPU），2-4小时（CPU）
- **总计**: 6-10小时（2-3天，每天3小时投入）

#### 阶段5（AIGC与大模型）
- **理论学习**: 2-3小时
  - AIGC/LLM概述（GPT原理/应用场景）: 1小时
  - LLM开发技术（Prompt/RAG/Agent/微调）: 1-2小时
- **项目实战**: 6-9小时
  - LLM API配置与测试: 0.5小时
  - 完成对话系统端到端项目: 5-8小时
    - 数据准备: 1小时
    - RAG搭建: 2-3小时
    - Agent逻辑实现: 1-2小时
    - 测试与优化: 1-2小时
- **总计**: 8-12小时（1-2天全职投入，或3-4天每天3小时）

**Learning Path Options**:

#### 快速路径（最小可行）
- **目标**: 快速入门，建立基本认知
- **内容**: 每阶段只完成1个核心项目 + 基础理论
- **时间**: 总计12-18小时（1周每天2-3小时）
- **适合**: 时间紧张、需快速了解AI全貌的学员

#### 标准路径（推荐）
- **目标**: 系统学习，达到可独立完成小型AI项目的能力
- **内容**: 完整理论 + 每阶段2-3个项目 + 所有平台指引
- **时间**: 总计25-35小时（2-3周每天2-3小时）
- **适合**: 大多数学员

#### 深入路径（进阶）
- **目标**: 深度掌握，能改进模型、优化性能、阅读论文
- **内容**: 所有项目 + 拓展阅读 + 多框架对比实践 + 自定义项目
- **时间**: 总计50-80小时（1-2个月）
- **适合**: 有充裕时间、目标转型AI方向的学员

**Milestone Checklist**:
- 在`docs/learning-path.md`中提供里程碑checklist:
  - [ ] 阶段3-M1: 环境配置完成，能运行示例代码
  - [ ] 阶段3-M2: 完成首个分类项目，理解训练-评估流程
  - [ ] 阶段3-M3: 完成所有阶段3项目，熟练使用scikit-learn
  - [ ] 阶段4-M1: GPU环境就绪（本地或云端），运行首个CNN
  - [ ] 阶段4-M2: 完成CV与NLP各1个项目，理解深度学习训练技巧
  - [ ] 阶段4-M3: 对比PyTorch/TensorFlow实现差异
  - [ ] 阶段5-M1: LLM API调用成功，理解Prompt Engineering
  - [ ] 阶段5-M2: 搭建RAG系统，实现知识检索增强
  - [ ] 阶段5-M3: 完成对话系统端到端项目，可部署演示

**Self-Paced Support**:
- 每个模块标注"前置依赖"（如阶段4需先完成阶段3的模型评估部分）
- 提供"跳过指引"（如已熟悉Pandas可跳过科学计算库章节）
- 设置"卡点检测"（如阶段3测验题，未通过建议重学）

---

## Summary of Decisions

| 决策项 | 选择 | 关键理由 |
|--------|------|----------|
| 静态站点生成器 | MkDocs Material | 易用、中文支持、功能完整、离线能力 |
| 包管理工具 | uv + pyproject.toml | 现代化、高性能、章程合规 |
| 跨平台策略 | 分层适配 + 云端逃生舱 | 覆盖广、容错好、遇阻有退路 |
| 数据管理 | 本地优先 + 离线包 | 可达性、可复现、存储效率 |
| 代码形式 | Notebook + 脚本双轨 | 教学可视化 + 生产工程化 |
| 框架对比 | 核心双实现 + 对比文档 | 平衡学习深度与开发成本 |
| 可复现性 | 指标范围 + 种子控制 | 现实可行、降低挫败感 |
| 学习路径 | 模块化里程碑 + 自适应 | 灵活、支持不同基础学员 |

---

## Next Steps (Phase 1)

基于以上研究结论，Phase 1将生成:
1. **data-model.md**: 定义教程内容的数据模型（Module、Project、Dataset、Rubric等实体）
2. **contracts/**: 定义评估API接口（如自动评分、指标校验）与数据格式规范（项目模板结构）
3. **quickstart.md**: 快速上手指南，覆盖环境配置与首个项目运行
4. **agent context update**: 更新Claude/AI助手的项目上下文，记录技术栈与最佳实践
