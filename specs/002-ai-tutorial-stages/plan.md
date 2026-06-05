# Implementation Plan: 阶段3-5教程与跨平台指引

**Branch**: `002-ai-tutorial-stages` | **Date**: 2025-11-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-ai-tutorial-stages/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

构建覆盖阶段3（机器学习与数据挖掘）、阶段4（深度学习）、阶段5（AIGC与大模型）的渐进式AI教程体系，为具备3-5年Python后端经验的工程师提供从传统机器学习到生成式AI的完整学习路径。教程采用实战驱动方式，提供跨平台（macOS/Linux/Windows）环境配置指引、CPU/GPU两套执行路径、可复现的项目模板和故障恢复清单，交付形式为Markdown静态网站 + Jupyter Notebook + PDF。

## Technical Context

**Language/Version**: Python ≥3.9（推荐3.11+以获得更好的类型检查和性能）
**Primary Dependencies**:
- 阶段3: NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- 阶段4: PyTorch, TensorFlow, OpenCV, PaddlePaddle, torchvision, transformers
- 阶段5: LangChain, DeepSeek SDK, HuggingFace transformers, ChromaDB, LoRA/QLoRA库
- 文档生成: MkDocs/Docusaurus, Jupyter Lab
- 包管理: uv（现代化包管理工具，遵循章程原则八）

**Storage**:
- 数据集: 本地文件系统（HDF5/Parquet格式）+ 可选云端对象存储（S3兼容）
- 模型权重: 本地缓存 + HuggingFace Hub镜像
- 实验记录: MLflow或本地JSON/CSV日志
- 数据库: 不涉及（教程场景下使用文件即可）

**Testing**:
- pytest用于代码单元测试
- nbval用于Notebook测试
- 数据验证: Great Expectations或自定义校验脚本
- 模型性能: 自定义评估脚本（含指标阈值断言）

**Target Platform**:
- macOS（Intel x86_64 + Apple Silicon arm64）
- Linux（Ubuntu 20.04+, CentOS 8+）
- Windows 10/11（原生 + WSL2优先建议）
- 云端GPU: Linux（CUDA 11.8+, 单卡T4/V100/A10级别）

**Project Type**: 教程文档项目（单一仓库，包含文档源码、Notebook、数据脚本）

**Performance Goals**:
- 阶段3项目: CPU环境下单次训练<5分钟
- 阶段4项目: CPU可运行（小批次<30分钟），GPU加速版<10分钟
- 阶段5项目: 推理延迟<2秒（API调用），端到端流程<1小时（含数据准备）

**Constraints**:
- 环境配置时间<60分钟（任一OS）
- 单个项目存储占用<5GB（含数据与模型缓存）
- 离线模式支持（提供预下载包，<10GB）
- 故障自助解决率≥70%，平均停滞时间≤30分钟

**Scale/Scope**:
- 3个阶段，每阶段2-4个核心模块
- 阶段3: 9个小项目（医疗、电商、金融、通信、零售、互联网、电商年度复盘、航空、信贷）
- 阶段4: 7个小项目（工业视觉、YOLOv11、OCR、图像分割、医学影像、Transformer翻译、预训练模型信息提取）
- 阶段5: 1个综合项目（对话系统）
- 15+术语中英对照，5+常见故障恢复步骤
- 预计学习路径: 阶段3（4-6小时），阶段4（6-10小时），阶段5（1-2天）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### 原则一：循序渐进式教学
✅ **合规**: 阶段3→4→5明确前置依赖，阶段3覆盖传统ML基础，阶段4引入DL，阶段5聚焦LLM/AIGC

### 原则二：实战驱动开发
✅ **合规**: 每阶段包含多个可运行项目（阶段3: 9个，阶段4: 7个，阶段5: 1个综合），理论<30%，实践≥70%

### 原则三：代码质量标准
✅ **合规**: 所有代码遵循PEP 8、类型注解、文档字符串、异常处理、日志记录与单元测试

### 原则四：数学概念通俗化
✅ **合规**: 数学公式配备Python实现与可视化（Matplotlib/Seaborn），提供类比与直观解释

### 原则五：多框架对比教学
✅ **合规**: 阶段4项目提供PyTorch与TensorFlow两版本，阶段4还包含PaddlePaddle示例；阶段5涵盖LangChain与多LLM API对比

### 原则六：版本兼容性管理
✅ **合规**: 所有项目提供pyproject.toml（精确依赖版本），Python≥3.9，提供Docker/conda环境配置

### 原则七：中文优先，双语支持
✅ **合规**: 文档与注释中文为主，技术术语首次出现标注英文，提供≥15条术语中英对照表

### 原则八：现代化工具链
✅ **合规**: 优先使用uv作为包管理工具，pyproject.toml作为项目配置文件，教程包含uv安装与使用指南

**结论**: 🟢 通过所有章程检查，无需豁免，可进入Phase 0研究

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# 教程文档项目结构（单一仓库）
docs/
├── stage3/                          # 阶段3：机器学习与数据挖掘
│   ├── 01-scientific-computing/     # 科学计算库（NumPy/Pandas/Matplotlib）
│   ├── 02-pandas-practice/          # Pandas项目实战
│   ├── 03-ml-basics/                # AI数学基础
│   ├── 04-ml-advanced/              # 机器学习进阶（回归/分类/集成）
│   └── projects/                    # 9个小项目
│       ├── p01-healthcare/
│       ├── p02-ecommerce/
│       ├── p03-finance/
│       ├── p04-telecom/
│       ├── p05-retail/
│       ├── p06-internet/
│       ├── p07-ecommerce-annual/
│       ├── p08-airline/
│       └── p09-credit/
├── stage4/                          # 阶段4：深度学习
│   ├── 01-dl-basics/                # 深度学习基础理论
│   ├── 02-cv-basics/                # 计算机视觉基础
│   ├── 03-nlp-basics/               # 自然语言处理基础
│   └── projects/                    # 7个小项目
│       ├── p01-industrial-vision/
│       ├── p02-yolov11-realtime/
│       ├── p03-ocr/
│       ├── p04-image-segmentation/
│       ├── p05-medical-imaging/
│       ├── p06-transformer-translation/
│       └── p07-pretrained-info-extraction/
├── stage5/                          # 阶段5：AIGC与大模型
│   ├── 01-aigc-llm-intro/           # AIGC与LLM概述
│   ├── 02-llm-dev/                  # 大模型开发（DeepSeek/Prompt/RAG/Agent/Fine-tuning）
│   └── projects/                    # 1个综合项目
│       └── p01-dialogue-system/
├── cross-platform/                  # 跨平台指引
│   ├── setup-macos-intel.md
│   ├── setup-macos-arm64.md
│   ├── setup-linux.md
│   ├── setup-windows-native.md
│   ├── setup-windows-wsl2.md
│   ├── setup-cloud-gpu.md
│   └── troubleshooting.md           # 故障恢复清单
├── glossary.md                      # 术语中英对照表
├── prerequisites.md                 # 先修要求说明
└── learning-path.md                 # 学习路线图与里程碑

notebooks/
├── stage3/                          # 阶段3配套Notebook
│   └── [项目对应的.ipynb文件]
├── stage4/                          # 阶段4配套Notebook（CPU+GPU两套）
│   ├── cpu-version/
│   └── gpu-version/
└── stage5/                          # 阶段5配套Notebook
    ├── cpu-version/
    └── gpu-version/

scripts/
├── data/                            # 数据获取与预处理脚本
│   ├── download.py
│   ├── verify.py
│   └── mirror-offline.py
├── env/                             # 环境检测与配置脚本
│   ├── check-deps.py
│   ├── setup-uv.sh
│   └── setup-docker.sh
└── evaluation/                      # 评估与验证脚本
    ├── run-stage3-eval.py
    ├── run-stage4-eval.py
    └── run-stage5-eval.py

data/
├── stage3/                          # 阶段3数据集（或下载说明）
├── stage4/                          # 阶段4数据集
└── stage5/                          # 阶段5数据集

templates/                           # 项目模板
├── project-template/
│   ├── README.md
│   ├── pyproject.toml
│   ├── data/
│   ├── configs/
│   ├── src/
│   ├── scripts/
│   ├── outputs/
│   └── tests/
└── submission-template/             # 提交物模板

tests/
├── notebooks/                       # Notebook测试（nbval）
├── scripts/                         # 脚本单元测试（pytest）
└── data/                            # 数据校验测试

site/                                # 生成的静态站点（MkDocs输出）
mkdocs.yml                           # MkDocs配置
pyproject.toml                       # 项目配置与依赖管理（uv）
.python-version                      # Python版本锁定
README.md                            # 项目总览
```

**Structure Decision**: 采用单一仓库教程项目结构，按阶段（stage3/4/5）组织教学内容，每阶段包含理论模块（docs/）与配套Notebook（notebooks/），提供跨平台指引（cross-platform/）与项目模板（templates/）。数据与脚本独立管理，生成的静态站点输出到site/目录。此结构便于学习者按阶段推进，同时支持多OS环境与CI/CD集成。

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**无需填写** - 所有章程检查已通过，无违规项需要豁免或复杂度追踪。

---

## Phase 0-1 Completion Summary

### Artifacts Generated

✅ **Phase 0: Research & Technical Decisions**
- `research.md`: 8个关键技术决策（静态站点生成器、包管理工具、跨平台策略、数据管理、代码形式、框架对比、可复现性、学习路径）
- 所有NEEDS CLARIFICATION项已解决

✅ **Phase 1: Design & Contracts**
- `data-model.md`: 7个核心实体定义（Stage、Module、Project、Dataset、Environment Profile、Rubric、Metric）+ 关系图 + 验证规则
- `contracts/project-template-schema.yaml`: 标准项目结构模板（目录树、文件规范、代码质量标准、交付物清单、评估接口）
- `contracts/evaluation-api.yaml`: 评估系统OpenAPI规范（评估提交、状态查询、项目列表、指标定义API）
- `quickstart.md`: 快速上手指南（环境配置、数据准备、首个项目运行、故障排查、云端GPU迁移）

✅ **Agent Context Update**
- `CLAUDE.md`: 更新技术栈（Python ≥3.9、uv、PyTorch/TensorFlow、LangChain等）

### Constitution Check Re-evaluation

**Phase 1设计完成后的章程复查结果**:

- ✅ **原则一（循序渐进式教学）**: 数据模型中明确定义了Stage的前置依赖（prerequisites）与Module的顺序（order），确保学习路径清晰
- ✅ **原则二（实战驱动开发）**: Project实体占据核心地位（17个项目 vs 10+模块），每个项目包含Notebook与脚本双轨实现
- ✅ **原则三（代码质量标准）**: project-template-schema.yaml中强制要求PEP 8、类型注解、文档字符串、异常处理、日志记录，评估API包含code_quality_check
- ✅ **原则四（数学概念通俗化）**: Module实体包含topics字段，要求配备可视化（notebooks）与代码实现
- ✅ **原则五（多框架对比教学）**: Project实体的frameworks字段支持多框架，research.md明确"核心项目双实现（PyTorch + TensorFlow）"策略
- ✅ **原则六（版本兼容性管理）**: project-template-schema.yaml强制要求pyproject.toml（精确依赖版本）与.python-version（版本锁定）
- ✅ **原则七（中文优先，双语支持）**: 所有实体包含name（中文）与name_en（英文）字段，project-template要求中文注释与中英术语对照
- ✅ **原则八（现代化工具链）**: research.md选择uv + pyproject.toml，quickstart.md包含完整uv安装与使用指引

**结论**: 🟢 **Phase 1设计通过章程复查，无新增违规项，可进入Phase 2（任务生成）**

---

## Next Steps: Phase 2 (NOT executed by /speckit.plan)

**Note**: 以下Phase 2任务由 `/speckit.tasks` 命令执行（而非本计划命令），此处仅作为参考。

### Phase 2 Scope

根据Phase 0-1的设计产物，Phase 2将生成`tasks.md`，包含以下任务类别：

#### 2.1 基础设施任务
- 创建仓库目录结构（docs/、notebooks/、scripts/、data/、templates/、tests/、site/）
- 配置pyproject.toml（项目元数据、依赖管理、工具配置）
- 配置MkDocs（mkdocs.yml、主题、插件、导航结构）
- 编写环境检测脚本（scripts/env/detect-platform.py）
- 编写数据下载脚本（scripts/data/download-stage3/4/5.py）
- 编写数据验证脚本（scripts/data/verify.py）
- 配置CI/CD（GitHub Actions，运行测试、校验数据模型、构建文档站点）

#### 2.2 内容创建任务（阶段3: 机器学习与数据挖掘）
- **模块M01**: 科学计算库（NumPy/Pandas/Matplotlib）
  - 编写教程文档（docs/stage3/01-scientific-computing/README.md）
  - 创建配套Notebook（notebooks/stage3/01-numpy-basics.ipynb等）
- **模块M02**: Pandas项目实战
- **模块M03**: AI数学基础
- **模块M04**: 机器学习进阶
- **项目P01-P09**: 9个行业项目（医疗、电商、金融、通信、零售、互联网、电商年度复盘、航空、信贷）
  - 每个项目包含：README、pyproject.toml、src/、notebooks/、configs/、tests/

#### 2.3 内容创建任务（阶段4: 深度学习）
- **模块M01**: 深度学习基础理论
- **模块M02**: 计算机视觉基础
- **模块M03**: 自然语言处理基础
- **项目P01-P07**: 7个项目（工业视觉、YOLOv11、OCR、图像分割、医学影像、Transformer翻译、预训练模型信息提取）
  - 核心项目提供PyTorch与TensorFlow双实现

#### 2.4 内容创建任务（阶段5: AIGC与大模型）
- **模块M01**: AIGC/LLM概述
- **模块M02**: 大模型开发（DeepSeek/Prompt/RAG/Agent/Fine-tuning）
- **项目P01**: 对话系统端到端项目

#### 2.5 跨平台支持任务
- 编写环境配置文档（docs/cross-platform/setup-*.md，6个OS变体）
- 编写故障恢复清单（docs/cross-platform/troubleshooting.md）
- 创建环境验证脚本（scripts/env/verify-*.sh，6个OS变体）

#### 2.6 评估系统任务
- 实现评估API后端（FastAPI，评估提交、状态查询、项目列表、指标计算）
- 实现自动评分引擎（结构验证、代码质量检查、指标计算、Rubric评分）
- 编写评估CLI（scripts/evaluation/run-eval.py）
- 创建评分量表配置（configs/content/rubrics.yaml）

#### 2.7 辅助文档任务
- 编写术语中英对照表（docs/glossary.md，≥15条）
- 编写先修要求说明（docs/prerequisites.md）
- 编写学习路线图（docs/learning-path.md，包含里程碑checklist）
- 编写框架对比文档（docs/framework-comparison.md）

#### 2.8 测试与验证任务
- 编写数据模型验证脚本（scripts/validation/validate-*.py）
- 编写Notebook测试（使用nbval）
- 编写项目示例测试（pytest）
- 编写端到端测试（完整学习路径模拟）

#### 2.9 发布与交付任务
- 生成静态站点（mkdocs build）
- 生成PDF版本（pandoc或wkhtmltopdf）
- 打包离线数据（stage3/4/5-data.tar.gz）
- 编写项目总览README
- 创建GitHub Releases（附离线包下载链接）

### Estimated Task Count

- **总任务数**: ~120-150个（按模块、项目、文档、脚本细分）
- **关键路径**: 数据模型配置 → 阶段3内容 → 阶段4内容 → 阶段5内容 → 评估系统 → 发布
- **并行性**: 不同阶段的模块与项目可并行开发（团队协作）
- **时间估算**:
  - 单人全职: 3-4个月
  - 小团队（3人）: 1-2个月
  - 内容复用（已有部分教程）: 可缩短至2-4周

---

## Appendix: Plan Workflow Execution Log

**Command**: `/speckit.plan "开始指定计划吧,可以参考[Image #2][Image #3][Image #4][Image #5][Image #6]"`

**Execution Steps**:
1. ✅ Setup: 运行setup-plan.sh，解析特性ID、分支、路径
2. ✅ Load context: 读取spec.md与constitution.md
3. ✅ Technical Context: 填充语言、依赖、平台、性能目标、约束、范围
4. ✅ Constitution Check: 验证所有8条章程原则，通过无违规
5. ✅ Project Structure: 定义教程项目目录树（docs/、notebooks/、scripts/、data/等）
6. ✅ Phase 0: 生成research.md（8个技术决策，解决所有未决项）
7. ✅ Phase 1: 生成data-model.md（7个实体 + 关系 + 验证规则）
8. ✅ Phase 1: 生成contracts/（project-template-schema.yaml + evaluation-api.yaml）
9. ✅ Phase 1: 生成quickstart.md（6步环境配置 + 首个项目运行 + FAQ）
10. ✅ Phase 1: 更新agent context（CLAUDE.md技术栈）
11. ✅ Constitution Re-check: Phase 1设计通过章程复查
12. ✅ Stop: 命令结束（Phase 2由/speckit.tasks执行）

**Branch**: `002-ai-tutorial-stages`
**Plan File**: `/Users/hanlinqi/Desktop/Code/AICode/py_ai_tutorial/specs/002-ai-tutorial-stages/plan.md`
**Generated Artifacts**:
- `research.md` (8 decisions)
- `data-model.md` (7 entities)
- `contracts/project-template-schema.yaml` (project structure)
- `contracts/evaluation-api.yaml` (OpenAPI spec)
- `quickstart.md` (快速开始指南)
- `CLAUDE.md` (agent context, updated)

**Status**: ✅ **Phase 0-1 Complete. Ready for Phase 2 (/speckit.tasks).**
