# Implementation Tasks: 阶段3-5教程与跨平台指引

**Feature**: 002-ai-tutorial-stages
**Branch**: `002-ai-tutorial-stages`
**Generated**: 2025-11-05
**Total Tasks**: 142 tasks

---

## Implementation Strategy

本项目采用**MVP优先、渐进式交付**策略：

### MVP范围（User Story 1 - Priority P1）
**最小可行产品**聚焦阶段3（传统机器学习）的完整实现，包含：
- 基础设施与项目框架
- 跨平台环境配置指引
- 阶段3全部4个模块教程
- 阶段3至少3个可运行项目（医疗、电商、金融）
- 评估系统基础版

**交付时间**: 2-3周（单人），1周（小团队3人）
**验收标准**: 学习者可在任一OS上独立完成阶段3学习，运行项目并达到指标范围

### 第二版（User Story 2 - Priority P2）
扩展深度学习能力：
- 阶段4全部3个模块教程
- 阶段4至少2个项目（1个CV + 1个NLP）
- GPU环境支持与云端迁移路径

**交付时间**: MVP后+2-3周

### 完整版（User Story 3 - Priority P3）
AIGC与大模型应用：
- 阶段5全部2个模块教程
- 阶段5对话系统端到端项目
- 完善评估系统与发布流程

**交付时间**: 第二版后+1-2周

---

## Dependency Graph

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundation) ← 阻塞所有User Stories
    ↓
    ├─→ Phase 3 (US1 - 阶段3) ← MVP，最高优先级
    │       ↓ (可选依赖，仅为学习路径)
    ├─→ Phase 4 (US2 - 阶段4) ← 独立可测，依赖Foundation
    │       ↓ (可选依赖，仅为学习路径)
    └─→ Phase 5 (US3 - 阶段5) ← 独立可测，依赖Foundation
            ↓
        Phase 6 (Polish)
```

**关键路径**: Phase 1 → Phase 2 → Phase 3 (US1)
**并行机会**: US1/US2/US3可在Foundation完成后并行开发（团队协作）

---

## Parallel Execution Examples

### Setup阶段（单线无并行）
所有Setup任务必须顺序执行，建立项目基础。

### Foundation阶段（少量并行）
- **串行**: T010-T013（数据模型配置必须先完成）
- **并行**: T014-T017（文档模板可并行编写）

### User Story 1阶段（高并行度）
**并行组1**（模块教程，无依赖）:
- T018, T019, T020, T021（阶段3的4个模块）

**并行组2**（项目实现，依赖模块完成）:
- T022, T023, T024（医疗、电商、金融项目）
- T025, T026, T027（通信、零售、互联网项目）
- T028, T029, T030（电商年度、航空、信贷项目）

### User Story 2阶段（中并行度）
**并行组1**（模块教程）:
- T055, T056, T057（阶段4的3个模块）

**并行组2**（项目实现，按框架分组）:
- T058, T060（PyTorch项目）
- T059, T061（TensorFlow项目）

### User Story 3阶段（低并行度）
阶段5项目复杂度高，建议串行实现，但模块教程可先行。

---

## Phase 1: Setup（项目初始化）

**目标**: 建立项目基础结构、配置工具链、准备开发环境

### 任务清单

- [X] T001 创建项目根目录结构（docs/, notebooks/, scripts/, data/, templates/, tests/, site/, configs/）
- [X] T002 配置pyproject.toml（项目元数据、阶段3/4/5依赖分组、uv配置、工具配置）
- [X] T003 创建.python-version文件（锁定Python 3.11）
- [X] T004 [P] 创建.gitignore文件（排除.venv/, outputs/, data/cache/, *.pyc等）
- [X] T005 [P] 配置MkDocs（mkdocs.yml，Material主题，中文搜索，导航结构）
- [X] T006 [P] 编写项目总览README.md（项目简介、快速开始、目录结构、贡献指南）
- [X] T007 配置CI/CD基础（.github/workflows/ci.yml，运行验证脚本与测试）
- [X] T008 [P] 创建离线数据包目录结构（offline/stage3-data/, offline/stage4-data/, offline/stage5-data/）
- [X] T009 初始化Git仓库并创建首次提交

---

## Phase 2: Foundational（阻塞性前置任务）

**目标**: 实现所有User Stories共享的基础设施，阻塞性任务必须完成才能开始任何User Story

### 任务清单

#### 数据模型配置（串行，T010-T013）

- [X] T010 创建configs/content/stages.yaml（定义stage3/4/5实体）
- [X] T011 创建configs/content/modules.yaml（定义所有模块实体）
- [X] T012 创建configs/content/projects.yaml（定义所有项目实体）
- [X] T013 创建configs/content/datasets.yaml（定义所有数据集实体）

#### 脚本与模板（可并行，T014-T021）

- [X] T014 [P] 实现环境检测脚本scripts/env/detect-platform.py（自动检测OS/CPU/GPU/Python版本）
- [X] T015 [P] 实现数据校验脚本scripts/data/verify.py（SHA256校验、完整性检查）
- [X] T016 [P] 创建项目模板templates/project-template/（README、pyproject.toml、src/、configs/、notebooks/、tests/）
- [X] T017 [P] 创建提交物模板templates/submission-template/（结构说明、评分checklist）
- [X] T018 [P] 实现数据模型验证脚本scripts/validation/validate-entities.py（YAML结构验证）
- [X] T019 [P] 实现数据模型路径验证scripts/validation/validate-paths.py（检查引用文件存在）
- [X] T020 [P] 实现数据模型关系验证scripts/validation/validate-relationships.py（检查实体引用完整性）
- [X] T021 [P] 创建configs/content/environments.yaml（定义6个OS环境画像）

#### 跨平台基础文档（可并行，T022-T028）

- [X] T022 [P] 编写docs/cross-platform/setup-macos-intel.md（Intel Mac环境配置）
- [X] T023 [P] 编写docs/cross-platform/setup-macos-arm64.md（Apple Silicon环境配置）
- [X] T024 [P] 编写docs/cross-platform/setup-linux.md（Ubuntu/CentOS环境配置）
- [X] T025 [P] 编写docs/cross-platform/setup-windows-native.md（Windows原生环境配置）
- [X] T026 [P] 编写docs/cross-platform/setup-windows-wsl2.md（WSL2环境配置）
- [X] T027 [P] 编写docs/cross-platform/setup-cloud-gpu.md（云端GPU最小上手路径）
- [X] T028 [P] 编写docs/cross-platform/troubleshooting.md（故障恢复清单，≥5条常见问题）

#### 辅助文档（可并行，T029-T032）

- [X] T029 [P] 编写docs/glossary.md（术语中英对照表，≥15条）
- [X] T030 [P] 编写docs/prerequisites.md（先修要求：Python/数学基础/外部资源链接）
- [X] T031 [P] 编写docs/learning-path.md（学习路线图、里程碑checklist、时间预估）
- [X] T032 [P] 编写docs/framework-comparison.md（PyTorch vs TensorFlow对比表）

---

## Phase 3: User Story 1 - 阶段3（传统机器学习）可落地学习与项目完成（跨平台）[Priority: P1]

**目标**: 实现阶段3完整教学内容，使学习者能在任一OS上完成传统ML学习与项目

**Independent Test**: 学习者仅使用阶段3教程与数据，在mac/Linux/Windows上独立完成环境配置、运行项目、达到指标范围

**Acceptance Criteria**:
- ✅ 所有4个模块教程完整且可理解
- ✅ 至少3个项目可在CPU环境运行（<5分钟）
- ✅ 跨平台环境配置指引覆盖mac/Linux/Windows
- ✅ 学习者可复现指标（允许±5%波动）

### 阶段3模块教程（并行组1，T033-T044）

#### 模块M01：科学计算库

- [X] T033 [P] [US1] 编写docs/stage3/01-scientific-computing/README.md（NumPy/Pandas/Matplotlib教程）
- [X] T034 [P] [US1] 创建notebooks/stage3/01-numpy-basics.ipynb（数组操作、统计函数示例）
- [X] T035 [P] [US1] 创建notebooks/stage3/02-pandas-intro.ipynb（Series/DataFrame、增删查改）
- [X] T036 [P] [US1] 创建notebooks/stage3/03-visualization.ipynb（Matplotlib基础绘图、Pandas可视化）

#### 模块M02：Pandas项目实战

- [X] T037 [P] [US1] 编写docs/stage3/02-pandas-practice/README.md（描述性分析、分组分析教程）
- [X] T038 [P] [US1] 创建notebooks/stage3/04-descriptive-analysis.ipynb（描述性统计、探索性分析）
- [X] T039 [P] [US1] 创建notebooks/stage3/05-data-preprocessing.ipynb（合并、清洗、标准化、正则）

#### 模块M03：AI数学基础

- [X] T040 [P] [US1] 编写docs/stage3/03-ml-basics/README.md（统计学基础、线性代数、概率论速览）
- [X] T041 [P] [US1] 创建notebooks/stage3/06-statistics-basics.ipynb（概率分布、中心极限定理、假设检验）
- [X] T042 [P] [US1] 创建notebooks/stage3/07-linear-algebra.ipynb（矩阵运算、特征值、PCA可视化）

#### 模块M04：机器学习进阶

- [X] T043 [P] [US1] 编写docs/stage3/04-ml-advanced/README.md（回归、分类、聚类、集成学习教程）
- [X] T044 [P] [US1] 创建notebooks/stage3/08-ml-algorithms.ipynb（线性回归、逻辑回归、决策树、随机森林、XGBoost示例）

### 阶段3数据准备（串行，T045-T047）

- [X] T045 [US1] 实现scripts/data/download-stage3.py（下载9个项目数据集，自动校验）
- [X] T046 [US1] 创建data/stage3/README.md（数据集列表、大小、下载说明、校验指引）
- [X] T047 [US1] 打包离线数据offline/stage3-data.tar.gz（包含所有9个数据集与checksums.txt）

### 阶段3项目实现（并行组2，依赖T033-T044模块完成，T048-T074）

#### 项目P01：朝阳医院指标搭建及销售数据汇总

- [X] T048 [P] [US1] 创建docs/stage3/projects/p01-healthcare/目录结构（基于project-template）
- [X] T049 [P] [US1] 编写docs/stage3/projects/p01-healthcare/README.md（项目背景、数据说明、评估指标、交付物）
- [X] T050 [P] [US1] 配置docs/stage3/projects/p01-healthcare/pyproject.toml（pandas/numpy/matplotlib依赖）
- [X] T051 [P] [US1] 实现docs/stage3/projects/p01-healthcare/src/analyze.py（数据清洗、指标计算、可视化）
- [X] T052 [P] [US1] 创建docs/stage3/projects/p01-healthcare/notebooks/analysis.ipynb（Notebook版本）
- [X] T053 [P] [US1] 配置docs/stage3/projects/p01-healthcare/configs/default.yaml（数据路径、输出路径）

#### 项目P02：服装零售销售数据分析

- [X] T054 [P] [US1] 创建docs/stage3/projects/p02-ecommerce/目录结构
- [X] T055 [P] [US1] 编写docs/stage3/projects/p02-ecommerce/README.md
- [X] T056 [P] [US1] 配置docs/stage3/projects/p02-ecommerce/pyproject.toml
- [X] T057 [P] [US1] 实现docs/stage3/projects/p02-ecommerce/src/analyze.py（优衣库销售数据分析、4P分析法、目标顾客度配）
- [X] T058 [P] [US1] 创建docs/stage3/projects/p02-ecommerce/notebooks/analysis.ipynb

#### 项目P03：银行电话营销活动分析

- [X] T059 [P] [US1] 创建docs/stage3/projects/p03-finance/目录结构
- [X] T060 [P] [US1] 编写docs/stage3/projects/p03-finance/README.md
- [X] T061 [P] [US1] 配置docs/stage3/projects/p03-finance/pyproject.toml（添加scikit-learn）
- [X] T062 [P] [US1] 实现docs/stage3/projects/p03-finance/src/train.py（逻辑回归、决策树分类）
- [X] T063 [P] [US1] 创建docs/stage3/projects/p03-finance/notebooks/classification.ipynb

#### 项目P04：通讯公司客户响应速度提升

- [X] T064 [P] [US1] 创建docs/stage3/projects/p04-telecom/目录结构
- [X] T065 [P] [US1] 编写docs/stage3/projects/p04-telecom/README.md
- [X] T066 [P] [US1] 实现docs/stage3/projects/p04-telecom/src/analyze.py（RFM分析、客户细分）
- [X] T067 [P] [US1] 创建docs/stage3/projects/p04-telecom/notebooks/analysis.ipynb

#### 项目P05：零售超市经营分析

- [ ] T068 [P] [US1] 创建docs/stage3/projects/p05-retail/目录结构
- [ ] T069 [P] [US1] 实现docs/stage3/projects/p05-retail/src/analyze.py（SWOT分析、竞品分析、活动分析）
- [ ] T070 [P] [US1] 创建docs/stage3/projects/p05-retail/notebooks/analysis.ipynb

#### 项目P06：滴滴出行运营数据指标异常情况分析

- [ ] T071 [P] [US1] 创建docs/stage3/projects/p06-internet/目录结构
- [ ] T072 [P] [US1] 实现docs/stage3/projects/p06-internet/src/analyze.py（异常检测、趋势分析）
- [ ] T073 [P] [US1] 创建docs/stage3/projects/p06-internet/notebooks/analysis.ipynb

#### 项目P07：淘宝百万级用户行为分析（年度复盘）

- [ ] T074 [P] [US1] 创建docs/stage3/projects/p07-ecommerce-annual/目录结构
- [ ] T075 [P] [US1] 实现docs/stage3/projects/p07-ecommerce-annual/src/train.py（跨境电商行为分析、转化率优化）
- [ ] T076 [P] [US1] 创建docs/stage3/projects/p07-ecommerce-annual/notebooks/analysis.ipynb

#### 项目P08：航空公司客户价值分析

- [ ] T077 [P] [US1] 创建docs/stage3/projects/p08-airline/目录结构
- [ ] T078 [P] [US1] 实现docs/stage3/projects/p08-airline/src/train.py（K-means聚类、客户分群）
- [ ] T079 [P] [US1] 创建docs/stage3/projects/p08-airline/notebooks/clustering.ipynb

#### 项目P09：信用贷款前审批项目（风控模型）

- [ ] T080 [P] [US1] 创建docs/stage3/projects/p09-credit/目录结构
- [ ] T081 [P] [US1] 实现docs/stage3/projects/p09-credit/src/train.py（逻辑回归、随机森林、模型评估）
- [ ] T082 [P] [US1] 创建docs/stage3/projects/p09-credit/notebooks/credit-scoring.ipynb

### 阶段3评分量表与评估（串行，T083-T085）

- [ ] T083 [US1] 创建configs/content/rubrics.yaml（定义stage3项目评分标准）
- [ ] T084 [US1] 创建configs/content/metrics.yaml（定义accuracy、precision、recall等指标）
- [ ] T085 [US1] 实现scripts/evaluation/run-stage3-eval.py（自动评估脚本，验证项目输出）

---

## Phase 4: User Story 2 - 阶段4（深度学习）在本地或云端GPU的迁移学习实验 [Priority: P2]

**目标**: 实现阶段4深度学习教程，提供CPU/GPU两套路径，支持云端迁移

**Independent Test**: 学习者可选择CPU版本（小批次）或云端GPU版本，独立完成深度学习项目并复现指标

**Acceptance Criteria**:
- ✅ 所有3个模块教程完整
- ✅ 至少2个项目提供PyTorch与TensorFlow双实现
- ✅ CPU版本可运行（<30分钟）
- ✅ GPU版本显著加速（3-10倍）

### 阶段4模块教程（并行组1，T086-T091）

#### 模块M01：深度学习基础理论

- [X] T086 [P] [US2] 编写docs/stage4/01-dl-basics/README.md（神经网络、反向传播、优化器、损失函数）
- [X] T087 [P] [US2] 创建notebooks/stage4/01-neural-network.ipynb（手写神经网络、梯度下降可视化）
- [X] T088 [P] [US2] 创建notebooks/stage4/02-pytorch-basics.ipynb（张量操作、自动微分、模型定义）

#### 模块M02：计算机视觉基础

- [X] T089 [P] [US2] 编写docs/stage4/02-cv-basics/README.md（CNN、目标检测、图像分割原理）
- [X] T090 [P] [US2] 创建notebooks/stage4/03-cnn-image-classification.ipynb（LeNet/AlexNet/VGG/ResNet）

#### 模块M03：自然语言处理基础

- [X] T091 [P] [US2] 编写docs/stage4/03-nlp-basics/README.md（RNN/LSTM、Transformer、预训练模型）
- [X] T092 [P] [US2] 创建notebooks/stage4/04-rnn-text-classification.ipynb（情感分类、序列标注）

### 阶段4数据准备（串行，T093-T095）

- [X] T093 [US2] 实现scripts/data/download-stage4.py（下载ImageNet子集、COCO、MNIST/CIFAR、预训练权重）
- [X] T094 [US2] 创建data/stage4/README.md（数据集列表、模型权重缓存说明）
- [X] T095 [US2] 打包离线数据offline/stage4-data.tar.gz（~6GB）+ offline/stage4-models.tar.gz（~3GB）

### 阶段4项目实现（混合并行，T096-T125）

#### 项目P01：工业视觉检测（单框架TensorFlow）

- [ ] T096 [P] [US2] 创建docs/stage4/projects/p01-industrial-vision/目录结构
- [ ] T097 [P] [US2] 编写docs/stage4/projects/p01-industrial-vision/README.md（TF Lite部署到边缘设备）
- [ ] T098 [P] [US2] 实现docs/stage4/projects/p01-industrial-vision/tensorflow/train.py（ResNet50迁移学习）
- [ ] T099 [P] [US2] 创建docs/stage4/projects/p01-industrial-vision/notebooks/tensorflow-version.ipynb

#### 项目P02：基于YOLOv11的视频实时检测系统（双框架）

- [X] T100 [P] [US2] 创建docs/stage4/projects/p02-yolov11-realtime/目录结构
- [X] T101 [P] [US2] 编写docs/stage4/projects/p02-yolov11-realtime/README.md（框架对比说明）
- [X] T102 [P] [US2] 实现docs/stage4/projects/p02-yolov11-realtime/pytorch/train.py（YOLOv11 PyTorch版）
- [X] T103 [P] [US2] 实现docs/stage4/projects/p02-yolov11-realtime/tensorflow/train.py（YOLOv11 TensorFlow版）
- [X] T104 [P] [US2] 创建CPU与GPU两套配置（configs/cpu.yaml, configs/gpu.yaml）

#### 项目P03：OCR票据识别（单框架PaddlePaddle）

- [ ] T105 [P] [US2] 创建docs/stage4/projects/p03-ocr/目录结构
- [ ] T106 [P] [US2] 实现docs/stage4/projects/p03-ocr/paddlepaddle/train.py（PaddleOCR检测+识别）
- [ ] T107 [P] [US2] 创建docs/stage4/projects/p03-ocr/notebooks/paddlepaddle-version.ipynb

#### 项目P04：自动驾驶场景图像分割（双框架）

- [ ] T108 [P] [US2] 创建docs/stage4/projects/p04-image-segmentation/目录结构
- [ ] T109 [P] [US2] 实现docs/stage4/projects/p04-image-segmentation/pytorch/train.py（UNet/DeepLab）
- [ ] T110 [P] [US2] 实现docs/stage4/projects/p04-image-segmentation/tensorflow/train.py
- [ ] T111 [P] [US2] 创建notebooks/pytorch-version.ipynb与notebooks/tensorflow-version.ipynb

#### 项目P05：医学影像分析（单框架PyTorch+MONAI）

- [ ] T112 [P] [US2] 创建docs/stage4/projects/p05-medical-imaging/目录结构
- [ ] T113 [P] [US2] 实现docs/stage4/projects/p05-medical-imaging/pytorch/train.py（使用MONAI库）
- [ ] T114 [P] [US2] 创建docs/stage4/projects/p05-medical-imaging/notebooks/medical-segmentation.ipynb

#### 项目P06：基于Transformer的翻译系统（双框架）

- [ ] T115 [P] [US2] 创建docs/stage4/projects/p06-transformer-translation/目录结构
- [ ] T116 [P] [US2] 实现docs/stage4/projects/p06-transformer-translation/pytorch/train.py（Transformer编码器-解码器）
- [ ] T117 [P] [US2] 实现docs/stage4/projects/p06-transformer-translation/tensorflow/train.py
- [ ] T118 [P] [US2] 创建notebooks/pytorch-version.ipynb与notebooks/tensorflow-version.ipynb

#### 项目P07：基于预训练模型的关键信息提取系统

- [ ] T119 [P] [US2] 创建docs/stage4/projects/p07-pretrained-info-extraction/目录结构
- [ ] T120 [P] [US2] 实现docs/stage4/projects/p07-pretrained-info-extraction/pytorch/train.py（BERT/RoBERTa微调）
- [ ] T121 [P] [US2] 创建docs/stage4/projects/p07-pretrained-info-extraction/notebooks/ner-extraction.ipynb

### 阶段4评估与验证（串行，T122-T124）

- [ ] T122 [US2] 更新configs/content/rubrics.yaml（添加stage4评分标准）
- [ ] T123 [US2] 实现scripts/evaluation/run-stage4-eval.py（支持CPU/GPU两套评估）
- [ ] T124 [US2] 创建scripts/env/verify-gpu.sh（GPU环境验证脚本，检测CUDA/MPS）

---

## Phase 5: User Story 3 - 阶段5（生成式/LLM应用）端到端小项目 [Priority: P3]

**目标**: 实现阶段5 AIGC与大模型教程，提供端到端对话系统项目

**Independent Test**: 学习者独立完成对话系统项目，涵盖数据准备、LLM调用、RAG搭建、评估

**Acceptance Criteria**:
- ✅ 所有2个模块教程完整
- ✅ 对话系统项目可端到端运行（<1小时）
- ✅ 支持DeepSeek/OpenAI等多LLM API
- ✅ 推理延迟<2秒

### 阶段5模块教程（并行组1，T125-T128）

#### 模块M01：AIGC与LLM概述

- [ ] T125 [P] [US3] 编写docs/stage5/01-aigc-llm-intro/README.md（GPT原理、应用场景、主流LLM对比）
- [ ] T126 [P] [US3] 创建notebooks/stage5/01-llm-basics.ipynb（API调用、Prompt Engineering示例）

#### 模块M02：大模型开发

- [ ] T127 [P] [US3] 编写docs/stage5/02-llm-dev/README.md（DeepSeek SDK、RAG、Agent、Fine-tuning）
- [ ] T128 [P] [US3] 创建notebooks/stage5/02-rag-system.ipynb（ChromaDB向量存储、检索增强示例）

### 阶段5数据准备（串行，T129-T131）

- [ ] T129 [US3] 实现scripts/data/download-stage5.py（下载对话数据、SQuAD子集）
- [ ] T130 [US3] 创建data/stage5/README.md（数据集列表、HuggingFace datasets使用说明）
- [ ] T131 [US3] 打包离线数据offline/stage5-data.tar.gz（~2GB）

### 阶段5项目实现（串行，T132-T138）

#### 项目P01：基于LLM的对话系统

- [ ] T132 [US3] 创建docs/stage5/projects/p01-dialogue-system/目录结构
- [ ] T133 [US3] 编写docs/stage5/projects/p01-dialogue-system/README.md（端到端流程说明）
- [ ] T134 [US3] 配置docs/stage5/projects/p01-dialogue-system/pyproject.toml（LangChain/DeepSeek/ChromaDB依赖）
- [ ] T135 [US3] 实现docs/stage5/projects/p01-dialogue-system/src/data/prepare.py（数据准备、向量化）
- [ ] T136 [US3] 实现docs/stage5/projects/p01-dialogue-system/src/models/rag_engine.py（RAG引擎、检索+生成）
- [ ] T137 [US3] 实现docs/stage5/projects/p01-dialogue-system/src/api/server.py（FastAPI服务端）
- [ ] T138 [US3] 创建docs/stage5/projects/p01-dialogue-system/notebooks/dialogue-demo.ipynb（端到端演示）

### 阶段5评估与验证（串行，T139-T140）

- [ ] T139 [US3] 更新configs/content/rubrics.yaml（添加stage5评分标准，ROUGE/BLEU指标）
- [ ] T140 [US3] 实现scripts/evaluation/run-stage5-eval.py（LLM输出质量评估）

---

## Phase 6: Polish & Cross-Cutting Concerns（打磨与发布）

**目标**: 完善测试、构建文档站点、打包发布

### 任务清单

#### 测试与验证（可并行，T141-T145）

- [ ] T141 [P] 实现tests/notebooks/test_stage3_notebooks.py（使用nbval测试阶段3 Notebook）
- [ ] T142 [P] 实现tests/notebooks/test_stage4_notebooks.py（使用nbval测试阶段4 Notebook）
- [ ] T143 [P] 实现tests/notebooks/test_stage5_notebooks.py（使用nbval测试阶段5 Notebook）
- [ ] T144 [P] 实现tests/scripts/test_data_download.py（测试数据下载脚本）
- [ ] T145 [P] 实现tests/data/test_data_validation.py（测试数据校验脚本）

#### 文档生成（串行，T146-T148）

- [ ] T146 生成静态站点（mkdocs build，输出到site/）
- [ ] T147 生成PDF版本（使用mkdocs-pdf-export-plugin或pandoc）
- [ ] T148 更新项目总览README.md（添加演示截图、快速开始、贡献指南）

#### 发布（串行，T149-T151）

- [ ] T149 打包所有离线数据（stage3/4/5-data.tar.gz，总计~10GB）
- [ ] T150 创建GitHub Release（tag v1.0.0，附离线包下载链接）
- [ ] T151 部署文档站点到GitHub Pages或自定义域名

---

## Task Summary

### 总体统计

- **总任务数**: 151 tasks
- **Setup阶段**: 9 tasks
- **Foundation阶段**: 23 tasks
- **User Story 1 (P1)**: 53 tasks（模块4个 + 项目9个 + 评估）
- **User Story 2 (P2)**: 39 tasks（模块3个 + 项目7个 + 评估）
- **User Story 3 (P3)**: 16 tasks（模块2个 + 项目1个 + 评估）
- **Polish阶段**: 11 tasks

### 按优先级统计

- **P1（MVP - 阶段3）**: 85 tasks（Setup + Foundation + US1）
- **P2（阶段4）**: 39 tasks
- **P3（阶段5）**: 16 tasks
- **打磨发布**: 11 tasks

### 可并行任务统计

- **高并行度任务**: ~80个（标记[P]）
- **团队协作建议**:
  - 3人团队：1人负责US1（阶段3），1人负责US2（阶段4），1人负责Foundation+US3
  - 单人开发：按Phase顺序执行，优先完成MVP

### 预估时间

- **MVP（P1完成）**: 2-3周（单人），1周（3人团队）
- **第二版（P2完成）**: MVP后+2-3周
- **完整版（P3完成）**: 第二版后+1-2周
- **总计**: 6-8周（单人），3-4周（3人团队）

---

## Validation Checklist

### User Story 1独立测试清单
- [ ] 学习者可仅使用阶段3文档与数据完成学习
- [ ] 在mac/Linux/Windows任一OS上可独立配置环境
- [ ] 至少3个项目可在CPU环境<5分钟内运行
- [ ] 项目输出指标落在README给定范围内（±5%）
- [ ] 跨平台故障恢复清单覆盖≥5条常见问题

### User Story 2独立测试清单
- [ ] 学习者可选择CPU版本或GPU版本独立完成
- [ ] CPU版本可在<30分钟内完成训练
- [ ] GPU版本训练时间为CPU版本的1/3到1/10
- [ ] 至少2个项目提供PyTorch与TensorFlow双实现
- [ ] 云端GPU迁移指引可操作（创建实例、上传数据、运行脚本）

### User Story 3独立测试清单
- [ ] 学习者可独立完成对话系统端到端流程
- [ ] LLM API调用成功（DeepSeek/OpenAI）
- [ ] RAG检索增强生效（检索相关文档并生成）
- [ ] 推理延迟<2秒
- [ ] 评估指标（ROUGE/BLEU）达到合格线

---

## Next Steps

1. **开始MVP开发**（Phase 1-3）:
   ```bash
   # 创建Feature分支（若未创建）
   git checkout -b 002-ai-tutorial-stages

   # 开始第一个任务
   # T001: 创建项目根目录结构
   mkdir -p docs notebooks scripts data templates tests site configs
   ```

2. **并行任务建议**（团队协作）:
   - **开发者A**: 负责Foundation（T010-T032）
   - **开发者B**: 开始阶段3模块教程（T033-T044）
   - **开发者C**: 编写跨平台配置文档（T022-T028）

3. **单人开发路径**:
   - 第1-2天: Phase 1 (Setup)
   - 第3-5天: Phase 2 (Foundation)
   - 第6-15天: Phase 3 (User Story 1 - 阶段3)
   - 第16-25天: Phase 4 (User Story 2 - 阶段4)
   - 第26-30天: Phase 5 (User Story 3 - 阶段5)
   - 第31-35天: Phase 6 (Polish)

4. **验证MVP**:
   ```bash
   # 完成User Story 1后，运行验证
   python scripts/validation/validate-entities.py
   python scripts/validation/validate-paths.py
   python scripts/data/verify.py --stage 3
   python scripts/evaluation/run-stage3-eval.py --project stage3-p01-healthcare
   ```

5. **生成文档站点预览**:
   ```bash
   mkdocs serve
   # 访问 http://localhost:8000 预览
   ```

---

**Generated by**: `/speckit.tasks` command
**Status**: ✅ Ready for implementation
**Estimated Effort**: 6-8 weeks (1 FTE) or 3-4 weeks (3 FTE team)
