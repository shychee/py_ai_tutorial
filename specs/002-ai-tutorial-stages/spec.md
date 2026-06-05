# Feature Specification: 阶段3-5教程与跨平台指引

**Feature Branch**: `002-ai-tutorial-stages`  
**Created**: 2025-11-05  
**Status**: Draft  
**Input**: User description: "基于PDF的第3-5阶段大纲与项目作业，产出教程与平台差异说明（mac/linux/windows），必要时标注GPU云端执行路径。"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - 阶段3（传统机器学习）可落地学习与项目完成（跨平台）(Priority: P1)

面向初学者到进阶学习者，读者在 mac、Linux 或 Windows 上，根据操作系统差异化指南完成阶段3的环境准备与基础项目（分类/回归与模型评估），不依赖 GPU。

**Why this priority**: 阶段3是后续深度学习与生成式任务的基础，且可在 CPU 环境完成，覆盖人群最广，优先保障落地。

**Independent Test**: 仅基于阶段3教程与数据、在任一 OS 上按“平台差异说明”完成环境配置、运行并复现实验结果（指标与可视化）。

**Acceptance Scenarios**:

1. **Given** 提供 OS 专属环境配置指引，**When** 学习者在 mac 上安装依赖并运行阶段3样例，**Then** 能顺利加载数据、训练并输出指标（如准确率/MAE）与图表。
2. **Given** Windows 用户可选 WSL2 或原生安装路径，**When** 按指引运行项目脚本，**Then** 结果与教程预期一致且无路径/编码问题。
3. **Given** Linux 用户按指引配置 Python 环境与数据缓存目录，**When** 运行评估流程，**Then** 指标落在教程给定范围内（允许±5%随机差异）。

---

### User Story 2 - 阶段4（深度学习）在本地或云端 GPU 的迁移学习实验 (Priority: P2)

当阶段4需要 GPU 时，读者可选择：本地（若具备支持的 GPU/加速框架）或迁移到云端 Linux GPU（提供最小可行路径与成本提示），完成一个小型图像或文本迁移学习项目。

**Why this priority**: 深度学习为阶段5的基础；提供“CPU 可运行的最小版本 + GPU 推荐路径”可平衡可达性与性能需求。

**Independent Test**: 在无 GPU 的 mac 上完成 CPU 教程版；在云端 Linux GPU 上完成加速版（单卡、小数据子集），两者均能独立复现实验与记录对比结果。

**Acceptance Scenarios**:

1. **Given** 教程提供 CPU 与 GPU 两套可执行说明，**When** 学习者选择 CPU 流程，**Then** 能以较小批次/数据子集完成训练并达到基准指标。
2. **Given** 提供云端 Linux GPU 最小上手指引（创建实例/笔记本、数据同步、运行脚本），**When** 学习者在云端执行，**Then** 训练时间显著短于 CPU（例如 3-10 倍）且指标达到教程目标。
3. **Given** 平台差异与常见报错表，**When** 学习者遇到驱动/依赖不兼容，**Then** 可按“故障恢复清单”自助解决或切换到云端方案继续推进。

---

### User Story 3 - 阶段5（生成式/LLM 应用）端到端小项目 (Priority: P3)

读者在本地或云端完成一个端到端的小项目（如：小型问答/检索增强/微型文本分类生成），覆盖数据准备、模型调用/轻量调优或增强、评估与交付物产出。

**Why this priority**: 阶段5体现综合能力与真实应用价值，需要稳定的环境与复现实验路径。

**Independent Test**: 独立使用阶段5教程与模板仓库即可完成从数据到结果的端到端流程，并生成可复现实验记录与报告。

**Acceptance Scenarios**:

1. **Given** 提供端到端项目模板与评估指标定义，**When** 学习者替换数据并执行流程，**Then** 能生成结果指标、样例输出与报告。
2. **Given** 提供“无 GPU 轻量路径 + 有 GPU 加速路径”，**When** 分别执行，**Then** 两者均能得到合格指标，GPU 版具备更好时延/吞吐表现。
3. **Given** 平台差异文档，**When** 在 mac/Windows/Linux 上执行，**Then** 均可按文档完成依赖与环境差异的适配。

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Apple Silicon（arm64）在深度学习依赖上不完全一致：提供 CPU 兼容路线与云端 GPU 迁移步骤（含数据同步与环境镜像化建议）。
- Windows 原生/Python 路径编码与换行符差异：提供 WSL2 优先建议与原生替代指引；脚本需避免硬编码绝对路径与不兼容命令。
- 离线/受限网络环境：提供“离线依赖缓存”与“数据集本地镜像”说明，明确所需空间与校验方式。
- CUDA/驱动/编译器不匹配：列出最小版本矩阵与“先清后装”的恢复步骤；若耗时>30分钟，建议直接切云。
- 数据量/内存不足：提供小样本/分片处理策略与可复现实验的最小数据配置。

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: 教程必须覆盖阶段3-5的核心知识点与每阶段至少1个可执行的小项目作业，保证可在本地 CPU 上完成基础版本。
- **FR-002**: 提供按操作系统划分的“平台差异说明文档”（mac Intel/Apple Silicon、Windows 原生与 WSL2、Linux），包含环境准备、常见报错与故障恢复清单。
- **FR-003**: 对涉及 GPU 的阶段4/5内容，必须提供“CPU 兼容路径（小批次/小数据）”与“云端 Linux GPU 最小上手路径”，并明确何时建议切换到云端。
- **FR-004**: 每个阶段提供可复现的评估指标范围（允许合理随机波动），并给出“结果核验步骤”。
- **FR-005**: 提供标准化的项目模板（数据、配置、脚本入口、日志/结果产出目录结构），并包含“如何提交/展示”的交付要求。
- **FR-006**: 提供数据获取与镜像方案（含离线缓存选项）、数据校验方式与存储空间建议。
- **FR-007**: 提供时间预估（学习+实践）与分阶段里程碑，便于学习者自我管理。
- **FR-008**: 提供“遇阻 30 分钟即查看故障恢复清单或切云”的明确建议与路径，降低卡顿时间。
- **FR-009**: 提供基础术语表与先修要求说明，避免初学者因前置欠缺而无法推进。
- **FR-010**: 教程、模板与文档需避免实现细节绑定单一厂商或专有平台，保持中立与可替换性。

*Clarifications resolved:*

- **FR-011**: 云端 GPU 首选“已有企业/自托管平台”（你方已有云端环境，按需提供）；教程保持中立可移植，附通用自建云主机作为备选与迁移指引。
- **FR-012**: 学员先修与目标深度采用“标准进阶（选项B）”：具备基本线代/概率/微积分；阶段4以迁移学习 + 基础调参为主，阶段5可引入轻量微调或增强（不过度依赖多卡）。
- **FR-013**: 交付采用“中文 + Markdown 静态网站 + Notebook + PDF（选项C）”；文档以中文为主，网站为长期可维护载体，Notebook 可运行，PDF 便于离线分发。

#### Deliverables（与发布形态）

- Markdown 静态站点（教程正文、平台差异独立章节、故障恢复清单、术语表）。
- 可运行 Notebook（阶段3/4/5 配套，CPU 与 GPU 两套入口/配置）。
- PDF 打包版（站点主要章节与平台差异说明的导出稿）。

#### Acceptance Criteria per FR

- FR-001: 每阶段至少提供1个可运行脚本或 Notebook；在任一 OS（mac/Windows/Linux）按文档运行可得到与教程给定范围一致的指标与产出目录。
- FR-002: 提供独立文件《平台差异说明》或章节，覆盖 Apple Silicon、Windows 原生/WSL2、Linux 的安装步骤、常见错误与解决；至少列出5条常见问题与对应恢复步骤。
- FR-003: 同时提供“CPU 路径”（小批次/小数据）与“云端 GPU 路径”（创建环境、同步数据、启动训练、收集结果）；二者均可独立复现实验并达到最低指标线。
- FR-004: 每个阶段提供≥2个核心指标与合格区间（允许±5%随机波动），并给出如何复算与核验的步骤（含随机种子或多次运行建议）。
- FR-005: 项目模板包含 README、data/、configs/、scripts/、outputs/ 结构说明；运行后在 outputs/ 下生成指标文件与样例可视化/样例输出。
- FR-006: 提供数据下载脚本或镜像链接；提供校验（如 SHA256 或条目计数）与最小磁盘需求；离线模式说明如何预拉取与切换源。
- FR-007: 给出阶段学习与实践的时间拆分建议；设置里程碑（环境准备完成、首次运行成功、指标达标、提交物完成）。
- FR-008: 故障恢复清单包含“遇阻>30分钟即执行的替代路径”；列出切换到云端 GPU 的具体判据（如本地一次 epoch 超过 X 分钟）。
- FR-009: 附基础术语表（≥15条）与先修要求（Python/线代/概率统计简述），并附外部学习链接（可替代、非强绑定）。
- FR-010: 文档与模板避免使用单一厂商特有 CLI/SDK 作为唯一路径；若示例出现，必须同时给出中立替代方案或通用步骤。

### Key Entities *(include if feature involves data)*

- **Phase/Module（阶段/模块）**: 表示课程阶段（3/4/5），属性：学习目标、所需时长、项目作业、评估指标。
- **Project（项目）**: 以模板形式提供，属性：数据源说明、任务定义、评估指标、提交格式、示例结果。
- **Environment Profile（环境画像）**: OS 类型、CPU/GPU 可用性、安装/运行步骤、常见问题与恢复步骤。
- **Dataset（数据集）**: 获取路径、镜像/缓存策略、大小/校验信息、许可与使用限制。
- **Rubric（评分量表）**: 指标阈值、定性与定量标准、扣分项与加分项。

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: 80% 学习者可在 60 分钟内完成阶段3环境配置并成功运行基础项目一次（任一 OS）。
- **SC-002**: 70% 学习者可在 90 分钟内完成阶段4的最小可行实验（CPU 版或云端 GPU 版）并记录可复现指标。
- **SC-003**: 60% 学习者可在 1-2 天内完成阶段5端到端小项目，提交包含指标与样例输出的报告。
- **SC-004**: 平台相关问题单（环境/依赖/路径）通过“故障恢复清单”自助解决率≥70%，平均停滞时间≤30分钟。
- **SC-005**: 事后问卷中，对“平台差异说明”与“云端 GPU 路径”的满意度≥4/5。

## Assumptions & Dependencies

- 学习者具备基础 Python 与命令行使用能力；具备安装软件的权限（本地或云端账号）。
- 允许使用开源/公开可得的数据集，或提供自带小样本数据；遵守相应许可条款。
- 如需 GPU，加速环境由云端 Linux 提供（单卡即可）；本地 mac/Windows 可不具备 GPU。
- 网络可访问常见开源镜像源；若受限，需预留本地离线镜像空间（≥10GB 建议）。
- 时间预算：阶段3（4-6小时），阶段4（6-10小时，GPU 版更快），阶段5（1-2天端到端）。

## Out of Scope

- 不提供与单一云厂商强绑定的计费/资源代运维；仅给出通用最小上手路径。
- 不覆盖大型分布式训练、复杂多机多卡并行与高可用部署。
- 不涉及商业数据的采购与清洗；以教学用途小型数据为主。
