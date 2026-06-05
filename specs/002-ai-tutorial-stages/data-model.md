# Data Model: 阶段3-5教程与跨平台指引

**Feature**: 002-ai-tutorial-stages
**Date**: 2025-11-05
**Purpose**: 定义教程内容的核心数据实体、关系与验证规则

## Overview

本数据模型描述了AI教程系统的核心实体及其关系，涵盖教学内容（Stage、Module、Project）、环境配置（Environment Profile）、数据管理（Dataset）、评估体系（Rubric、Metric）等。这些实体通过文件系统组织（Markdown文档、YAML配置、代码仓库），而非数据库，但需要保持结构化与一致性。

---

## Core Entities

### 1. Stage（阶段）

**Description**: 课程的顶层组织单位，对应学习者的主要能力进阶阶段。

**Attributes**:
- `id` (string, required): 阶段唯一标识符，如"stage3"、"stage4"、"stage5"
- `name` (string, required): 阶段名称（中文），如"机器学习与数据挖掘"
- `name_en` (string, optional): 阶段名称（英文），如"Machine Learning & Data Mining"
- `description` (string, required): 阶段简介（2-3句话），说明学习目标与核心能力
- `order` (integer, required): 显示顺序，从1开始
- `prerequisites` (list of stage_ids, optional): 前置阶段要求，如stage4依赖stage3
- `estimated_hours` (object, required):
  - `theory_min` (integer): 理论学习最少时长（小时）
  - `theory_max` (integer): 理论学习最多时长（小时）
  - `practice_min` (integer): 项目实践最少时长（小时）
  - `practice_max` (integer): 项目实践最多时长（小时）
- `modules` (list of module_ids, required): 包含的模块列表（有序）
- `projects` (list of project_ids, required): 包含的项目列表
- `learning_outcomes` (list of strings, required): 学习成果描述（3-5条），如"能使用scikit-learn完成分类任务"

**Relationships**:
- **Has Many**: Module（一个阶段包含多个模块）
- **Has Many**: Project（一个阶段包含多个项目）
- **Depends On**: Stage（可选，前置阶段依赖）

**Validation Rules**:
- `id` 必须匹配正则`^stage[0-9]+$`
- `order` 必须唯一且连续
- `estimated_hours.theory_min` ≤ `theory_max`
- `estimated_hours.practice_min` ≤ `practice_max`
- `prerequisites` 中的阶段必须存在且`order`小于当前阶段

**Example**:
```yaml
id: stage3
name: 机器学习与数据挖掘
name_en: Machine Learning & Data Mining
description: 掌握传统机器学习算法（分类、回归、聚类、集成学习）及其在实际业务场景中的应用，熟练使用scikit-learn与数据分析工具栈。
order: 1
prerequisites: []
estimated_hours:
  theory_min: 2
  theory_max: 3
  practice_min: 2
  practice_max: 3
modules:
  - stage3-m01-scientific-computing
  - stage3-m02-pandas-practice
  - stage3-m03-ml-basics
  - stage3-m04-ml-advanced
projects:
  - stage3-p01-healthcare
  - stage3-p02-ecommerce
  # ... 其余7个项目
learning_outcomes:
  - 能使用NumPy进行高效数组运算与数据预处理
  - 能使用Pandas完成数据清洗、探索性分析与特征工程
  - 理解分类、回归、聚类算法原理并能选择合适算法解决业务问题
  - 能使用scikit-learn训练模型、调参与评估，并解释模型结果
  - 能完成端到端的机器学习项目（从数据到模型交付）
```

---

### 2. Module（模块）

**Description**: 阶段内的知识单元，对应一个独立的教学主题（如"科学计算库"、"深度学习基础"）。

**Attributes**:
- `id` (string, required): 模块唯一标识符，如"stage3-m01-scientific-computing"
- `stage_id` (string, required): 所属阶段ID
- `name` (string, required): 模块名称（中文），如"科学计算库"
- `name_en` (string, optional): 模块名称（英文），如"Scientific Computing Libraries"
- `description` (string, required): 模块简介
- `order` (integer, required): 在阶段内的显示顺序
- `estimated_hours` (object, required):
  - `min` (integer): 学习最少时长（小时）
  - `max` (integer): 学习最多时长（小时）
- `content_path` (string, required): 模块内容文档路径（相对于仓库根目录），如"docs/stage3/01-scientific-computing/README.md"
- `topics` (list of strings, required): 涵盖的知识点列表（3-10条），如"NumPy数组操作"、"Pandas数据清洗"
- `notebook_paths` (list of strings, optional): 配套Notebook路径列表
- `skip_conditions` (string, optional): 可跳过条件（自然语言描述），如"已熟练掌握NumPy与Pandas的学员可跳过本模块"

**Relationships**:
- **Belongs To**: Stage
- **Has Many**: Topic（概念性的，不作为独立实体）

**Validation Rules**:
- `id` 必须匹配正则`^stage[0-9]+-m[0-9]+-[a-z-]+$`
- `stage_id` 必须存在于Stage表中
- `order` 在同一阶段内必须唯一且连续
- `content_path` 指向的文件必须存在
- `estimated_hours.min` ≤ `max`

**Example**:
```yaml
id: stage3-m01-scientific-computing
stage_id: stage3
name: 科学计算库
name_en: Scientific Computing Libraries
description: 掌握NumPy数组操作、Pandas数据处理与Matplotlib/Seaborn可视化，为后续数据分析与建模打下基础。
order: 1
estimated_hours:
  min: 1
  max: 1.5
content_path: docs/stage3/01-scientific-computing/README.md
topics:
  - NumPy基本知识：创建Ndarray数组、数组属性、数组操作、统计函数、算数函数
  - Pandas基础知识：创建Series、DataFrame、增删查改操作、Pandas获取数据
  - Matplotlib及Pandas可视化：Matplotlib基础绘图、Pandas数据可视化
  - 描述性分析及分组分析：描述性统计分析、探索性分析、数据透视表、交叉表
  - 数据预处理：合并数据、清洗数据、标准化数据、正则、二手房源数据预处理
notebook_paths:
  - notebooks/stage3/01-numpy-basics.ipynb
  - notebooks/stage3/02-pandas-intro.ipynb
  - notebooks/stage3/03-visualization.ipynb
skip_conditions: 已熟练掌握NumPy与Pandas进行数据分析的学员可跳过本模块，但建议至少浏览一遍确认知识点覆盖一致。
```

---

### 3. Project（项目）

**Description**: 实战项目，学习者通过完成项目来应用所学知识并达成阶段学习目标。

**Attributes**:
- `id` (string, required): 项目唯一标识符，如"stage3-p01-healthcare"
- `stage_id` (string, required): 所属阶段ID
- `name` (string, required): 项目名称（中文），如"朝阳医院指标搭建及销售数据汇总"
- `name_en` (string, optional): 项目名称（英文）
- `description` (string, required): 项目背景与目标（2-3段）
- `order` (integer, required): 在阶段内的显示顺序
- `difficulty` (enum, required): 难度级别，可选值：`easy`、`medium`、`hard`
- `estimated_hours` (object, required):
  - `min` (integer): 完成最少时长（小时）
  - `max` (integer): 完成最多时长（小时）
- `required_modules` (list of module_ids, optional): 前置模块要求
- `datasets` (list of dataset_ids, required): 使用的数据集
- `frameworks` (list of strings, required): 使用的框架/库，如["scikit-learn", "pandas"]
- `code_paths` (object, required):
  - `notebook` (string, optional): Notebook路径
  - `script` (string, optional): Python脚本路径
  - `readme` (string, required): 项目README路径
- `deliverables` (list of strings, required): 交付物要求（3-5条），如"训练好的模型文件"、"评估报告（含指标与可视化）"
- `rubric_id` (string, required): 评分量表ID
- `milestones` (list of objects, required): 项目里程碑
  - `name` (string): 里程碑名称，如"数据加载与探索"
  - `description` (string): 里程碑描述
  - `estimated_hours` (float): 预计时长
- `platform_notes` (object, optional): 跨平台注意事项
  - `macos` (string): macOS专属说明
  - `windows` (string): Windows专属说明
  - `linux` (string): Linux专属说明
  - `gpu_required` (boolean): 是否推荐GPU
  - `gpu_alternative` (string): 无GPU时的替代方案

**Relationships**:
- **Belongs To**: Stage
- **Depends On**: Module（多对多，前置模块）
- **Uses**: Dataset（多对多）
- **Has One**: Rubric（评分量表）

**Validation Rules**:
- `id` 必须匹配正则`^stage[0-9]+-p[0-9]+-.+$`
- `stage_id` 必须存在
- `difficulty` 必须为枚举值之一
- `estimated_hours.min` ≤ `max`
- `required_modules` 中的模块必须存在且属于同一阶段或前置阶段
- `datasets` 中的数据集必须存在
- `rubric_id` 必须存在
- `code_paths.readme` 指向的文件必须存在

**Example**:
```yaml
id: stage3-p01-healthcare
stage_id: stage3
name: 朝阳医院指标搭建及销售数据汇总
name_en: Hospital Metrics & Sales Data Aggregation
description: |
  基于朝阳医院的销售明细数据，完成数据预处理、指标提取与汇总分析，
  为业务决策提供数据支持。涉及Pandas数据清洗、时间序列处理与聚合统计。
order: 1
difficulty: easy
estimated_hours:
  min: 0.5
  max: 1
required_modules:
  - stage3-m01-scientific-computing
  - stage3-m02-pandas-practice
datasets:
  - stage3-ds-healthcare-sales
frameworks:
  - pandas
  - numpy
  - matplotlib
code_paths:
  notebook: notebooks/stage3/p01-healthcare.ipynb
  script: docs/stage3/projects/p01-healthcare/src/analyze.py
  readme: docs/stage3/projects/p01-healthcare/README.md
deliverables:
  - 清洗后的数据文件（Parquet格式）
  - 指标汇总表（CSV，含月度、季度销售额等）
  - 可视化图表（销售趋势、TOP产品等）
  - 实验报告（Markdown，含数据说明、处理步骤、结论）
rubric_id: stage3-rubric-data-analysis
milestones:
  - name: 数据加载与探索
    description: 加载CSV数据，查看数据结构、缺失值与异常值
    estimated_hours: 0.1
  - name: 数据清洗
    description: 处理缺失值、去重、类型转换、时间解析
    estimated_hours: 0.2
  - name: 指标计算
    description: 按时间维度（月/季度）与产品维度聚合销售额、销量
    estimated_hours: 0.2
  - name: 可视化与报告
    description: 绘制趋势图、柱状图，编写实验报告
    estimated_hours: 0.1
platform_notes:
  macos: 无特殊要求，CPU即可运行
  windows: 注意文件路径使用pathlib，避免反斜杠问题
  linux: 无特殊要求
  gpu_required: false
  gpu_alternative: N/A（项目不涉及GPU加速）
```

---

### 4. Dataset（数据集）

**Description**: 项目使用的数据集，包含获取方式、格式与校验信息。

**Attributes**:
- `id` (string, required): 数据集唯一标识符，如"stage3-ds-healthcare-sales"
- `name` (string, required): 数据集名称（中文），如"朝阳医院销售明细"
- `name_en` (string, optional): 数据集名称（英文）
- `description` (string, required): 数据集描述（来源、内容、用途）
- `format` (string, required): 文件格式，如"CSV"、"Parquet"、"HDF5"、"COCO JSON"
- `size_mb` (float, required): 文件大小（MB）
- `num_samples` (integer, optional): 样本数量（如行数、图像数）
- `license` (string, required): 许可协议，如"MIT"、"CC BY 4.0"、"学术使用"
- `source` (object, required): 数据来源
  - `type` (enum): 来源类型，可选值：`builtin`（内置）、`download`（下载）、`kaggle`、`huggingface`、`custom`
  - `url` (string, optional): 下载链接（若type为download/kaggle/huggingface）
  - `mirror_url` (string, optional): 镜像链接（境内加速）
  - `instructions` (string, optional): 获取说明（如需API key）
- `checksum` (object, required): 校验信息
  - `sha256` (string): SHA256哈希值
  - `file_name` (string): 文件名
- `local_path` (string, required): 本地存储路径（相对于data/目录），如"stage3/healthcare_sales.csv"
- `offline_package` (string, optional): 离线包路径（若提供），如"offline/stage3-data.tar.gz"
- `preprocessing_required` (boolean, required): 是否需要预处理
- `preprocessing_script` (string, optional): 预处理脚本路径（若需要）

**Relationships**:
- **Used By**: Project（多对多）

**Validation Rules**:
- `id` 必须匹配正则`^stage[0-9]+-ds-.+$`
- `format` 必须为已知格式之一
- `size_mb` > 0
- `source.type` 必须为枚举值之一
- 若`source.type`为`download`/`kaggle`/`huggingface`，则`url`必须存在
- `checksum.sha256` 必须为64字符十六进制字符串
- 若`preprocessing_required`为true，则`preprocessing_script`必须存在

**Example**:
```yaml
id: stage3-ds-healthcare-sales
name: 朝阳医院销售明细
name_en: Chaoyang Hospital Sales Details
description: 朝阳医院的药品与医疗耗材销售明细数据，包含销售日期、产品名称、数量、金额等字段，用于数据清洗与聚合分析实践。
format: CSV
size_mb: 12.5
num_samples: 50000
license: 学术使用（虚构数据，仅供教学）
source:
  type: custom
  instructions: 数据已包含在仓库中，无需额外下载
local_path: stage3/healthcare_sales.csv
checksum:
  sha256: a3b2c1d4e5f6789...
  file_name: healthcare_sales.csv
offline_package: offline/stage3-data.tar.gz
preprocessing_required: false
preprocessing_script: null
```

---

### 5. Environment Profile（环境画像）

**Description**: 描述不同操作系统与硬件配置的环境配置方案。

**Attributes**:
- `id` (string, required): 环境唯一标识符，如"env-macos-intel"、"env-linux-gpu"
- `name` (string, required): 环境名称，如"macOS Intel"
- `os_type` (enum, required): 操作系统类型，可选值：`macos`、`linux`、`windows`
- `os_variant` (string, optional): 系统变体，如"Apple Silicon (arm64)"、"WSL2"、"Ubuntu 20.04"
- `hardware` (object, required):
  - `cpu_arch` (string): CPU架构，如"x86_64"、"arm64"
  - `gpu_available` (boolean): 是否有GPU
  - `gpu_type` (string, optional): GPU类型，如"NVIDIA GTX 1080"、"Apple M1 GPU"
  - `min_ram_gb` (integer): 最低内存要求（GB）
- `python_version` (string, required): Python版本要求，如">=3.9,<3.13"
- `installation_steps` (list of objects, required): 安装步骤
  - `order` (integer): 步骤顺序
  - `description` (string): 步骤描述
  - `commands` (list of strings): 执行命令
  - `notes` (string, optional): 注意事项
- `common_issues` (list of objects, required): 常见问题与解决方案
  - `symptom` (string): 问题症状
  - `cause` (string): 可能原因
  - `solution` (string): 解决方案
- `verification_script` (string, required): 环境验证脚本路径，如"scripts/env/verify-macos-intel.sh"
- `doc_path` (string, required): 详细文档路径，如"docs/cross-platform/setup-macos-intel.md"

**Relationships**:
- **Independent**: 无直接关系，但被项目的`platform_notes`引用

**Validation Rules**:
- `id` 必须匹配正则`^env-[a-z0-9-]+$`
- `os_type` 必须为枚举值之一
- `hardware.gpu_available` 为true时，`gpu_type`必须存在
- `hardware.min_ram_gb` ≥ 4
- `python_version` 必须为有效的版本约束字符串
- `installation_steps` 必须至少包含1条
- `doc_path` 指向的文件必须存在

**Example**:
```yaml
id: env-macos-intel
name: macOS Intel (x86_64)
os_type: macos
os_variant: Intel芯片（x86_64架构）
hardware:
  cpu_arch: x86_64
  gpu_available: false
  gpu_type: null
  min_ram_gb: 8
python_version: ">=3.9,<3.13"
installation_steps:
  - order: 1
    description: 安装Homebrew（若未安装）
    commands:
      - '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    notes: 若已安装Homebrew，跳过此步骤
  - order: 2
    description: 安装Python 3.11
    commands:
      - brew install python@3.11
      - python3.11 --version
    notes: null
  - order: 3
    description: 安装uv包管理工具
    commands:
      - curl -LsSf https://astral.sh/uv/install.sh | sh
      - uv --version
    notes: 若curl失败，可使用pipx安装：pipx install uv
  - order: 4
    description: 克隆项目并创建虚拟环境
    commands:
      - git clone https://github.com/yourusername/py_ai_tutorial.git
      - cd py_ai_tutorial
      - uv venv --python 3.11
      - source .venv/bin/activate
    notes: null
  - order: 5
    description: 安装项目依赖
    commands:
      - uv pip install -e .
    notes: 首次安装可能需要5-10分钟（下载NumPy、PyTorch等大型库）
common_issues:
  - symptom: 运行`uv`命令时提示"command not found"
    cause: uv安装后未添加到PATH
    solution: 重启终端，或手动执行`source $HOME/.cargo/env`
  - symptom: 安装PyTorch时报错"ld: library not found for -lSystem"
    cause: Xcode Command Line Tools未安装或损坏
    solution: 运行`xcode-select --install`重新安装
  - symptom: 导入NumPy时报错"ImportError: dlopen failed"
    cause: Homebrew安装的Python与系统Python冲突
    solution: 使用完整路径`/opt/homebrew/bin/python3.11`或重新创建虚拟环境
verification_script: scripts/env/verify-macos-intel.sh
doc_path: docs/cross-platform/setup-macos-intel.md
```

---

### 6. Rubric（评分量表）

**Description**: 项目评分标准，定义定量指标与定性评估维度。

**Attributes**:
- `id` (string, required): 评分量表唯一标识符，如"stage3-rubric-data-analysis"
- `name` (string, required): 量表名称，如"数据分析项目评分"
- `description` (string, required): 量表说明
- `total_points` (integer, required): 总分，如100
- `passing_score` (integer, required): 及格分，如60
- `dimensions` (list of objects, required): 评分维度
  - `name` (string): 维度名称，如"代码质量"
  - `weight` (float): 权重（0-1之间，所有维度权重和为1）
  - `criteria` (list of objects): 评分标准
    - `level` (string): 等级，如"优秀"、"良好"、"及格"、"不及格"
    - `description` (string): 等级描述
    - `score_range` (object): 分数范围
      - `min` (integer): 最低分
      - `max` (integer): 最高分
- `quantitative_metrics` (list of objects, required): 定量指标（自动评估）
  - `metric_id` (string): 指标ID，如"accuracy"
  - `name` (string): 指标名称，如"准确率"
  - `unit` (string, optional): 单位，如"%"、"秒"
  - `expected_range` (object): 预期范围
    - `min` (float): 最小值
    - `max` (float): 最大值
  - `threshold` (object): 阈值
    - `pass` (float): 及格线
    - `excellent` (float): 优秀线
  - `weight_in_total` (float): 在总分中的权重

**Relationships**:
- **Used By**: Project（一对一）

**Validation Rules**:
- `id` 必须匹配正则`^stage[0-9]+-rubric-.+$`
- `total_points` > 0
- `passing_score` < `total_points`
- `dimensions` 的`weight`之和必须等于1.0
- `quantitative_metrics` 的`weight_in_total`之和必须≤1.0

**Example**:
```yaml
id: stage3-rubric-data-analysis
name: 数据分析项目评分标准
description: 适用于阶段3数据清洗、探索与可视化类项目的评分量表
total_points: 100
passing_score: 60
dimensions:
  - name: 代码质量
    weight: 0.3
    criteria:
      - level: 优秀
        description: 遵循PEP 8，有类型注解与文档字符串，逻辑清晰，无冗余代码
        score_range: {min: 85, max: 100}
      - level: 良好
        description: 基本遵循PEP 8，有注释但不完整，逻辑较清晰
        score_range: {min: 70, max: 84}
      - level: 及格
        description: 代码可运行但风格不统一，缺少注释
        score_range: {min: 60, max: 69}
      - level: 不及格
        description: 代码风格混乱，难以阅读或无法运行
        score_range: {min: 0, max: 59}
  - name: 数据处理正确性
    weight: 0.4
    criteria:
      - level: 优秀
        description: 正确处理所有缺失值、异常值与重复项，数据类型转换准确
        score_range: {min: 85, max: 100}
      - level: 良好
        description: 处理大部分数据问题，少数边界情况未考虑
        score_range: {min: 70, max: 84}
      - level: 及格
        description: 基本完成数据清洗，但有明显遗漏（如未去重）
        score_range: {min: 60, max: 69}
      - level: 不及格
        description: 数据处理错误导致结果不可信
        score_range: {min: 0, max: 59}
  - name: 分析深度
    weight: 0.2
    criteria:
      - level: 优秀
        description: 进行多维度分析（时间、类别、关联），得出有洞察的结论
        score_range: {min: 85, max: 100}
      - level: 良好
        description: 完成基本统计分析与分组聚合，有初步结论
        score_range: {min: 70, max: 84}
      - level: 及格
        description: 完成简单统计（均值、计数），缺乏深入分析
        score_range: {min: 60, max: 69}
      - level: 不及格
        description: 仅输出原始数据，无有效分析
        score_range: {min: 0, max: 59}
  - name: 可视化质量
    weight: 0.1
    criteria:
      - level: 优秀
        description: 图表类型选择合理，标题、标签、图例完整，美观易读
        score_range: {min: 85, max: 100}
      - level: 良好
        description: 有基本图表，标注不完整但可理解
        score_range: {min: 70, max: 84}
      - level: 及格
        description: 有简单图表但缺少标注或类型不当
        score_range: {min: 60, max: 69}
      - level: 不及格
        description: 无可视化或图表无法理解
        score_range: {min: 0, max: 59}
quantitative_metrics:
  - metric_id: code_pass_rate
    name: 代码运行通过率
    unit: "%"
    expected_range: {min: 80, max: 100}
    threshold: {pass: 80, excellent: 95}
    weight_in_total: 0.2
  - metric_id: data_coverage
    name: 数据覆盖率（处理的数据行数/总行数）
    unit: "%"
    expected_range: {min: 90, max: 100}
    threshold: {pass: 90, excellent: 98}
    weight_in_total: 0.1
```

---

### 7. Metric（指标）

**Description**: 项目评估中使用的量化指标（如准确率、F1、ROUGE等），与Rubric中的quantitative_metrics关联。

**Attributes**:
- `id` (string, required): 指标唯一标识符，如"metric-accuracy"
- `name` (string, required): 指标名称（中文），如"准确率"
- `name_en` (string, required): 指标名称（英文），如"Accuracy"
- `category` (enum, required): 指标类别，可选值：`classification`（分类）、`regression`（回归）、`clustering`（聚类）、`nlp`（NLP）、`cv`（CV）、`general`（通用）
- `formula` (string, required): 计算公式（LaTeX或代码），如"$\frac{TP + TN}{TP + TN + FP + FN}$"
- `code_snippet` (string, optional): 计算代码示例（Python）
- `unit` (string, optional): 单位，如"%"、"分"
- `range` (object, required): 值域
  - `min` (float): 最小值
  - `max` (float): 最大值
- `interpretation` (string, required): 解释说明（如何解读指标值）
- `higher_is_better` (boolean, required): 是否越高越好

**Relationships**:
- **Used By**: Rubric（通过quantitative_metrics引用）

**Validation Rules**:
- `id` 必须匹配正则`^metric-[a-z-]+$`
- `category` 必须为枚举值之一
- `range.min` < `range.max`

**Example**:
```yaml
id: metric-accuracy
name: 准确率
name_en: Accuracy
category: classification
formula: $\frac{TP + TN}{TP + TN + FP + FN}$
code_snippet: |
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_true, y_pred)
unit: null (0-1之间的比例，可转换为%)
range: {min: 0.0, max: 1.0}
interpretation: |
  准确率表示所有预测中正确预测的比例。适用于类别平衡的分类任务。
  当类别极度不平衡时（如欺诈检测），准确率可能产生误导，应配合Precision、Recall、F1使用。
higher_is_better: true
```

---

## Entity Relationships Diagram (Textual)

```
Stage
 ├── has_many: Module
 ├── has_many: Project
 └── depends_on: Stage (optional, prerequisites)

Module
 ├── belongs_to: Stage
 └── has_many: Topic (conceptual, not modeled)

Project
 ├── belongs_to: Stage
 ├── depends_on: Module (many-to-many, required_modules)
 ├── uses: Dataset (many-to-many)
 └── has_one: Rubric

Dataset
 └── used_by: Project (many-to-many)

Environment Profile
 └── referenced_by: Project (via platform_notes, conceptual)

Rubric
 ├── belongs_to: Project (one-to-one)
 └── uses: Metric (via quantitative_metrics)

Metric
 └── used_by: Rubric (many-to-many)
```

---

## File Storage Convention

由于这是教程项目（而非数据库驱动的应用），实体数据存储在结构化文件中：

### Configuration Files（配置文件）
- **Location**: `configs/content/`
- **Format**: YAML
- **Files**:
  - `stages.yaml`: 所有Stage实体
  - `modules.yaml`: 所有Module实体
  - `projects.yaml`: 所有Project实体
  - `datasets.yaml`: 所有Dataset实体
  - `environments.yaml`: 所有Environment Profile实体
  - `rubrics.yaml`: 所有Rubric实体
  - `metrics.yaml`: 所有Metric实体

### Content Files（内容文件）
- **Location**: `docs/`、`notebooks/`、`scripts/`（按项目结构组织）
- **Format**: Markdown、Jupyter Notebook、Python脚本
- **Relationship**: 配置文件通过路径字段（`content_path`、`notebook_paths`、`code_paths`等）引用实际内容文件

### Example Structure:
```
configs/content/
├── stages.yaml
├── modules.yaml
├── projects.yaml
├── datasets.yaml
├── environments.yaml
├── rubrics.yaml
└── metrics.yaml

docs/
└── stage3/
    ├── 01-scientific-computing/
    │   └── README.md  # Module内容
    └── projects/
        └── p01-healthcare/
            ├── README.md  # Project说明
            └── src/
                └── analyze.py  # Project脚本

notebooks/
└── stage3/
    └── p01-healthcare.ipynb  # Project Notebook

data/
└── stage3/
    └── healthcare_sales.csv  # Dataset文件
```

---

## Data Integrity & Validation

### Validation Scripts
- **Location**: `scripts/validation/`
- **Scripts**:
  - `validate-entities.py`: 验证所有YAML配置文件的结构与约束
  - `validate-paths.py`: 验证所有路径字段指向的文件存在
  - `validate-checksums.py`: 验证数据集SHA256校验和
  - `validate-relationships.py`: 验证实体间引用关系（如project.stage_id存在于stages中）

### CI/CD Integration
- 在GitHub Actions中运行验证脚本，PR提交前必须通过
- 示例工作流:
  ```yaml
  - name: Validate Data Model
    run: |
      python scripts/validation/validate-entities.py
      python scripts/validation/validate-paths.py
      python scripts/validation/validate-relationships.py
  ```

### Schema Definition
- 使用JSON Schema或Pydantic定义实体模型，自动生成验证逻辑
- 示例（Pydantic）:
  ```python
  from pydantic import BaseModel, Field, validator
  from typing import List, Optional
  from enum import Enum

  class DifficultyLevel(str, Enum):
      easy = "easy"
      medium = "medium"
      hard = "hard"

  class EstimatedHours(BaseModel):
      min: int = Field(..., ge=0)
      max: int = Field(..., ge=0)

      @validator('max')
      def max_gte_min(cls, v, values):
          if 'min' in values and v < values['min']:
              raise ValueError('max must be >= min')
          return v

  class Project(BaseModel):
      id: str = Field(..., regex=r'^stage[0-9]+-p[0-9]+-.+$')
      stage_id: str
      name: str
      difficulty: DifficultyLevel
      estimated_hours: EstimatedHours
      # ... 其他字段
  ```

---

## State Transitions（状态转换）

虽然教程内容实体大多静态（由开发者编写，非用户生成），但学习者的进度状态需要追踪：

### Learner Progress（学习者进度）- 扩展实体

**Description**: 追踪学习者在各阶段/模块/项目的进度（可选，用于未来的学习管理系统）

**Attributes**:
- `learner_id` (string): 学习者唯一标识
- `entity_type` (enum): 实体类型，可选值：`stage`、`module`、`project`
- `entity_id` (string): 实体ID
- `status` (enum): 状态，可选值：
  - `not_started`: 未开始
  - `in_progress`: 进行中
  - `completed`: 已完成
  - `skipped`: 已跳过
- `started_at` (datetime, optional): 开始时间
- `completed_at` (datetime, optional): 完成时间
- `actual_hours` (float, optional): 实际耗时（小时）
- `score` (integer, optional): 项目得分（仅Project）
- `submission_path` (string, optional): 提交物路径（仅Project）

**State Transition Rules**:
- `not_started` → `in_progress`: 学习者开始学习
- `in_progress` → `completed`: 学习者完成并通过评估（Project需要score ≥ passing_score）
- `in_progress` → `skipped`: 学习者跳过（仅Module允许）
- `not_started` → `skipped`: 学习者直接跳过（仅Module允许）

**Example**:
```yaml
learner_id: user_12345
entity_type: project
entity_id: stage3-p01-healthcare
status: completed
started_at: 2025-11-05T09:00:00Z
completed_at: 2025-11-05T10:30:00Z
actual_hours: 1.5
score: 85
submission_path: submissions/user_12345/stage3-p01/
```

---

## Summary

本数据模型定义了AI教程系统的7个核心实体（Stage、Module、Project、Dataset、Environment Profile、Rubric、Metric）及其关系，支持：
1. **结构化内容组织**: 按阶段-模块-项目三层组织教学内容
2. **跨平台适配**: 通过Environment Profile管理不同OS/硬件配置
3. **数据管理**: 通过Dataset管理数据获取、校验与离线支持
4. **评估体系**: 通过Rubric与Metric实现定量+定性评估
5. **可扩展性**: 预留Learner Progress实体，支持未来的学习管理功能

所有实体存储在YAML配置文件中，通过脚本验证数据完整性与引用关系，确保教程内容的一致性与可维护性。
