# Stage 3 机器学习项目合集

本目录包含Stage 3（机器学习基础）的所有实战项目。

## 📂 项目位置

所有项目位于：`specs/002-ai-tutorial-stages/docs/stage3/projects/`

```
projects/
├── p01-healthcare/          # P01 医院销售分析（EDA实战）
├── p02-ecommerce/           # P02 服装零售RFM分析（客户价值）
└── p03-bank-marketing/      # P03 银行营销分类预测（机器学习）
```

## 🎯 项目列表

### ✅ P01 医院销售分析
- **目录**: `p01-healthcare/`
- **类型**: 探索性数据分析 (EDA)
- **数据**: 生成数据，1,000条记录
- **技能**: 数据清洗、可视化、统计分析
- **输出**: 6张图表、完整分析报告

### ✅ P02 服装零售RFM分析
- **目录**: `p02-ecommerce/`
- **类型**: RFM客户价值分析
- **数据**: 生成数据，2,000条订单，493个客户
- **技能**: RFM模型、客户细分、精准营销
- **输出**: 2张图表（客户分群、RFM热力图）、营销策略

### ✅ P03 银行营销分类预测
- **目录**: `p03-bank-marketing/`
- **类型**: 二分类机器学习
- **数据**: UCI真实数据，45,211条记录
- **技能**: 逻辑回归、决策树、特征工程、模型评估
- **输出**: 5张图表、3个训练好的模型、分类报告

## 🚀 快速开始

### 方式1：运行分析脚本

```bash
# P01 医院销售
cd p01-healthcare
uv run --no-project --with pandas --with numpy --with matplotlib --with seaborn --with pyyaml \
  python src/analyze.py --config configs/default.yaml

# P02 服装零售
cd p02-ecommerce
uv run --no-project --with pandas --with numpy --with matplotlib --with seaborn --with pyyaml \
  python src/analyze.py --config configs/default.yaml

# P03 银行营销
cd p03-bank-marketing
uv run --no-project --with pandas --with numpy --with scikit-learn --with matplotlib --with seaborn --with pyyaml \
  python src/analyze.py --config configs/default.yaml
```

### 方式2：Jupyter Notebook交互式学习

```bash
# 进入任意项目目录
cd p01-healthcare  # 或 p02-ecommerce 或 p03-bank-marketing

# 启动Jupyter
jupyter notebook notebooks/analysis.ipynb
```

## 📊 数据文件位置

所有项目的数据文件位于项目根目录：`data/stage3/`

```bash
data/stage3/
├── hospital_sales.csv      # P01数据（1K行，15个字段）
├── clothing_retail.csv     # P02数据（2K行，21个字段）
└── bank_marketing.csv      # P03数据（45K行，17个字段，UCI真实数据）
```

## 📝 项目结构

每个项目都遵循统一的结构：

```
pXX-project/
├── README.md               # 项目说明文档（算法原理、业务场景）
├── pyproject.toml          # 依赖配置
├── src/
│   ├── __init__.py
│   └── analyze.py          # 分析主脚本
├── notebooks/
│   └── analysis.ipynb      # 交互式教程
├── configs/
│   └── default.yaml        # 配置文件
└── outputs/                # 自动生成
    ├── analysis.log        # 运行日志
    ├── figures/            # 可视化图表
    ├── reports/            # 分析报告
    └── models/             # 训练好的模型（仅P03）
```

## 🎓 学习路径

建议按以下顺序学习：

1. **P01 医院销售** → 掌握数据分析基础（清洗、可视化、统计）
2. **P02 服装零售** → 理解业务模型（RFM客户价值）
3. **P03 银行营销** → 入门机器学习（分类算法、模型评估）

## ⏳ 待实现项目

- P04: 电信客户流失预测（RFM + 流失预测）
- P05: 零售超市SWOT分析（关联规则挖掘）
- P06: 滴滴运营异常检测（时序分析）
- P07: 淘宝电商分析（大规模数据处理）
- P08: 航空客户价值分析（K-means聚类）
- P09: 信贷风险模型（风险评估）

## 📖 详细文档

每个项目的 `README.md` 包含：
- 业务场景说明
- 数据字段详解
- 算法原理讲解（含公式）
- 完整代码示例
- 业务应用建议
- 扩展思考题

## 🔗 相关资源

- **验证报告**: `/tmp/p01_verification.md`, `/tmp/p02_verification.md`, `/tmp/p03_verification.md`
- **数据生成脚本**: `scripts/data/generate-stage3-data.py`
- **Stage 3文档**: `specs/002-ai-tutorial-stages/docs/stage3/`

## ✅ 完成状态

| 项目 | 状态 | 完成时间 | 亮点 |
|-----|------|---------|------|
| P01 医院销售 | ✅ 完成 | 2025-11-13 | 620行代码，30+图表 |
| P02 服装零售 | ✅ 完成 | 2025-11-13 | 7客户群体，RFM模型 |
| P03 银行营销 | ✅ 完成 | 2025-11-13 | 2模型对比，AUC 0.77 |
| P04-P09 | ⏳ 待实现 | - | - |

**总进度**: 3/9 (33.3%)

---

**最后更新**: 2025-11-13
**作者**: Claude Code Assistant
