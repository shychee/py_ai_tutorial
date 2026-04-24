# Stage 3 数据集说明

本目录包含阶段3（机器学习与数据挖掘）的项目数据集。

## 数据集概览

| 项目 | 数据集 | 来源 | 大小 | 状态 |
|------|--------|------|------|------|
| P01 朝阳医院销售分析 | `hospital_sales.csv` | 合成数据 | 176KB | 已包含在仓库中 |
| P02 服装零售分析 | `clothing_retail.csv` | 合成数据 | 311KB | 已包含在仓库中 |
| P03 银行营销分析 | `bank_marketing.csv` | UCI ML Repository | 3.5MB | 已包含在仓库中 |
| P04 电信客户流失预测 | `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Kaggle (IBM) | ~1MB | 需手动下载 |
| P05-P09 | — | — | — | Coming Soon |

## 快速开始

### P01-P03：直接使用

这三个数据集已包含在 Git 仓库中，clone 后即可使用：

```bash
# 验证数据文件存在
ls data/stage3/*.csv
```

### P04：从 Kaggle 下载

P04 使用 IBM Telco Customer Churn 数据集（真实数据），需要从 Kaggle 下载：

1. 访问 https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. 点击 "Download" 下载 `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. 将文件放到 `data/stage3/` 目录下

或使用 Kaggle CLI：

```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/stage3/ --unzip
```

### 使用下载脚本

也可以使用项目提供的下载脚本查看数据状态：

```bash
python scripts/data/download-stage3.py --verify-only
```

## 数据集详细说明

### P01: 朝阳医院销售数据（合成）

- **文件**: `hospital_sales.csv`
- **行数**: 1,000
- **列数**: 18
- **用途**: 数据清洗、分组聚合、时间序列分析
- **许可证**: MIT

### P02: 服装零售销售数据（合成）

- **文件**: `clothing_retail.csv`
- **行数**: 2,000
- **列数**: 22
- **用途**: RFM 模型、客户细分、关联规则挖掘
- **许可证**: MIT

### P03: 银行营销数据（UCI 公开数据集）

- **文件**: `bank_marketing.csv`
- **行数**: 45,211
- **列数**: 17
- **用途**: 分类模型、特征工程、不平衡数据处理
- **来源**: UCI Machine Learning Repository
- **许可证**: CC BY 4.0
- **引用**: Moro et al., 2014. A Data-Driven Approach to Predict the Success of Bank Telemarketing.

### P04: 电信客户流失数据（Kaggle 公开数据集）

- **文件**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **行数**: 7,043
- **列数**: 21
- **用途**: 特征工程、流失预测建模、模型解释
- **来源**: Kaggle / IBM Sample Data Sets
- **许可证**: 公开数据集

### P05-P09: Coming Soon

以下项目正在开发中：

- P05: 零售超市经营分析
- P06: 滴滴出行运营数据异常分析
- P07: 淘宝百万级用户行为分析
- P08: 航空公司客户价值分析
- P09: 信用贷款前审批项目

欢迎关注项目更新或参与贡献！
