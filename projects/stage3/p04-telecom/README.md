# P04: 电信客户流失预测

## 项目概述

基于 IBM Telco Customer Churn 数据集，构建客户流失预测模型。通过探索性数据分析、特征工程和机器学习建模，识别影响客户流失的关键因素，并提供业务策略建议。

## 数据集

- **来源**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **规模**: 7,043 行 × 21 列
- **流失率**: 约 26.5%（不平衡数据集）

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| customerID | string | 客户唯一标识 |
| gender | string | 性别 |
| SeniorCitizen | int | 是否老年人 (0/1) |
| Partner | string | 是否有伴侣 |
| Dependents | string | 是否有家属 |
| tenure | int | 在网时长（月） |
| PhoneService | string | 是否有电话服务 |
| MultipleLines | string | 是否有多线路 |
| InternetService | string | 互联网服务类型 |
| OnlineSecurity | string | 在线安全服务 |
| OnlineBackup | string | 在线备份 |
| DeviceProtection | string | 设备保护 |
| TechSupport | string | 技术支持 |
| StreamingTV | string | 流媒体TV |
| StreamingMovies | string | 流媒体电影 |
| Contract | string | 合约类型 |
| PaperlessBilling | string | 无纸化账单 |
| PaymentMethod | string | 支付方式 |
| MonthlyCharges | float | 月费用 |
| TotalCharges | float | 总费用 |
| Churn | string | 是否流失 (Yes/No) |

## 环境要求

```bash
pip install pandas scikit-learn matplotlib seaborn pyyaml tabulate
```

## 快速开始

### 1. 下载数据

```bash
# 方式1: Kaggle CLI
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/stage3/ --unzip

# 方式2: 浏览器下载
# 访问 https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# 将 CSV 文件放到项目根目录的 data/stage3/ 下
```

### 2. 运行分析

```bash
cd projects/stage3/p04-telecom
python src/analyze.py --config configs/default.yaml
```

### 3. 查看结果

分析完成后，结果保存在 `outputs/` 目录：
- `figures/` — EDA 图表、ROC 曲线、混淆矩阵、特征重要性
- `models/` — 训练好的模型文件
- `reports/` — 分析报告

## 项目结构

```
p04-telecom/
├── configs/
│   └── default.yaml          # 项目配置
├── src/
│   ├── analyze.py             # 主分析脚本
│   ├── data/
│   │   └── loader.py          # 数据加载
│   ├── models/
│   │   └── churn_predictor.py # 流失预测模型
│   └── utils/
│       ├── logger.py          # 日志工具
│       ├── metrics.py         # 评估指标
│       └── visualization.py   # 可视化工具
├── outputs/                   # 运行输出（gitignored）
└── README.md
```

## 学习要点

1. **探索性数据分析**: 理解数据分布，发现流失相关特征
2. **特征工程**: 二值编码、One-Hot 编码、缺失值处理
3. **不平衡数据处理**: class_weight="balanced" 策略
4. **模型对比**: 逻辑回归 vs 随机森林
5. **模型评估**: Accuracy、Precision、Recall、F1、AUC-ROC
6. **特征重要性**: 识别影响流失的关键因素
7. **业务洞察**: 将模型结果转化为可执行的业务策略
