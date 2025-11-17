# P03 银行营销预测 (Bank Marketing Classification)

## 项目概述

本项目基于葡萄牙银行的真实营销数据集，使用机器学习分类模型预测客户是否会订购定期存款产品。这是一个经典的二分类问题，适合学习逻辑回归和决策树等基础分类算法。

**学习目标**:
- 理解二分类问题的建模流程
- 掌握逻辑回归(Logistic Regression)和决策树(Decision Tree)算法
- 学习特征工程：数值特征标准化、类别特征编码
- 掌握模型评估指标：准确率、精确率、召回率、F1、AUC-ROC
- 理解类别不平衡问题及处理方法

**数据来源**: UCI Machine Learning Repository - Bank Marketing Dataset
**数据规模**: 45,211条记录, 17个特征, 1个目标变量
**业务场景**: 银行电话营销，预测客户是否会订购定期存款

---

## 项目结构

```
p03-bank-marketing/
├── README.md              # 项目说明文档
├── pyproject.toml         # 依赖配置
├── src/
│   ├── __init__.py
│   └── analyze.py         # 分类分析主脚本
├── notebooks/
│   └── analysis.ipynb     # 交互式教程
├── configs/
│   └── default.yaml       # 配置文件
└── outputs/               # 自动生成
    ├── analysis.log       # 运行日志
    ├── models/            # 模型文件
    │   ├── logistic_model.pkl
    │   └── tree_model.pkl
    ├── figures/           # 可视化图表
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── feature_importance.png
    │   └── correlation_heatmap.png
    └── reports/
        └── classification_report.md
```

---

## 数据说明

### 数据字段 (17个特征 + 1个目标)

#### 客户信息 (Demographic)
- `age`: 年龄 (数值型)
- `job`: 职业 (类别型: admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
- `marital`: 婚姻状况 (类别型: divorced, married, single)
- `education`: 教育程度 (类别型: primary, secondary, tertiary, unknown)

#### 账户信息 (Account)
- `default`: 是否有信用违约 (类别型: yes, no)
- `balance`: 账户余额 (数值型, 欧元)
- `housing`: 是否有住房贷款 (类别型: yes, no)
- `loan`: 是否有个人贷款 (类别型: yes, no)

#### 营销活动信息 (Campaign)
- `contact`: 联系方式 (类别型: cellular, telephone, unknown)
- `day`: 最后联系日期 (数值型, 1-31)
- `month`: 最后联系月份 (类别型: jan, feb, mar, ..., dec)
- `duration`: 最后通话时长 (数值型, 秒) ⚠️ 注意：这个特征在实际预测时不可用(只有通话后才知道时长)
- `campaign`: 本次营销活动联系次数 (数值型)
- `pdays`: 距离上次联系天数 (数值型, -1表示未联系过)
- `previous`: 上次营销活动联系次数 (数值型)
- `poutcome`: 上次营销结果 (类别型: failure, other, success, unknown)

#### 目标变量 (Target)
- `y`: 是否订购定期存款 (类别型: yes=1, no=0)

### 类别分布

- **正样本 (yes)**: ~11.7% (约5,289条)
- **负样本 (no)**: ~88.3% (约39,922条)
- **类别不平衡比**: 约 1:8 ⚠️ 需要特殊处理

---

## 快速开始

### 1. 环境准备

```bash
# 使用 uv 运行(推荐)
uv run --no-project --with pandas --with numpy --with scikit-learn --with matplotlib --with seaborn --with pyyaml \
  python src/analyze.py --config configs/default.yaml

# 或安装依赖后运行
uv pip install pandas numpy scikit-learn matplotlib seaborn pyyaml
python src/analyze.py --config configs/default.yaml
```

### 2. 运行分析

```bash
# 使用默认配置
python src/analyze.py --config configs/default.yaml

# 交互式教程
jupyter notebook notebooks/analysis.ipynb
```

### 3. 查看结果

分析完成后查看:
- `outputs/reports/classification_report.md` - 详细评估报告
- `outputs/figures/` - 可视化图表
- `outputs/models/` - 训练好的模型文件
- `outputs/analysis.log` - 运行日志

---

## 核心算法

### 1. 逻辑回归 (Logistic Regression)

**原理**: 线性模型 + Sigmoid激活函数，输出概率值

**数学公式**:
```
P(y=1|x) = 1 / (1 + e^(-z))
其中 z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**优点**:
- 模型简单，训练速度快
- 输出概率值，便于解释
- 对线性可分数据效果好
- 不容易过拟合

**缺点**:
- 假设线性关系，对非线性数据效果差
- 对特征缩放敏感
- 不能处理特征交互

**适用场景**:
- 特征与目标呈线性关系
- 需要快速训练和预测
- 需要解释模型权重

### 2. 决策树 (Decision Tree)

**原理**: 递归划分特征空间，构建树形结构

**划分标准**:
- **基尼指数(Gini)**: `Gini = 1 - Σ(pᵢ²)` (CART算法)
- **信息增益(Entropy)**: `Entropy = -Σ(pᵢ·log₂pᵢ)` (ID3/C4.5算法)

**优点**:
- 可以捕捉非线性关系
- 自动进行特征选择
- 不需要特征缩放
- 输出可视化，易于理解
- 可以处理类别特征

**缺点**:
- 容易过拟合(需要剪枝)
- 对噪声敏感
- 不稳定(数据小变化可能导致树结构大变化)

**适用场景**:
- 特征与目标呈非线性关系
- 需要模型可解释性
- 数据包含复杂交互关系

---

## 特征工程

### 1. 数值特征处理

**标准化 (Standardization)**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
# 转换为: mean=0, std=1
```

**为什么需要标准化?**
- 逻辑回归使用梯度下降优化，不同尺度的特征会导致收敛慢
- 某些特征(如balance, duration)数值范围大，会主导模型
- 标准化后所有特征在同一尺度，权重可比较

### 2. 类别特征处理

**独热编码 (One-Hot Encoding)**:
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# job: [admin, blue-collar, ...] -> [1,0,0,...], [0,1,0,...], ...
```

**标签编码 (Label Encoding)**:
```python
# 二值类别: yes/no -> 1/0
# 有序类别: primary/secondary/tertiary -> 1/2/3
```

### 3. 特征选择

**移除duration特征**: 虽然duration是最强特征，但实际预测时不可用(通话后才知道时长)，移除以避免数据泄漏。

**相关性分析**: 计算特征与目标的相关系数，保留相关性高的特征。

---

## 模型评估

### 1. 混淆矩阵 (Confusion Matrix)

```
                预测 No    预测 Yes
实际 No (TN)      8,000      100
实际 Yes (FN)       300      600
```

- **TN (True Negative)**: 正确预测为负
- **FP (False Positive)**: 错误预测为正 (Type I Error)
- **FN (False Negative)**: 错误预测为负 (Type II Error)
- **TP (True Positive)**: 正确预测为正

### 2. 评估指标

**准确率 (Accuracy)**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
适用于类别平衡数据
```

**精确率 (Precision)**:
```
Precision = TP / (TP + FP)
在所有预测为正的样本中，真正为正的比例
业务含义: 营销成功率
```

**召回率 (Recall/Sensitivity)**:
```
Recall = TP / (TP + FN)
在所有真正为正的样本中，被正确预测的比例
业务含义: 潜在客户覆盖率
```

**F1分数 (F1-Score)**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
精确率和召回率的调和平均数
适用于类别不平衡数据
```

**AUC-ROC**:
- ROC曲线: Recall(TPR) vs FPR曲线
- AUC: ROC曲线下面积 (0.5-1.0, 越大越好)
- 适用于类别不平衡数据，衡量模型排序能力

### 3. 类别不平衡处理

**方法1: 调整类别权重**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
# 自动计算权重: n_samples / (n_classes * np.bincount(y))
```

**方法2: 重采样**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**方法3: 调整决策阈值**
```python
# 默认阈值0.5, 可调整为0.3提高召回率
y_pred = (model.predict_proba(X_test)[:, 1] >= 0.3).astype(int)
```

---

## 业务应用

### 1. 营销策略优化

**场景**: 银行有10,000个潜在客户，如何分配有限的营销资源?

**解决方案**:
1. 使用模型预测每个客户的订购概率
2. 按概率从高到低排序
3. 只联系Top 20%高概率客户
4. 预期转化率提升3-5倍

### 2. 客户细分

基于预测概率将客户分为4组:
- **高潜力客户** (p > 0.7): 重点跟进，提供优惠
- **中等潜力** (0.3 < p < 0.7): 标准营销话术
- **低潜力客户** (0.1 < p < 0.3): 低频次接触
- **无潜力客户** (p < 0.1): 不联系，避免骚扰

### 3. A/B测试

- **对照组**: 随机选择客户进行营销
- **实验组**: 基于模型预测选择客户
- **评估指标**: 转化率、ROI、客户满意度

---

## 扩展思考

### 1. 特征工程优化
- 创建交互特征: age×job, balance×housing
- 时间特征: 季节性(month)、周期性
- 聚合特征: 历史营销总次数、平均通话时长

### 2. 模型优化
- 超参数调优: Grid Search, Random Search
- 集成学习: Random Forest, XGBoost
- 神经网络: 多层感知机(MLP)

### 3. 业务优化
- 多目标优化: 平衡转化率与营销成本
- 动态阈值: 根据营销预算动态调整决策阈值
- 实时预测: 部署在线服务API

### 4. 进阶分析
- SHAP值解释: 理解每个特征对预测的贡献
- Uplift模型: 预测营销活动的增量效果
- 因果推断: 评估营销活动的真实因果效应

---

## 常见问题

### Q1: 为什么不能使用duration特征?
**A**: duration是最后通话时长，只有通话结束后才知道。在实际预测时(通话前)，这个特征不可用。使用它会导致**数据泄漏**(Data Leakage)，模型在训练集上表现很好，但实际应用时失效。

### Q2: 如何处理类别不平衡?
**A**:
1. 使用F1、AUC-ROC等适合不平衡数据的指标
2. 调整模型的`class_weight='balanced'`
3. 使用SMOTE等重采样技术
4. 调整决策阈值(如从0.5降到0.3)

### Q3: 逻辑回归 vs 决策树如何选择?
**A**:
- 数据呈线性关系 -> 逻辑回归
- 数据呈非线性关系 -> 决策树
- 需要概率输出 -> 逻辑回归
- 需要可视化解释 -> 决策树
- 特征很多且有交互 -> 决策树或集成方法

### Q4: 如何评估模型是否过拟合?
**A**: 比较训练集和测试集的性能:
- 训练集准确率95%, 测试集准确率70% -> 过拟合
- 解决方法: 正则化(L1/L2)、剪枝、减少特征、增加数据

### Q5: 模型部署后如何监控?
**A**:
- 监控预测分布: 检测数据漂移
- 监控业务指标: 转化率、ROI
- 定期重新训练: 每月或每季度
- A/B测试: 对比新旧模型效果

---

## 参考资料

### 数据集
- UCI Repository: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- 论文: [Moro et al., 2014] A Data-Driven Approach to Predict the Success of Bank Telemarketing

### 算法原理
- 《统计学习方法》李航 - 第6章 逻辑回归, 第5章 决策树
- Scikit-learn文档: https://scikit-learn.org/stable/supervised_learning.html

### 类别不平衡
- imbalanced-learn库: https://imbalanced-learn.org/
- 论文: SMOTE - Synthetic Minority Over-sampling Technique

### 模型评估
- 混淆矩阵可视化: https://scikit-learn.org/stable/modules/model_evaluation.html
- ROC-AUC解读: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

---

## 更新日志

- **2025-11-13**: 初始版本，包含逻辑回归和决策树分类模型
- 数据集: 45,211条记录, 17个特征
- 实现: 完整的数据预处理、特征工程、模型训练、评估、可视化流程
