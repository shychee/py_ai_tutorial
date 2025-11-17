# Project P02: 服装零售销售数据分析

本项目通过分析服装零售销售数据，学习RFM模型、客户细分、关联规则挖掘等核心技能。

---

## 📋 项目背景

**业务场景**:
优衣库等服装零售企业希望通过分析销售数据进行精准营销，提升客户留存率和复购率。

**数据来源**:
服装零售系统导出的销售明细数据，包含2000条销售记录（测试数据），时间跨度2023-2024年。

**业务目标**:
1. 构建RFM模型，进行客户价值分析
2. 进行客户细分，识别高价值客户
3. 分析销售模式和产品关联
4. 制定差异化营销策略

---

## 🎯 学习目标

完成本项目后，你将掌握：
- ✅ **RFM模型**: 最近购买时间、购买频率、购买金额三维度客户分析
- ✅ **客户细分**: 基于RFM评分进行客户分群
- ✅ **销售分析**: 产品、类别、渠道多维度分析
- ✅ **购买行为**: 复购率、客单价、偏好分析
- ✅ **商业洞察**: 从数据到业务建议的完整流程

---

## 📊 数据说明

### 数据集基本信息

- **文件名**: `clothing_retail.csv`
- **数据路径**: `data/stage3/clothing_retail.csv`
- **文件大小**: 0.30 MB
- **记录数**: 2,000 条
- **字段数**: 21 个
- **时间范围**: 2023-01-01 至 2024-12-31

### 字段说明

| 字段名 | 数据类型 | 说明 | 示例值 |
|--------|---------|------|--------|
| order_id | string | 订单编号 | EC20230101001 |
| customer_id | string | 客户编号 | C000123 |
| order_date | datetime | 订单日期 | 2023-05-20 |
| product_id | string | 产品编号 | P001234 |
| product_name | string | 产品名称 | 优衣库 上衣 |
| category | string | 产品类别 | 上衣 |
| brand | string | 品牌 | 优衣库 |
| size | string | 尺码 | M |
| color | string | 颜色 | 黑色 |
| price | float | 单价(元) | 199.00 |
| quantity | integer | 数量 | 2 |
| total_amount | float | 总金额(元) | 398.00 |
| discount | float | 折扣率 | 0.15 |
| payment_method | string | 支付方式 | 支付宝 |
| shipping_method | string | 配送方式 | 快递 |
| region | string | 地区 | 华东 |
| customer_age | integer | 客户年龄 | 28 |
| customer_gender | string | 客户性别 | 女 |
| is_member | boolean | 是否会员 | True |
| channel | string | 渠道 | 线上 |
| status | string | 订单状态 | 已完成 |

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 依赖包: pandas, numpy, matplotlib, seaborn

### 安装依赖

```bash
# 在项目根目录
pip install -e ".[stage3]"

# 或在P02目录
cd docs/stage3/projects/p02-ecommerce
pip install -e .
```

### 运行分析

#### 方法1: Python脚本

```bash
# 完整分析（从项目根目录运行）
uv run --no-project --with pandas --with numpy --with matplotlib --with seaborn --with pyyaml \
  python specs/002-ai-tutorial-stages/docs/stage3/projects/p02-ecommerce/src/analyze.py

# 指定配置
python src/analyze.py --config configs/custom.yaml
```

#### 方法2: Jupyter Notebook

```bash
# 启动Jupyter
jupyter lab

# 打开notebooks/analysis.ipynb
```

---

## 📂 项目结构

```
p02-ecommerce/
├── README.md              # 项目说明文档
├── pyproject.toml         # 依赖配置
├── src/
│   ├── __init__.py
│   └── analyze.py         # 主分析脚本（RFM + 客户细分）
├── notebooks/
│   └── analysis.ipynb     # 交互式分析笔记本
├── configs/
│   └── default.yaml       # 默认配置
├── tests/
│   └── test_analyze.py    # 单元测试
└── outputs/               # 输出目录(自动生成)
    ├── figures/           # 图表
    ├── reports/           # 分析报告
    └── processed_data/    # 清洗后数据
```

---

## 🔍 分析流程

### Step 1: 数据加载与清洗

```python
import pandas as pd

# 加载数据
df = pd.read_csv('data/stage3/clothing_retail.csv')

# 转换日期类型
df['order_date'] = pd.to_datetime(df['order_date'])

# 基本统计
print(df.info())
print(df.describe())
```

### Step 2: RFM模型构建

**RFM模型**是客户价值分析的经典方法：
- **R (Recency)**: 最近一次购买距今天数，越小越好
- **F (Frequency)**: 购买频率（订单数），越大越好
- **M (Monetary)**: 购买金额（总消费），越大越好

```python
# 计算分析日期（最后一个订单日期 + 1天）
analysis_date = df['order_date'].max() + pd.Timedelta(days=1)

# 按客户聚合计算RFM
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (analysis_date - x.max()).days,  # R
    'order_id': 'count',                                      # F
    'total_amount': 'sum'                                     # M
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'total_amount': 'monetary'
})

print(rfm.head())
```

### Step 3: RFM评分

将RFM三个维度分别打分（1-5分），5分最好：

```python
# 创建评分函数（R越小越好，F和M越大越好）
def rfm_score(df, col, ascending=False):
    """基于四分位数打分"""
    return pd.qcut(df[col], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# 计算RFM评分
rfm['R_score'] = rfm_score(rfm, 'recency', ascending=True)   # R越小越好
rfm['F_score'] = rfm_score(rfm, 'frequency', ascending=False) # F越大越好
rfm['M_score'] = rfm_score(rfm, 'monetary', ascending=False)  # M越大越好

# 计算RFM总分
rfm['RFM_score'] = rfm['R_score'].astype(int) * 100 + \
                   rfm['F_score'].astype(int) * 10 + \
                   rfm['M_score'].astype(int)

print(rfm.sort_values('RFM_score', ascending=False).head(10))
```

### Step 4: 客户细分

基于RFM评分进行客户分群：

```python
def segment_customers(row):
    """客户细分逻辑"""
    r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])

    # 重要价值客户：RFM都高
    if r >= 4 and f >= 4 and m >= 4:
        return '重要价值客户'
    # 重要保持客户：R高，F或M一般
    elif r >= 4 and (f >= 2 or m >= 2):
        return '重要保持客户'
    # 重要挽留客户：R低，F和M高
    elif r <= 2 and f >= 4 and m >= 4:
        return '重要挽留客户'
    # 一般发展客户：F或M高，R一般
    elif (f >= 3 or m >= 3) and r >= 2:
        return '一般发展客户'
    # 一般维持客户：RFM都中等
    elif r >= 2 and f >= 2 and m >= 2:
        return '一般维持客户'
    # 一般挽留客户：R低，F或M一般
    elif r <= 2 and (f >= 2 or m >= 2):
        return '一般挽留客户'
    # 潜在客户：RFM都低
    else:
        return '潜在客户'

rfm['customer_segment'] = rfm.apply(segment_customers, axis=1)

# 客户分布
print(rfm['customer_segment'].value_counts())
```

### Step 5: 数据可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 客户细分分布饼图
plt.figure(figsize=(10, 8))
rfm['customer_segment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('客户细分分布', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig('outputs/figures/customer_segments.png', dpi=300)
plt.show()

# 2. RFM三维散点图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(rfm['recency'], rfm['frequency'], rfm['monetary'],
                     c=rfm['RFM_score'], cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('Recency (天)')
ax.set_ylabel('Frequency (次)')
ax.set_zlabel('Monetary (元)')
plt.title('RFM三维分布', fontsize=16, fontweight='bold')
plt.colorbar(scatter, label='RFM总分')
plt.tight_layout()
plt.savefig('outputs/figures/rfm_3d.png', dpi=300)
plt.show()

# 3. 各细分客户的RFM均值对比
segment_avg = rfm.groupby('customer_segment')[['recency', 'frequency', 'monetary']].mean()
segment_avg.plot(kind='bar', figsize=(12, 6))
plt.title('各客户群体RFM均值对比', fontsize=16, fontweight='bold')
plt.xlabel('客户群体')
plt.ylabel('均值')
plt.legend(['最近购买(天)', '购买频率(次)', '购买金额(元)'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/segment_rfm_comparison.png', dpi=300)
plt.show()
```

### Step 6: 业务洞察与建议

```python
# 生成分析报告
report = f"""
# 服装零售客户价值分析报告

## 客户细分结果

{rfm['customer_segment'].value_counts().to_markdown()}

## 各群体特征

### 重要价值客户
- 占比: {(rfm['customer_segment'] == '重要价值客户').sum() / len(rfm) * 100:.1f}%
- 平均消费: {rfm[rfm['customer_segment'] == '重要价值客户']['monetary'].mean():.2f} 元
- **策略**: VIP服务，专属优惠，优先推荐新品

### 重要挽留客户
- 占比: {(rfm['customer_segment'] == '重要挽留客户').sum() / len(rfm) * 100:.1f}%
- 平均沉默期: {rfm[rfm['customer_segment'] == '重要挽留客户']['recency'].mean():.0f} 天
- **策略**: 发送关怀短信，限时优惠券，唤回活动

## 营销建议

1. **精准营销**: 针对不同客户群体制定差异化策略
2. **会员运营**: 强化会员体系，提升忠诚度
3. **流失预警**: 监控R值上升的高价值客户，及时干预
"""

print(report)
```

---

## 📈 评估指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 客户细分覆盖率 | 100% | 所有客户均完成分群 |
| RFM模型准确性 | 合理分布 | 各分数段客户数量合理 |
| 可视化图表 | ≥5个 | RFM分布、客户细分、对比分析 |
| 业务建议 | ≥5条 | 针对性强，可操作 |

---

## 💡 扩展思考

1. **动态RFM**: 按季度更新RFM模型，跟踪客户价值变化
2. **生命周期**: 结合客户注册时长分析生命周期价值（CLV）
3. **关联规则**: 使用Apriori算法挖掘商品关联
4. **推荐系统**: 基于协同过滤进行个性化推荐
5. **流失预测**: 训练分类模型预测客户流失概率

---

## 🐛 常见问题

### Q1: RFM评分如何选择分位数？
A: 通常使用四分位数(quintile)分为5档，也可根据业务需求调整为3档或10档。

### Q2: 客户细分标准如何确定？
A: 可结合业务经验调整，或使用聚类算法（K-means）自动分群。

### Q3: 如何处理只购买一次的客户？
A: 设为"潜在客户"或"新客户"，重点观察是否复购。

---

## 📚 参考资源

- [RFM模型详解](https://en.wikipedia.org/wiki/RFM_(market_research))
- [客户细分策略](https://www.optimove.com/resources/learning-center/rfm-segmentation)
- [Python客户分析案例](https://github.com/topics/rfm-analysis)

---

## ✅ 检查清单

完成项目前，请确认：
- [ ] 成功加载数据并完成RFM计算
- [ ] 客户细分覆盖所有客户
- [ ] 生成至少5个可视化图表
- [ ] 编写针对性营销建议
- [ ] 代码可复现且注释清晰
- [ ] 理解RFM模型的业务价值

---

**项目完成时间**: 预计2-4小时
**难度等级**: ⭐⭐ 入门
**前置知识**: Pandas基础、Matplotlib基础

**下一个项目**: [P03 - 银行营销分类](../p03-finance/)

---

**最后更新**: 2025-11-13
**维护者**: py_ai_tutorial团队
