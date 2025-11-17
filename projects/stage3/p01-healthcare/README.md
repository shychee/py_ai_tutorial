# Project P01: 朝阳医院销售数据分析

本项目通过分析医院销售数据,掌握数据清洗、探索性分析、可视化等核心技能。

---

## 📋 项目背景

**业务场景**:
朝阳医院是一家综合性医疗机构,希望通过分析药品销售数据优化库存管理、发现销售趋势、提升运营效率。

**数据来源**:
医院信息系统(HIS)导出的销售明细数据,包含50万条销售记录,时间跨度2022-2024年。

**业务目标**:
1. 清洗数据,处理缺失值和异常值
2. 分析销售趋势,识别畅销和滞销药品
3. 按类别、时间维度进行多维度分析
4. 生成可视化报告,辅助管理层决策

---

## 🎯 学习目标

完成本项目后,你将掌握:
- ✅ **数据清洗**: 处理缺失值、重复值、异常值的实用方法
- ✅ **探索性分析**: 使用Pandas进行分组聚合、透视表分析
- ✅ **时间序列**: 分析销售趋势,按月/季度/年度汇总
- ✅ **数据可视化**: 使用Matplotlib/Seaborn创建专业图表
- ✅ **指标计算**: 销售额、增长率、同比环比等业务指标

---

## 📊 数据说明

### 数据集基本信息

- **文件名**: `hospital_sales.csv`
- **数据路径**: `data/stage3/hospital_sales.csv`
- **文件大小**: 52MB
- **记录数**: 500,000 条
- **字段数**: 18 个
- **时间范围**: 2022-01-01 至 2024-12-31

### 字段说明

| 字段名 | 数据类型 | 说明 | 示例值 |
|--------|---------|------|--------|
| order_id | string | 订单编号 | ORD20220101001 |
| order_date | datetime | 订单日期 | 2022-01-15 |
| product_name | string | 药品名称 | 阿莫西林胶囊 |
| category | string | 药品分类 | 抗生素 |
| quantity | integer | 销售数量 | 50 |
| unit_price | float | 单价(元) | 12.50 |
| total_amount | float | 总金额(元) | 625.00 |
| customer_type | string | 客户类型 | 个人/机构 |
| department | string | 科室 | 内科 |
| doctor_name | string | 医生姓名 | 张医生 |
| manufacturer | string | 生产厂家 | XX制药 |
| batch_number | string | 批次号 | B202201 |
| expiry_date | datetime | 有效期 | 2025-01-01 |
| payment_method | string | 支付方式 | 医保/自费 |
| discount_rate | float | 折扣率 | 0.95 |
| sales_rep | string | 销售代表 | 李代表 |
| region | string | 地区 | 北京市朝阳区 |
| notes | string | 备注 | (可能为空) |

### 数据质量

- **缺失值**: 0.5% (主要在notes字段)
- **重复值**: 0.2% (少量重复订单)
- **异常值**: 1% (单价或数量异常)
- **数据类型**: 需要转换order_date和expiry_date为日期类型

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 依赖包: pandas, numpy, matplotlib, seaborn

### 安装依赖

```bash
# 在项目根目录
pip install -e ".[stage3]"

# 或在P01目录使用pyproject.toml
cd docs/stage3/projects/p01-healthcare
pip install -e .
```

### 运行分析

#### 方法1: Python脚本

```bash
# 运行完整分析
python src/analyze.py

# 指定自定义配置
python src/analyze.py --config configs/custom.yaml

# 仅生成图表
python src/analyze.py --visualize-only
```

#### 方法2: Jupyter Notebook

```bash
# 启动Jupyter
jupyter lab

# 打开notebooks/analysis.ipynb
# 按顺序运行所有单元格
```

---

## 📂 项目结构

```
p01-healthcare/
├── README.md              # 项目说明文档
├── pyproject.toml         # 依赖配置
├── src/
│   ├── __init__.py
│   ├── analyze.py         # 主分析脚本
│   ├── data_loader.py     # 数据加载模块
│   ├── cleaner.py         # 数据清洗模块
│   └── visualizer.py      # 可视化模块
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

### Step 1: 数据加载与初步检查

```python
import pandas as pd

# 加载数据
df = pd.read_csv('data/stage3/hospital_sales.csv')

# 基本信息
print(df.info())
print(df.describe())
print(df.head())
```

**检查内容**:
- 数据形状(行数、列数)
- 数据类型是否正确
- 缺失值统计
- 基础统计信息

### Step 2: 数据清洗

**处理缺失值**:
```python
# 检查缺失值
print(df.isnull().sum())

# 填充或删除缺失值
df['notes'].fillna('无备注', inplace=True)
df.dropna(subset=['product_name', 'total_amount'], inplace=True)
```

**处理重复值**:
```python
# 检查重复订单
duplicates = df.duplicated(subset=['order_id'])
print(f"重复订单数: {duplicates.sum()}")

# 删除重复
df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
```

**处理异常值**:
```python
# 异常单价检测(使用IQR方法)
Q1 = df['unit_price'].quantile(0.25)
Q3 = df['unit_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记或删除异常值
outliers = df[(df['unit_price'] < lower_bound) | (df['unit_price'] > upper_bound)]
print(f"异常值数量: {len(outliers)}")
```

**数据类型转换**:
```python
# 转换日期字段
df['order_date'] = pd.to_datetime(df['order_date'])
df['expiry_date'] = pd.to_datetime(df['expiry_date'])

# 提取日期特征
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['quarter'] = df['order_date'].dt.quarter
df['day_of_week'] = df['order_date'].dt.dayofweek
```

### Step 3: 探索性数据分析(EDA)

**总体销售情况**:
```python
# 总销售额
total_sales = df['total_amount'].sum()
print(f"总销售额: {total_sales:,.2f} 元")

# 平均单价
avg_price = df['unit_price'].mean()
print(f"平均单价: {avg_price:.2f} 元")

# 总订单数
total_orders = len(df)
print(f"总订单数: {total_orders:,}")
```

**按类别分析**:
```python
# 各类别销售额
category_sales = df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
print(category_sales)

# 各类别销售占比
category_ratio = category_sales / total_sales * 100
print(category_ratio)
```

**时间趋势分析**:
```python
# 按月汇总销售额
monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['total_amount'].sum()
print(monthly_sales)

# 同比增长率
yearly_sales = df.groupby('year')['total_amount'].sum()
growth_rate = yearly_sales.pct_change() * 100
print(f"年度增长率:\n{growth_rate}")
```

**畅销药品TOP10**:
```python
# 按销售额排名
top10_products = df.groupby('product_name')['total_amount'].sum().sort_values(ascending=False).head(10)
print(top10_products)

# 按销量排名
top10_volume = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(10)
print(top10_volume)
```

### Step 4: 数据可视化

**销售趋势图**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 月度销售趋势
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('月度销售趋势', fontsize=16)
plt.xlabel('月份')
plt.ylabel('销售额(元)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/monthly_trend.png', dpi=300)
plt.show()
```

**类别占比饼图**:
```python
plt.figure(figsize=(10, 8))
category_sales.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('各类别销售额占比', fontsize=16)
plt.ylabel('')
plt.tight_layout()
plt.savefig('outputs/figures/category_pie.png', dpi=300)
plt.show()
```

**TOP10药品柱状图**:
```python
plt.figure(figsize=(12, 6))
top10_products.plot(kind='barh')
plt.title('销售额TOP10药品', fontsize=16)
plt.xlabel('销售额(元)')
plt.ylabel('药品名称')
plt.tight_layout()
plt.savefig('outputs/figures/top10_products.png', dpi=300)
plt.show()
```

**热力图**:
```python
# 按月份和类别的销售热力图
pivot_table = df.pivot_table(
    values='total_amount',
    index='month',
    columns='category',
    aggfunc='sum'
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('月度-类别销售热力图', fontsize=16)
plt.tight_layout()
plt.savefig('outputs/figures/heatmap.png', dpi=300)
plt.show()
```

### Step 5: 生成报告

```python
# 生成汇总报告
report = f"""
# 朝阳医院销售数据分析报告

## 数据概览
- 分析周期: {df['order_date'].min()} 至 {df['order_date'].max()}
- 订单总数: {len(df):,}
- 总销售额: {total_sales:,.2f} 元
- 平均订单金额: {df['total_amount'].mean():.2f} 元

## 主要发现
1. 销售额最高的类别是: {category_sales.index[0]} ({category_sales.iloc[0]:,.2f}元)
2. 畅销药品TOP3: {', '.join(top10_products.index[:3])}
3. 年度同比增长率: {growth_rate.iloc[-1]:.2f}%

## 建议
1. 加强畅销药品库存管理
2. 关注滞销类别,考虑促销或调整采购
3. 根据季节性趋势优化库存策略
"""

with open('outputs/reports/summary_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("报告已生成: outputs/reports/summary_report.md")
```

---

## 📈 评估指标

本项目的完成质量通过以下指标评估:

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 数据清洗 | 100% | 无缺失值、重复值、异常值 |
| 分析维度 | ≥5个 | 时间、类别、药品、科室、地区 |
| 可视化图表 | ≥6个 | 趋势图、饼图、柱状图、热力图等 |
| 指标计算 | ≥8个 | 销售额、增长率、占比、TOP10等 |
| 代码质量 | 通过测试 | 函数化、注释清晰、可复现 |

---

## 💡 扩展思考

完成基础分析后,可以尝试以下进阶任务:

1. **时间序列预测**: 使用ARIMA或Prophet预测未来销售趋势
2. **客户细分**: 基于RFM模型分析客户价值
3. **关联规则**: 使用Apriori算法发现药品关联销售模式
4. **异常检测**: 识别异常订单和销售行为
5. **交互式仪表盘**: 使用Plotly Dash创建动态仪表盘

---

## 🐛 常见问题

### Q1: 数据文件找不到?
A: 确保已运行`python scripts/data/download-stage3.py`下载数据,或检查路径是否正确。

### Q2: 中文显示为方框?
A: 配置matplotlib中文字体:
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
```

### Q3: 内存不足?
A: 使用分块读取:
```python
chunks = pd.read_csv('data/stage3/hospital_sales.csv', chunksize=50000)
for chunk in chunks:
    process(chunk)
```

### Q4: 如何保存清洗后的数据?
A: 使用Parquet格式节省空间:
```python
df.to_parquet('outputs/processed_data/cleaned_data.parquet')
```

---

## 📚 参考资源

- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib教程](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn画廊](https://seaborn.pydata.org/examples/index.html)
- [探索性数据分析指南](https://github.com/mwaskom/seaborn-data)

---

## ✅ 检查清单

完成项目前,请确认:
- [ ] 成功加载数据并检查基本信息
- [ ] 完成数据清洗(缺失值、重复值、异常值)
- [ ] 计算至少8个业务指标
- [ ] 生成至少6个可视化图表
- [ ] 创建汇总分析报告
- [ ] 代码通过测试并能复现结果
- [ ] 理解每一步的业务含义

---

**项目完成时间**: 预计2-4小时  
**难度等级**: ⭐⭐ 入门  
**前置知识**: Pandas基础、Matplotlib基础

**下一个项目**: [P02 - 电商用户画像](../p02-ecommerce/)

---

**最后更新**: 2025-11-12
**维护者**: py_ai_tutorial团队
