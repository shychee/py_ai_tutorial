#!/usr/bin/env python3
"""
P01 Healthcare Data Analysis - Main Analysis Script
医院销售数据分析主脚本

This script performs end-to-end analysis of hospital pharmaceutical sales data:
1. Data loading and validation
2. Data cleaning (missing values, duplicates, outliers)
3. Exploratory data analysis (EDA)
4. Visualization generation
5. Report creation

Usage:
    python src/analyze.py
    python src/analyze.py --config configs/custom.yaml
    python src/analyze.py --visualize-only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

warnings.filterwarnings('ignore')


class HospitalSalesAnalyzer:
    """医院销售数据分析器"""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        初始化分析器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_visualization()
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_logging(self):
        """配置日志"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'outputs/analysis.log')

        # 确保输出目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_visualization(self):
        """配置可视化样式"""
        viz_config = self.config.get('visualization', {})
        figure_config = viz_config.get('figure', {})
        font_config = viz_config.get('fonts', {})

        # 设置matplotlib样式
        style = figure_config.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            self.logger.warning(f"样式 {style} 不可用，使用默认样式")

        # 配置中文字体
        font_family = font_config.get('family', ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'])
        plt.rcParams['font.sans-serif'] = font_family
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = font_config.get('size', 12)

        # 配置图表默认大小
        self.default_figsize = figure_config.get('figsize_default', [12, 6])
        self.dpi = figure_config.get('dpi', 300)
        self.fig_format = figure_config.get('format', 'png')

        # 配置颜色
        color_config = viz_config.get('colors', {})
        self.palette = color_config.get('palette', 'Set2')
        self.cmap = color_config.get('cmap', 'YlOrRd')

    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        self.logger.info("开始加载数据...")

        data_config = self.config.get('data', {})
        input_file = data_config.get('input_file', 'data/stage3/hospital_sales.csv')

        # 获取加载参数
        loading_config = self.config.get('loading', {})
        encoding = loading_config.get('encoding', 'utf-8')
        parse_dates = loading_config.get('parse_dates', [])
        dtype = loading_config.get('dtype', {})

        # 加载数据
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {input_file}")

        self.df = pd.read_csv(
            input_path,
            encoding=encoding,
            parse_dates=parse_dates,
            dtype=dtype
        )

        self.logger.info(f"数据加载完成: {len(self.df)} 行, {len(self.df.columns)} 列")
        self.logger.info(f"数据形状: {self.df.shape}")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """数据清洗"""
        self.logger.info("开始数据清洗...")

        if self.df is None:
            raise ValueError("请先加载数据")

        self.df_clean = self.df.copy()
        cleaning_config = self.config.get('cleaning', {})

        # 1. 处理缺失值
        self._handle_missing_values(cleaning_config.get('missing_values', {}))

        # 2. 处理重复值
        self._handle_duplicates(cleaning_config.get('duplicates', {}))

        # 3. 处理异常值
        self._handle_outliers(cleaning_config.get('outliers', {}))

        # 4. 提取日期特征
        self._extract_date_features()

        self.logger.info(f"数据清洗完成: 剩余 {len(self.df_clean)} 行")

        return self.df_clean

    def _handle_missing_values(self, config: Dict[str, Any]):
        """处理缺失值"""
        self.logger.info("处理缺失值...")

        # 统计缺失值
        missing_counts = self.df_clean.isnull().sum()
        missing_pct = (missing_counts / len(self.df_clean) * 100).round(2)

        for col in missing_counts[missing_counts > 0].index:
            self.logger.info(f"  {col}: {missing_counts[col]} ({missing_pct[col]}%)")

        # 填充特定列
        for col, fill_value in config.items():
            if col in ['drop_columns', 'drop_rows_if_missing']:
                continue
            if col in self.df_clean.columns:
                before = self.df_clean[col].isnull().sum()
                self.df_clean[col].fillna(fill_value, inplace=True)
                self.logger.info(f"  填充 {col}: {before} 个缺失值 → '{fill_value}'")

        # 删除关键列缺失的行
        drop_rows_if_missing = config.get('drop_rows_if_missing', [])
        if drop_rows_if_missing:
            before_len = len(self.df_clean)
            self.df_clean.dropna(subset=drop_rows_if_missing, inplace=True)
            dropped = before_len - len(self.df_clean)
            if dropped > 0:
                self.logger.info(f"  删除关键字段缺失行: {dropped} 行")

    def _handle_duplicates(self, config: Dict[str, Any]):
        """处理重复值"""
        self.logger.info("处理重复值...")

        subset = config.get('subset', None)
        keep = config.get('keep', 'first')

        duplicates = self.df_clean.duplicated(subset=subset)
        dup_count = duplicates.sum()

        if dup_count > 0:
            self.logger.info(f"  发现 {dup_count} 条重复记录")
            self.df_clean.drop_duplicates(subset=subset, keep=keep, inplace=True)
            self.logger.info(f"  删除重复记录: 保留 {keep}")
        else:
            self.logger.info("  未发现重复记录")

    def _handle_outliers(self, config: Dict[str, Any]):
        """处理异常值（IQR方法）"""
        if not config.get('enabled', False):
            self.logger.info("异常值检测已禁用")
            return

        self.logger.info("检测异常值（IQR方法）...")

        columns = config.get('columns', [])
        iqr_multiplier = config.get('iqr_multiplier', 1.5)
        action = config.get('action', 'flag')

        outlier_info = {}

        for col in columns:
            if col not in self.df_clean.columns:
                continue

            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            outliers = (self.df_clean[col] < lower_bound) | (self.df_clean[col] > upper_bound)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df_clean) * 100).round(2),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

                self.logger.info(f"  {col}: {outlier_count} 个异常值 ({outlier_info[col]['percentage']}%)")
                self.logger.info(f"    范围: [{lower_bound:.2f}, {upper_bound:.2f}]")

                if action == 'remove':
                    self.df_clean = self.df_clean[~outliers]
                    self.logger.info(f"    已删除异常值")
                elif action == 'cap':
                    self.df_clean.loc[self.df_clean[col] < lower_bound, col] = lower_bound
                    self.df_clean.loc[self.df_clean[col] > upper_bound, col] = upper_bound
                    self.logger.info(f"    已截断异常值")
                elif action == 'flag':
                    self.df_clean[f'{col}_outlier'] = outliers
                    self.logger.info(f"    已标记异常值（新增列 {col}_outlier）")

        self.results['outliers'] = outlier_info

    def _extract_date_features(self):
        """提取日期特征"""
        self.logger.info("提取日期特征...")

        if 'order_date' in self.df_clean.columns:
            self.df_clean['year'] = self.df_clean['order_date'].dt.year
            self.df_clean['month'] = self.df_clean['order_date'].dt.month
            self.df_clean['quarter'] = self.df_clean['order_date'].dt.quarter
            self.df_clean['day_of_week'] = self.df_clean['order_date'].dt.dayofweek
            self.df_clean['week_of_year'] = self.df_clean['order_date'].dt.isocalendar().week
            self.logger.info("  已提取: year, month, quarter, day_of_week, week_of_year")

    def analyze(self) -> Dict[str, Any]:
        """执行探索性数据分析"""
        self.logger.info("开始探索性数据分析...")

        if self.df_clean is None:
            raise ValueError("请先清洗数据")

        analysis_config = self.config.get('analysis', {})

        # 1. 总体指标
        self._calculate_overall_metrics()

        # 2. 按维度分析
        self._analyze_by_dimensions(analysis_config.get('dimensions', []))

        # 3. 时间序列分析
        self._analyze_time_series(analysis_config.get('time_aggregations', []))

        # 4. TOP N分析
        self._analyze_top_n(analysis_config.get('top_n', 10))

        self.logger.info("探索性数据分析完成")

        return self.results

    def _calculate_overall_metrics(self):
        """计算总体指标"""
        self.logger.info("计算总体指标...")

        metrics = {
            'total_sales': self.df_clean['total_amount'].sum(),
            'total_orders': len(self.df_clean),
            'average_order_value': self.df_clean['total_amount'].mean(),
            'average_unit_price': self.df_clean['unit_price'].mean(),
            'total_quantity': self.df_clean['quantity'].sum(),
            'date_range': {
                'start': str(self.df_clean['order_date'].min().date()),
                'end': str(self.df_clean['order_date'].max().date())
            }
        }

        self.results['overall_metrics'] = metrics

        self.logger.info(f"  总销售额: {metrics['total_sales']:,.2f} 元")
        self.logger.info(f"  总订单数: {metrics['total_orders']:,}")
        self.logger.info(f"  平均订单金额: {metrics['average_order_value']:.2f} 元")

    def _analyze_by_dimensions(self, dimensions: list):
        """按维度分析"""
        self.logger.info("按维度分析...")

        dimension_results = {}

        for dim in dimensions:
            if dim not in self.df_clean.columns:
                continue

            # 按维度汇总销售额
            dim_sales = self.df_clean.groupby(dim)['total_amount'].agg(['sum', 'count', 'mean'])
            dim_sales = dim_sales.sort_values('sum', ascending=False)
            dim_sales.columns = ['total_sales', 'order_count', 'avg_order_value']

            dimension_results[dim] = dim_sales

            self.logger.info(f"  {dim}: {len(dim_sales)} 个类别")

        self.results['dimensions'] = dimension_results

    def _analyze_time_series(self, aggregations: list):
        """时间序列分析"""
        self.logger.info("时间序列分析...")

        time_series_results = {}

        for agg in aggregations:
            if agg == 'daily':
                ts = self.df_clean.groupby('order_date')['total_amount'].sum()
            elif agg == 'monthly':
                ts = self.df_clean.groupby(self.df_clean['order_date'].dt.to_period('M'))['total_amount'].sum()
            elif agg == 'quarterly':
                ts = self.df_clean.groupby(self.df_clean['order_date'].dt.to_period('Q'))['total_amount'].sum()
            elif agg == 'yearly':
                ts = self.df_clean.groupby('year')['total_amount'].sum()
            else:
                continue

            time_series_results[agg] = ts
            self.logger.info(f"  {agg}: {len(ts)} 个时间点")

        self.results['time_series'] = time_series_results

    def _analyze_top_n(self, n: int):
        """TOP N分析"""
        self.logger.info(f"TOP {n} 分析...")

        # TOP N 产品（按销售额）
        top_products = self.df_clean.groupby('product_name')['total_amount'].sum().sort_values(ascending=False).head(n)
        self.results['top_products'] = top_products
        self.logger.info(f"  TOP {n} 产品（销售额）")

        # TOP N 产品（按销量）
        top_products_volume = self.df_clean.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(n)
        self.results['top_products_volume'] = top_products_volume
        self.logger.info(f"  TOP {n} 产品（销量）")

    def visualize(self):
        """生成可视化图表"""
        self.logger.info("生成可视化图表...")

        if self.df_clean is None:
            raise ValueError("请先清洗数据")

        # 确保输出目录存在
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs'))
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        charts_config = self.config.get('visualization', {}).get('charts', [])

        for chart in charts_config:
            if not chart.get('enabled', True):
                continue

            chart_name = chart.get('name')
            chart_type = chart.get('type')
            chart_title = chart.get('title')

            self.logger.info(f"  生成图表: {chart_name} ({chart_type})")

            try:
                if chart_name == 'monthly_trend':
                    self._plot_monthly_trend(figures_dir, chart_title)
                elif chart_name == 'category_pie':
                    self._plot_category_pie(figures_dir, chart_title)
                elif chart_name == 'top10_products':
                    self._plot_top10_products(figures_dir, chart_title)
                elif chart_name == 'heatmap':
                    self._plot_heatmap(figures_dir, chart_title)
                elif chart_name == 'customer_type_bar':
                    self._plot_customer_type_bar(figures_dir, chart_title)
                elif chart_name == 'quarterly_boxplot':
                    self._plot_quarterly_boxplot(figures_dir, chart_title)
            except Exception as e:
                self.logger.error(f"    生成图表失败: {e}")

        self.logger.info(f"图表已保存到: {figures_dir}")

    def _plot_monthly_trend(self, output_dir: Path, title: str):
        """月度销售趋势图"""
        monthly_sales = self.results['time_series']['monthly']

        plt.figure(figsize=self.default_figsize)
        monthly_sales.plot(kind='line', marker='o', linewidth=2, markersize=6)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('月份', fontsize=14)
        plt.ylabel('销售额（元）', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'monthly_trend.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_category_pie(self, output_dir: Path, title: str):
        """类别销售额占比饼图"""
        category_sales = self.results['dimensions']['category']['total_sales']

        plt.figure(figsize=[10, 8])
        plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=90)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'category_pie.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_top10_products(self, output_dir: Path, title: str):
        """TOP10药品柱状图"""
        top10 = self.results['top_products']

        plt.figure(figsize=self.default_figsize)
        top10.plot(kind='barh', color=sns.color_palette(self.palette, n_colors=len(top10)))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('销售额（元）', fontsize=14)
        plt.ylabel('药品名称', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'top10_products.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_heatmap(self, output_dir: Path, title: str):
        """月度-类别销售热力图"""
        pivot_table = self.df_clean.pivot_table(
            values='total_amount',
            index='month',
            columns='category',
            aggfunc='sum'
        )

        plt.figure(figsize=[14, 8])
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap=self.cmap, linewidths=0.5)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('类别', fontsize=14)
        plt.ylabel('月份', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_customer_type_bar(self, output_dir: Path, title: str):
        """客户类型销售对比"""
        customer_sales = self.results['dimensions']['customer_type']['total_sales']

        plt.figure(figsize=[10, 6])
        customer_sales.plot(kind='bar', color=sns.color_palette(self.palette, n_colors=len(customer_sales)))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('客户类型', fontsize=14)
        plt.ylabel('销售额（元）', fontsize=14)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'customer_type_bar.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_quarterly_boxplot(self, output_dir: Path, title: str):
        """季度销售额分布箱线图"""
        plt.figure(figsize=self.default_figsize)
        self.df_clean.boxplot(column='total_amount', by='quarter', grid=False)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.suptitle('')  # 移除默认标题
        plt.xlabel('季度', fontsize=14)
        plt.ylabel('销售额（元）', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'quarterly_boxplot.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成分析报告"""
        self.logger.info("生成分析报告...")

        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs'))
        reports_dir = output_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_config = self.config.get('report', {})
        report_format = report_config.get('format', 'markdown')

        if report_format == 'markdown':
            self._generate_markdown_report(reports_dir)

        self.logger.info(f"报告已保存到: {reports_dir}")

    def _generate_markdown_report(self, output_dir: Path):
        """生成Markdown格式报告"""
        metrics = self.results.get('overall_metrics', {})
        dimensions = self.results.get('dimensions', {})
        top_products = self.results.get('top_products', pd.Series())

        # 获取TOP类别
        category_sales = dimensions.get('category', pd.DataFrame())
        top_category = category_sales.index[0] if len(category_sales) > 0 else 'N/A'
        top_category_sales = category_sales.iloc[0]['total_sales'] if len(category_sales) > 0 else 0

        report = f"""# 朝阳医院销售数据分析报告

## 📊 数据概览

- **分析周期**: {metrics.get('date_range', {}).get('start', 'N/A')} 至 {metrics.get('date_range', {}).get('end', 'N/A')}
- **订单总数**: {metrics.get('total_orders', 0):,}
- **总销售额**: {metrics.get('total_sales', 0):,.2f} 元
- **平均订单金额**: {metrics.get('average_order_value', 0):.2f} 元
- **平均单价**: {metrics.get('average_unit_price', 0):.2f} 元
- **总销售数量**: {metrics.get('total_quantity', 0):,}

---

## 🔍 主要发现

### 1. 销售额最高的类别
**{top_category}**: {top_category_sales:,.2f} 元

### 2. 畅销药品 TOP 3
"""

        for i, (product, sales) in enumerate(top_products.head(3).items(), 1):
            report += f"{i}. **{product}**: {sales:,.2f} 元\n"

        report += f"""
### 3. 客户类型分布
"""

        if 'customer_type' in dimensions:
            for customer_type, row in dimensions['customer_type'].iterrows():
                report += f"- **{customer_type}**: {row['total_sales']:,.2f} 元 ({row['order_count']:,} 订单)\n"

        report += """
---

## 📈 可视化图表

本次分析生成了以下图表（保存在 `outputs/figures/` 目录）：

1. **monthly_trend.png** - 月度销售趋势
2. **category_pie.png** - 各类别销售额占比
3. **top10_products.png** - 销售额TOP10药品
4. **heatmap.png** - 月度-类别销售热力图
5. **customer_type_bar.png** - 客户类型销售对比
6. **quarterly_boxplot.png** - 季度销售额分布

---

## 💡 建议

### 库存管理
1. 加强畅销药品（TOP10）的库存管理，避免缺货
2. 关注销售额占比低的类别，考虑促销或调整采购策略

### 销售策略
1. 根据月度趋势优化营销活动时间
2. 针对不同客户类型定制差异化服务方案

### 运营优化
1. 分析季节性趋势，提前备货
2. 识别异常订单，优化风控机制

---

## 📝 数据质量说明

"""

        if 'outliers' in self.results:
            report += "### 异常值检测结果\n\n"
            for col, info in self.results['outliers'].items():
                report += f"- **{col}**: 检测到 {info['count']} 个异常值 ({info['percentage']}%)\n"
                report += f"  - 正常范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n"

        report += f"""
---

**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析工具**: py_ai_tutorial P01 Healthcare Analysis
**版本**: 1.0.0
"""

        # 保存报告
        report_path = output_dir / 'summary_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"  Markdown报告已生成: {report_path}")

    def save_processed_data(self):
        """保存清洗后的数据"""
        self.logger.info("保存清洗后的数据...")

        if self.df_clean is None:
            raise ValueError("请先清洗数据")

        output_dir = Path(self.config.get('data', {}).get('processed_data_dir', 'outputs/processed_data'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存为CSV
        csv_path = output_dir / 'cleaned_data.csv'
        self.df_clean.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"  CSV已保存: {csv_path}")

        # 保存为Parquet（更高效）
        try:
            parquet_path = output_dir / 'cleaned_data.parquet'
            self.df_clean.to_parquet(parquet_path, index=False)
            self.logger.info(f"  Parquet已保存: {parquet_path}")
        except Exception as e:
            self.logger.warning(f"  Parquet保存失败: {e}")


def run_analysis(config_path: str = "configs/default.yaml",
                 visualize_only: bool = False) -> HospitalSalesAnalyzer:
    """
    运行完整分析流程

    Args:
        config_path: 配置文件路径
        visualize_only: 仅生成可视化（需要已有清洗后的数据）

    Returns:
        分析器实例
    """
    analyzer = HospitalSalesAnalyzer(config_path)

    if visualize_only:
        # 仅可视化模式：加载已清洗的数据
        analyzer.logger.info("仅可视化模式：加载已清洗数据...")
        processed_data_dir = Path(analyzer.config.get('data', {}).get('processed_data_dir', 'outputs/processed_data'))
        cleaned_data_path = processed_data_dir / 'cleaned_data.parquet'

        if not cleaned_data_path.exists():
            cleaned_data_path = processed_data_dir / 'cleaned_data.csv'

        if cleaned_data_path.exists():
            if cleaned_data_path.suffix == '.parquet':
                analyzer.df_clean = pd.read_parquet(cleaned_data_path)
            else:
                analyzer.df_clean = pd.read_csv(cleaned_data_path)
            analyzer.analyze()
            analyzer.visualize()
        else:
            raise FileNotFoundError(f"未找到清洗后的数据: {processed_data_dir}")
    else:
        # 完整分析流程
        analyzer.load_data()
        analyzer.clean_data()
        analyzer.analyze()
        analyzer.visualize()
        analyzer.generate_report()
        analyzer.save_processed_data()

    return analyzer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="P01 医院销售数据分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/analyze.py
  python src/analyze.py --config configs/custom.yaml
  python src/analyze.py --visualize-only
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径 (默认: configs/default.yaml)'
    )

    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='仅生成可视化（需要已有清洗后的数据）'
    )

    args = parser.parse_args()

    try:
        analyzer = run_analysis(
            config_path=args.config,
            visualize_only=args.visualize_only
        )

        print("\n" + "="*60)
        print("✅ 分析完成！")
        print("="*60)

        if not args.visualize_only:
            metrics = analyzer.results.get('overall_metrics', {})
            print(f"\n总销售额: {metrics.get('total_sales', 0):,.2f} 元")
            print(f"总订单数: {metrics.get('total_orders', 0):,}")
            print(f"平均订单金额: {metrics.get('average_order_value', 0):.2f} 元")

            print("\n📊 输出文件:")
            output_dir = Path(analyzer.config.get('data', {}).get('output_dir', 'outputs'))
            print(f"  - 图表: {output_dir / 'figures'}/")
            print(f"  - 报告: {output_dir / 'reports'}/")
            print(f"  - 清洗数据: {analyzer.config.get('data', {}).get('processed_data_dir', 'outputs/processed_data')}/")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
