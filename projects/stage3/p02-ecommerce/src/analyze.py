#!/usr/bin/env python3
"""P02 Ecommerce RFM Analysis - 服装零售RFM客户分析"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent


class RFMAnalyzer:
    """RFM客户价值分析器"""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_visualization()
        self.df = None
        self.rfm = None
        self.results = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'outputs/analysis.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_visualization(self):
        viz_config = self.config.get('visualization', {})
        font_config = viz_config.get('fonts', {})
        plt.rcParams['font.sans-serif'] = font_config.get('family', ['SimHei'])
        plt.rcParams['axes.unicode_minus'] = False
        self.dpi = viz_config.get('figure', {}).get('dpi', 300)
    
    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")
        input_file = self.config['data']['input_file']
        self.df = pd.read_csv(input_file, parse_dates=['order_date'])
        self.logger.info(f"数据加载完成: {len(self.df)} 行, {len(self.df.columns)} 列")
        return self.df
    
    def calculate_rfm(self):
        """计算RFM指标"""
        self.logger.info("计算RFM指标...")
        
        # 分析日期
        analysis_date = self.df['order_date'].max() + pd.Timedelta(days=1)
        
        # 按客户聚合
        self.rfm = self.df.groupby('customer_id').agg({
            'order_date': lambda x: (analysis_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        }).rename(columns={
            'order_date': 'recency',
            'order_id': 'frequency',
            'total_amount': 'monetary'
        })
        
        # 计算RFM评分
        self.rfm['R_score'] = pd.qcut(self.rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        self.rfm['F_score'] = pd.qcut(self.rfm['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
        self.rfm['M_score'] = pd.qcut(self.rfm['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        self.rfm['RFM_score'] = self.rfm['R_score'].astype(int) * 100 + \
                                self.rfm['F_score'].astype(int) * 10 + \
                                self.rfm['M_score'].astype(int)
        
        self.logger.info(f"RFM计算完成: {len(self.rfm)} 个客户")
        return self.rfm
    
    def segment_customers(self):
        """客户细分"""
        self.logger.info("进行客户细分...")
        
        def classify(row):
            r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
            if r >= 4 and f >= 4 and m >= 4:
                return '重要价值客户'
            elif r >= 4 and (f >= 2 or m >= 2):
                return '重要保持客户'
            elif r <= 2 and f >= 4 and m >= 4:
                return '重要挽留客户'
            elif (f >= 3 or m >= 3) and r >= 2:
                return '一般发展客户'
            elif r >= 2 and f >= 2 and m >= 2:
                return '一般维持客户'
            elif r <= 2:
                return '一般挽留客户'
            else:
                return '潜在客户'
        
        self.rfm['customer_segment'] = self.rfm.apply(classify, axis=1)
        self.logger.info("客户细分完成")
        
        for segment, count in self.rfm['customer_segment'].value_counts().items():
            self.logger.info(f"  {segment}: {count} ({count/len(self.rfm)*100:.1f}%)")
        
        return self.rfm
    
    def visualize(self):
        """生成可视化"""
        self.logger.info("生成可视化图表...")
        output_dir = Path(self.config['data']['output_dir']) / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 客户细分饼图
        plt.figure(figsize=(10, 8))
        self.rfm['customer_segment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('客户细分分布', fontsize=16, fontweight='bold')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(output_dir / 'customer_segments.png', dpi=self.dpi)
        plt.close()
        
        # 2. RFM热力图
        segment_rfm = self.rfm.groupby('customer_segment')[['recency', 'frequency', 'monetary']].mean()
        plt.figure(figsize=(10, 6))
        sns.heatmap(segment_rfm.T, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('客户群体RFM特征', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'rfm_heatmap.png', dpi=self.dpi)
        plt.close()
        
        self.logger.info(f"图表已保存到: {output_dir}")
    
    def generate_report(self):
        """生成报告"""
        self.logger.info("生成分析报告...")
        output_dir = Path(self.config['data']['output_dir']) / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        segment_dist = self.rfm['customer_segment'].value_counts()
        
        report = f"""# 服装零售RFM客户分析报告

## 数据概览
- 总客户数: {len(self.rfm):,}
- 总订单数: {self.df['order_id'].nunique():,}
- 总销售额: {self.df['total_amount'].sum():,.2f} 元
- 平均客单价: {self.df.groupby('customer_id')['total_amount'].sum().mean():.2f} 元

## 客户细分结果

"""
        for segment, count in segment_dist.items():
            pct = count / len(self.rfm) * 100
            avg_monetary = self.rfm[self.rfm['customer_segment'] == segment]['monetary'].mean()
            report += f"### {segment}\n"
            report += f"- 客户数: {count} ({pct:.1f}%)\n"
            report += f"- 平均消费: {avg_monetary:.2f} 元\n\n"
        
        report += """
## 营销建议

1. **重要价值客户**: VIP服务，专属优惠，新品优先
2. **重要挽留客户**: 唤回活动，限时折扣，个性化推荐
3. **一般发展客户**: 会员权益，积分激励，社交裂变
4. **潜在客户**: 首单优惠，试用装，低门槛活动
"""
        
        with open(output_dir / 'rfm_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"报告已保存: {output_dir / 'rfm_report.md'}")


def run_rfm_analysis(config_path: str = "configs/default.yaml"):
    """运行完整RFM分析"""
    analyzer = RFMAnalyzer(config_path)
    analyzer.load_data()
    analyzer.calculate_rfm()
    analyzer.segment_customers()
    analyzer.visualize()
    analyzer.generate_report()
    return analyzer


def main():
    parser = argparse.ArgumentParser(description="P02 服装零售RFM分析")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    try:
        analyzer = run_rfm_analysis(args.config)
        print("\n" + "="*60)
        print("✅ 分析完成！")
        print("="*60)
        print(f"\n客户总数: {len(analyzer.rfm):,}")
        print(f"客户细分:")
        for segment, count in analyzer.rfm['customer_segment'].value_counts().items():
            print(f"  {segment}: {count} ({count/len(analyzer.rfm)*100:.1f}%)")
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
