#!/usr/bin/env python3
"""
P03 银行营销分类分析
使用逻辑回归和决策树预测客户是否会订购定期存款
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)


class BankMarketingClassifier:
    """银行营销分类器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化分类器

        Args:
            config: 配置字典
        """
        self.config = config
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.scaler = None
        self.label_encoders = {}

        # 设置日志
        self.logger = self._setup_logger()

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = [config['visualization']['font_family']]
        plt.rcParams['axes.unicode_minus'] = False

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        output_dir = Path(self.config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(output_dir / 'analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")
        data_path = Path(self.config['data']['input_file'])

        self.df = pd.read_csv(data_path)
        self.logger.info(f"数据加载完成: {len(self.df)} 行, {len(self.df.columns)} 列")

        # 基本信息
        self.logger.info(f"正样本数量: {(self.df['y'] == 'yes').sum()} ({(self.df['y'] == 'yes').sum() / len(self.df) * 100:.2f}%)")
        self.logger.info(f"负样本数量: {(self.df['y'] == 'no').sum()} ({(self.df['y'] == 'no').sum() / len(self.df) * 100:.2f}%)")

    def preprocess_data(self):
        """数据预处理"""
        self.logger.info("数据预处理...")

        df = self.df.copy()

        # 移除duration特征(数据泄漏风险)
        if self.config['preprocessing']['remove_duration'] and 'duration' in df.columns:
            self.logger.info("移除duration特征(避免数据泄漏)")
            df = df.drop(columns=['duration'])

        # 分离特征和目标
        X = df.drop(columns=['y'])
        y = df['y'].map({'yes': 1, 'no': 0})

        # 识别数值和类别特征
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        self.logger.info(f"数值特征: {numeric_features}")
        self.logger.info(f"类别特征: {categorical_features}")

        # 编码类别特征
        if self.config['preprocessing']['categorical_encoding'] == 'onehot':
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        else:
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )

        # 标准化数值特征
        if self.config['preprocessing']['scale_numeric']:
            self.scaler = StandardScaler()
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )

        self.logger.info(f"训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
        self.logger.info(f"特征数量: {len(self.X_train.columns)}")

    def train_models(self):
        """训练模型"""
        self.logger.info("训练模型...")

        # 类别权重
        class_weight = self.config['preprocessing'].get('class_weight')

        # 1. 逻辑回归
        if self.config['models']['logistic_regression']['enabled']:
            self.logger.info("训练逻辑回归模型...")
            lr_config = self.config['models']['logistic_regression']

            lr = LogisticRegression(
                max_iter=lr_config['max_iter'],
                solver=lr_config['solver'],
                C=lr_config['C'],
                class_weight=class_weight,
                random_state=self.config['data']['random_state']
            )
            lr.fit(self.X_train, self.y_train)
            self.models['logistic_regression'] = lr

            # 交叉验证
            cv_scores = cross_val_score(
                lr, self.X_train, self.y_train,
                cv=self.config['evaluation']['cv_folds'],
                scoring='roc_auc'
            )
            self.logger.info(f"逻辑回归 CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 2. 决策树
        if self.config['models']['decision_tree']['enabled']:
            self.logger.info("训练决策树模型...")
            dt_config = self.config['models']['decision_tree']

            dt = DecisionTreeClassifier(
                max_depth=dt_config['max_depth'],
                min_samples_split=dt_config['min_samples_split'],
                min_samples_leaf=dt_config['min_samples_leaf'],
                criterion=dt_config['criterion'],
                class_weight=class_weight,
                random_state=self.config['data']['random_state']
            )
            dt.fit(self.X_train, self.y_train)
            self.models['decision_tree'] = dt

            # 交叉验证
            cv_scores = cross_val_score(
                dt, self.X_train, self.y_train,
                cv=self.config['evaluation']['cv_folds'],
                scoring='roc_auc'
            )
            self.logger.info(f"决策树 CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate_models(self):
        """评估模型"""
        self.logger.info("评估模型...")

        for model_name, model in self.models.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"模型: {model_name}")
            self.logger.info(f"{'='*50}")

            # 预测
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            self.predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # 计算指标
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }

            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name.upper()}: {value:.4f}")

            # 详细分类报告
            self.logger.info("\n分类报告:")
            self.logger.info("\n" + classification_report(self.y_test, y_pred, target_names=['No', 'Yes']))

    def visualize(self):
        """生成可视化"""
        self.logger.info("生成可视化...")

        figures_dir = Path(self.config['data']['output_dir']) / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        dpi = self.config['visualization']['dpi']
        figsize = tuple(self.config['visualization']['figsize'])

        # 1. 混淆矩阵
        if 'confusion_matrix' in self.config['visualization']['plots']:
            self._plot_confusion_matrices(figures_dir, figsize, dpi)

        # 2. ROC曲线
        if 'roc_curve' in self.config['visualization']['plots']:
            self._plot_roc_curves(figures_dir, figsize, dpi)

        # 3. 特征重要性
        if 'feature_importance' in self.config['visualization']['plots']:
            self._plot_feature_importance(figures_dir, figsize, dpi)

        # 4. 相关性热力图
        if 'correlation_heatmap' in self.config['visualization']['plots']:
            self._plot_correlation_heatmap(figures_dir, figsize, dpi)

        self.logger.info(f"图表已保存到: {figures_dir}")

    def _plot_confusion_matrices(self, output_dir: Path, figsize: tuple, dpi: int):
        """绘制混淆矩阵"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(figsize[0] * n_models, figsize[1]))
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = self.predictions[model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            axes[idx].set_title(f'{model_name}\n混淆矩阵', fontsize=14)
            axes[idx].set_ylabel('实际标签', fontsize=12)
            axes[idx].set_xlabel('预测标签', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=dpi, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, output_dir: Path, figsize: tuple, dpi: int):
        """绘制ROC曲线"""
        plt.figure(figsize=figsize)

        for model_name, model in self.models.items():
            y_pred_proba = self.predictions[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.savefig(output_dir / 'roc_curve.png', dpi=dpi, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, output_dir: Path, figsize: tuple, dpi: int):
        """绘制特征重要性"""
        # 决策树特征重要性
        if 'decision_tree' in self.models:
            dt = self.models['decision_tree']
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': dt.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            plt.figure(figsize=figsize)
            sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
            plt.title('决策树 - Top 15 特征重要性', fontsize=14)
            plt.xlabel('重要性', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance_tree.png', dpi=dpi, bbox_inches='tight')
            plt.close()

        # 逻辑回归系数
        if 'logistic_regression' in self.models:
            lr = self.models['logistic_regression']
            feature_coef = pd.DataFrame({
                'feature': self.X_train.columns,
                'coefficient': lr.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False).head(15)

            plt.figure(figsize=figsize)
            colors = ['red' if x < 0 else 'green' for x in feature_coef['coefficient']]
            sns.barplot(data=feature_coef, x='coefficient', y='feature', palette=colors)
            plt.title('逻辑回归 - Top 15 特征系数', fontsize=14)
            plt.xlabel('系数 (负=降低订购概率, 正=提高订购概率)', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance_logistic.png', dpi=dpi, bbox_inches='tight')
            plt.close()

    def _plot_correlation_heatmap(self, output_dir: Path, figsize: tuple, dpi: int):
        """绘制相关性热力图"""
        # 只绘制原始数值特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            plt.figure(figsize=figsize)
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('数值特征相关性热力图', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=dpi, bbox_inches='tight')
            plt.close()

    def generate_report(self):
        """生成分析报告"""
        self.logger.info("生成分析报告...")

        reports_dir = Path(self.config['data']['output_dir']) / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "# 银行营销分类分析报告\n",
            "## 数据概览\n",
            f"- 总样本数: {len(self.df):,}",
            f"- 正样本数: {(self.df['y'] == 'yes').sum():,} ({(self.df['y'] == 'yes').sum() / len(self.df) * 100:.2f}%)",
            f"- 负样本数: {(self.df['y'] == 'no').sum():,} ({(self.df['y'] == 'no').sum() / len(self.df) * 100:.2f}%)",
            f"- 特征数量: {len(self.X_train.columns)}",
            f"- 训练集大小: {len(self.X_train):,}",
            f"- 测试集大小: {len(self.X_test):,}\n",
            "## 模型评估结果\n"
        ]

        for model_name, model in self.models.items():
            y_pred = self.predictions[model_name]['y_pred']
            y_pred_proba = self.predictions[model_name]['y_pred_proba']

            report_lines.append(f"### {model_name}\n")
            report_lines.append(f"- **准确率**: {accuracy_score(self.y_test, y_pred):.4f}")
            report_lines.append(f"- **精确率**: {precision_score(self.y_test, y_pred):.4f}")
            report_lines.append(f"- **召回率**: {recall_score(self.y_test, y_pred):.4f}")
            report_lines.append(f"- **F1分数**: {f1_score(self.y_test, y_pred):.4f}")
            report_lines.append(f"- **AUC-ROC**: {roc_auc_score(self.y_test, y_pred_proba):.4f}\n")

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            report_lines.append("**混淆矩阵**:\n")
            report_lines.append("```")
            report_lines.append(f"TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
            report_lines.append(f"FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")
            report_lines.append("```\n")

        # 业务建议
        report_lines.extend([
            "## 业务建议\n",
            "### 1. 营销策略优化",
            "- 使用模型预测客户订购概率，优先联系高概率客户",
            "- 建议联系概率 > 0.3 的客户，预期转化率提升3-5倍\n",
            "### 2. 客户细分",
            "- **高潜力** (p > 0.7): 重点跟进，专人服务",
            "- **中等潜力** (0.3 < p < 0.7): 标准营销流程",
            "- **低潜力** (p < 0.3): 低频次或不联系\n",
            "### 3. A/B测试",
            "- 对比模型驱动营销 vs 随机营销的转化率和ROI",
            "- 持续优化模型和营销策略\n"
        ])

        report_path = reports_dir / 'classification_report.md'
        report_path.write_text('\n'.join(report_lines), encoding='utf-8')
        self.logger.info(f"报告已保存: {report_path}")

    def save_models(self):
        """保存模型"""
        self.logger.info("保存模型...")

        models_dir = Path(self.config['data']['output_dir']) / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = models_dir / f'{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"模型已保存: {model_path}")

        # 保存scaler
        if self.scaler:
            scaler_path = models_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"Scaler已保存: {scaler_path}")

    def run(self):
        """运行完整分析流程"""
        self.load_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.visualize()
        self.generate_report()
        self.save_models()
        self.logger.info("分析完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='P03 银行营销分类分析')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 运行分析
    classifier = BankMarketingClassifier(config)
    classifier.run()


if __name__ == '__main__':
    main()
