"""
可视化工具模块
提供RFM分析和模型评估的可视化功能
"""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def setup_plot_style(style: str = "seaborn", palette: str = "Set2") -> None:
    """设置绘图样式

    Args:
        style: matplotlib样式
        palette: seaborn调色板
    """
    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]  # 支持中文
    plt.rcParams["axes.unicode_minus"] = False  # 支持负号


def plot_rfm_distribution(
    rfm_data: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """绘制RFM分布图

    Args:
        rfm_data: 包含R、F、M评分的DataFrame
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # R、F、M分布直方图
    for idx, col in enumerate(["recency_score", "frequency_score", "monetary_score"]):
        ax = axes[idx // 2, idx % 2]
        sns.histplot(rfm_data[col], bins=5, ax=ax, kde=False)
        ax.set_title(f"{col.split('_')[0].upper()}评分分布", fontsize=14)
        ax.set_xlabel("评分", fontsize=12)
        ax.set_ylabel("客户数量", fontsize=12)

    # RFM总分分布
    ax = axes[1, 1]
    sns.histplot(rfm_data["rfm_score"], bins=15, ax=ax, kde=True)
    ax.set_title("RFM总分分布", fontsize=14)
    ax.set_xlabel("RFM总分", fontsize=12)
    ax.set_ylabel("客户数量", fontsize=12)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_customer_segments(
    segments: pd.Series, output_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)
) -> None:
    """绘制客户细分饼图

    Args:
        segments: 客户细分标签Series
        output_path: 输出文件路径
        figsize: 图表大小
    """
    segment_counts = segments.value_counts()

    plt.figure(figsize=figsize)
    plt.pie(
        segment_counts.values,
        labels=segment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("客户细分分布", fontsize=14)
    plt.axis("equal")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """绘制ROC曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC曲线 - 流失预测模型", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签列表
        output_path: 输出文件路径
        figsize: 图表大小
    """
    if labels is None:
        labels = ["未流失", "已流失"]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("混淆矩阵", fontsize=14)
    plt.ylabel("真实标签", fontsize=12)
    plt.xlabel("预测标签", fontsize=12)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 15,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """绘制特征重要性图

    Args:
        feature_names: 特征名称列表
        importances: 特征重要性数组
        top_n: 显示前N个重要特征
        output_path: 输出文件路径
        figsize: 图表大小
    """
    # 按重要性排序
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importances, align="center")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("重要性", fontsize=12)
    plt.title(f"Top {top_n} 特征重要性", fontsize=14)
    plt.gca().invert_yaxis()  # 最重要的在上面

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
