"""
可视化工具模块
"""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def setup_plot_style(style: str = "seaborn-v0_8", palette: str = "Set2") -> None:
    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_or_show(output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_churn_by_feature(
    df: pd.DataFrame, feature: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """按特征分组绘制流失率对比图"""
    plt.figure(figsize=figsize)

    if df[feature].dtype in ["float64", "int64"] and df[feature].nunique() > 10:
        df_no = df[df["Churn"] == 0][feature]
        df_yes = df[df["Churn"] == 1][feature]
        plt.hist(df_no, bins=30, alpha=0.5, label="未流失", density=True)
        plt.hist(df_yes, bins=30, alpha=0.5, label="已流失", density=True)
        plt.xlabel(feature)
        plt.ylabel("密度")
        plt.legend()
    else:
        ct = pd.crosstab(df[feature], df["Churn"], normalize="index")
        ct.columns = ["未流失", "已流失"]
        ct.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.ylabel("比例")
        plt.xticks(rotation=45, ha="right")

    plt.title(f"流失率 vs {feature}")
    plt.tight_layout()
    _save_or_show(output_path)


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC 曲线 - 流失预测")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    _save_or_show(output_path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    labels = ["未流失", "已流失"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("混淆矩阵")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    _save_or_show(output_path)


def plot_feature_importance(
    feature_names: list, importances: np.ndarray,
    top_n: int = 15,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importances, align="center")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("重要性")
    plt.title(f"Top {top_n} 特征重要性")
    plt.gca().invert_yaxis()
    _save_or_show(output_path)
