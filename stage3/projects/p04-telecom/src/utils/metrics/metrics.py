"""
评估指标计算模块
提供流失预测模型的各种评估指标
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """计算分类任务的评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率(用于计算AUC,可选)

    Returns:
        Dict[str, float]: 包含各项指标的字典

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = calculate_classification_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # 如果提供了预测概率,计算AUC
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # 如果只有一个类别,AUC无法计算
            metrics["roc_auc"] = None

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        np.ndarray: 2x2混淆矩阵 [[TN, FP], [FN, TP]]

    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> cm = get_confusion_matrix(y_true, y_pred)
        >>> print(cm)
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None,
) -> str:
    """生成分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表,如["未流失", "已流失"]

    Returns:
        str: 分类报告字符串

    Examples:
        >>> report = get_classification_report(y_true, y_pred, ["未流失", "已流失"])
        >>> print(report)
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """格式化指标用于显示

    Args:
        metrics: 指标字典

    Returns:
        str: 格式化后的字符串

    Examples:
        >>> metrics = {"accuracy": 0.851, "precision": 0.782}
        >>> print(format_metrics_for_display(metrics))
    """
    lines = ["模型评估指标:", "=" * 40]

    for key, value in metrics.items():
        if value is None:
            lines.append(f"{key.replace('_', ' ').title()}: N/A")
        elif isinstance(value, float):
            lines.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")

    lines.append("=" * 40)
    return "\n".join(lines)
