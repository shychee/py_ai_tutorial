#!/usr/bin/env python3
"""
P04 电信客户流失预测 - 主分析脚本

Usage:
    cd projects/stage3/p04-telecom
    python src/analyze.py --config configs/default.yaml
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml
import pandas as pd

from data.loader import load_telco_data, get_data_summary
from models.churn_predictor import ChurnPredictor
from utils.logger import setup_logger
from utils.metrics import calculate_classification_metrics, format_metrics_for_display
from utils.visualization import (
    setup_plot_style,
    plot_churn_by_feature,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_report(
    output_path: str, data_summary: dict, metrics: dict,
    feature_importance: pd.DataFrame = None,
) -> None:
    lines = [
        "# P04 电信客户流失预测分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据来源**: Kaggle IBM Telco Customer Churn",
        "",
        "## 1. 数据概览",
        "",
        f"- 总客户数: {data_summary['total_records']}",
        f"- 特征数: {data_summary['num_features']}",
        f"- 流失率: {data_summary['churn_rate']:.2%}",
        f"- 在网时长: {data_summary['tenure_range']}",
        f"- 月费范围: {data_summary['monthly_charges_range']}",
        "",
        "## 2. 模型性能",
        "",
        f"- Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"- Precision: {metrics.get('precision', 0):.4f}",
        f"- Recall: {metrics.get('recall', 0):.4f}",
        f"- F1-Score: {metrics.get('f1_score', 0):.4f}",
        f"- AUC-ROC: {metrics.get('roc_auc', 0):.4f}",
        "",
    ]

    if feature_importance is not None:
        lines.extend([
            "## 3. Top 10 重要特征",
            "",
            feature_importance.head(10).to_markdown(index=False),
            "",
        ])

    lines.extend([
        "## 4. 业务建议",
        "",
        "1. 月付合约客户流失风险最高，推荐引导转为年付合约",
        "2. 光纤用户流失率高于 DSL 用户，需关注服务质量",
        "3. 缺少在线安全/技术支持的客户更易流失，可作为增值服务推荐",
        "4. 新客户（tenure < 12月）流失风险高，需加强入网关怀",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(config_path: str) -> None:
    config = load_config(config_path)

    logger = setup_logger(
        name="p04_telecom",
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
        console=config["logging"]["console"],
    )

    logger.info("P04 电信客户流失预测 - 开始分析")

    setup_plot_style(
        style=config["visualization"]["style"],
        palette=config["visualization"]["palette"],
    )

    df = load_telco_data(config["data"]["input_file"])
    data_summary = get_data_summary(df)

    logger.info("生成 EDA 可视化")
    fig_dir = config["paths"]["figures_dir"]
    for feature in ["Contract", "InternetService", "tenure", "MonthlyCharges"]:
        plot_churn_by_feature(df, feature, output_path=f"{fig_dir}churn_by_{feature.lower()}.png")

    predictor = ChurnPredictor(
        model_type=config["model"]["type"],
        **config["model"][config["model"]["type"]],
    )
    X, y = predictor.prepare_features(df)
    train_results = predictor.train(
        X, y,
        test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
    )

    y_test = train_results["y_test"]
    X_test_scaled = train_results["X_test"]
    y_pred = predictor.model.predict(X_test_scaled)
    y_pred_proba = predictor.model.predict_proba(X_test_scaled)[:, 1]

    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
    logger.info(format_metrics_for_display(metrics))

    plot_roc_curve(y_test, y_pred_proba, output_path=f"{fig_dir}roc_curve.png")
    plot_confusion_matrix(y_test, y_pred, output_path=f"{fig_dir}confusion_matrix.png")

    feature_importance = None
    if config["model"]["type"] == "random_forest":
        feature_importance = predictor.get_feature_importance()
        plot_feature_importance(
            feature_importance["feature"].tolist(),
            feature_importance["importance"].values,
            output_path=f"{fig_dir}feature_importance.png",
        )

    model_path = f"{config['paths']['models_dir']}{config['paths']['model_file']}"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(predictor, f)

    generate_report(
        f"{config['paths']['reports_dir']}{config['paths']['report_file']}",
        data_summary, metrics, feature_importance,
    )

    logger.info("分析完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P04 电信客户流失预测")
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
