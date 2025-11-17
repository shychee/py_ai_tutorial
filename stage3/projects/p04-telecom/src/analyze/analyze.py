#!/usr/bin/env python3
"""
P04-Telecom项目主分析脚本
通讯公司客户响应速度提升: RFM分析 + 客户流失预测

使用方法:
    python src/analyze.py --config configs/default.yaml
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np

# 导入项目模块
from data.loader import load_telecom_data, get_data_summary
from models.rfm import RFMAnalyzer
from models.churn_predictor import ChurnPredictor
from utils.logger import setup_logger
from utils.metrics import calculate_classification_metrics, get_confusion_matrix, format_metrics_for_display
from utils.visualization import (
    setup_plot_style,
    plot_rfm_distribution,
    plot_customer_segments,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_experiment_info(
    output_path: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    rfm_summary: Dict[str, Any],
) -> None:
    """保存实验元数据

    Args:
        output_path: 输出文件路径
        config: 配置字典
        metrics: 评估指标
        rfm_summary: RFM分析摘要
    """
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{pd.__version__}",  # 使用pandas版本作为示例
        "framework": "scikit-learn",
        "config": config,
        "metrics": metrics,
        "rfm_summary": rfm_summary,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)


def generate_report(
    output_path: str,
    config: Dict[str, Any],
    data_summary: Dict[str, Any],
    rfm_summary: pd.DataFrame,
    metrics: Dict[str, float],
    feature_importance: pd.DataFrame = None,
) -> None:
    """生成Markdown实验报告

    Args:
        output_path: 输出文件路径
        config: 配置字典
        data_summary: 数据摘要
        rfm_summary: RFM分析摘要
        metrics: 评估指标
        feature_importance: 特征重要性DataFrame
    """
    report_lines = [
        "# P04-Telecom 客户响应速度提升分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 项目背景与目标",
        "",
        "本项目旨在通过RFM分析和机器学习方法,识别高价值客户和潜在流失客户,为通讯运营商提供数据驱动的营销策略建议。",
        "",
        "## 2. 数据说明",
        "",
        f"- **总记录数**: {data_summary['total_records']}条",
        f"- **特征数**: {data_summary['num_features']}个",
        f"- **流失率**: {data_summary['churn_rate']:.2%}",
        f"- **数据时间范围**: {data_summary['date_range']['min_registration']} 至 {data_summary['date_range']['latest_transaction']}",
        "",
        "## 3. RFM分析结果",
        "",
        "### 3.1 客户细分分布",
        "",
        rfm_summary.to_markdown(),
        "",
        "### 3.2 关键发现",
        "",
    ]

    # 添加关键发现
    top_segment = rfm_summary.nlargest(1, "count").index[0]
    report_lines.append(f"- **最大客户群体**: {top_segment} ({rfm_summary.loc[top_segment, 'percentage']:.1f}%)")

    if "at_risk" in rfm_summary.index:
        at_risk_pct = rfm_summary.loc["at_risk", "percentage"]
        report_lines.append(f"- **流失风险客户占比**: {at_risk_pct:.1f}%")

    if "champion" in rfm_summary.index:
        champion_pct = rfm_summary.loc["champion", "percentage"]
        report_lines.append(f"- **冠军客户占比**: {champion_pct:.1f}%")

    report_lines.extend([
        "",
        "## 4. 流失预测模型",
        "",
        f"### 4.1 模型配置",
        "",
        f"- **模型类型**: {config['model']['type']}",
        f"- **训练集/测试集划分**: {config['data']['train_split']}/{config['data']['test_split']}",
        "",
        "### 4.2 模型性能",
        "",
        f"- **Accuracy (准确率)**: {metrics.get('accuracy', 0):.4f}",
        f"- **Precision (精确率)**: {metrics.get('precision', 0):.4f}",
        f"- **Recall (召回率)**: {metrics.get('recall', 0):.4f}",
        f"- **F1-Score**: {metrics.get('f1_score', 0):.4f}",
        f"- **AUC-ROC**: {metrics.get('roc_auc', 0):.4f}",
        "",
    ])

    # 添加特征重要性
    if feature_importance is not None:
        report_lines.extend([
            "### 4.3 Top 10 重要特征",
            "",
            feature_importance.head(10).to_markdown(index=False),
            "",
        ])

    # 添加业务建议
    report_lines.extend([
        "## 5. 业务建议",
        "",
        "### 5.1 针对不同客户群体的策略",
        "",
    ])

    if "champion" in rfm_summary.index:
        report_lines.append("- **冠军客户**: 提供VIP服务,优先处理需求,赠送高价值权益")

    if "at_risk" in rfm_summary.index:
        report_lines.append("- **流失风险客户**: 主动联系了解不满原因,提供个性化挽留方案")

    if "hibernating" in rfm_summary.index:
        report_lines.append("- **休眠客户**: 发送激活优惠,如新用户福利、限时折扣等")

    report_lines.extend([
        "",
        "### 5.2 流失预防措施",
        "",
        "1. **预警机制**: 对流失概率>0.7的客户建立预警,提前干预",
        "2. **投诉处理**: 投诉次数是重要流失指标,需优先解决客户投诉",
        "3. **服务质量**: 提升客服响应速度,降低服务呼叫等待时间",
        "4. **合约优化**: 针对按量付费客户,推荐更优惠的套餐计划",
        "",
        "## 6. 结论与改进方向",
        "",
        "### 6.1 结论",
        "",
        f"本项目成功构建了客户价值评估与流失预测模型,模型F1-Score达到{metrics.get('f1_score', 0):.4f},",
        "能够有效识别高价值客户和流失风险客户,为精准营销提供数据支持。",
        "",
        "### 6.2 改进方向",
        "",
        "1. **特征工程**: 增加更多时间序列特征(如消费趋势、活跃度变化)",
        "2. **模型优化**: 尝试XGBoost、LightGBM等集成学习方法",
        "3. **实时预测**: 构建流式数据处理pipeline,实现实时流失预警",
        "4. **A/B测试**: 对不同挽留策略进行A/B测试,量化效果",
        "",
        "---",
        "",
        "**报告生成**: py_ai_tutorial - 阶段3项目P04",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main(config_path: str) -> None:
    """主函数

    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 设置日志
    logger = setup_logger(
        name="p04_telecom",
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
        console=config["logging"]["console"],
    )

    logger.info("=" * 60)
    logger.info("P04-Telecom 分析开始")
    logger.info("=" * 60)

    # 设置绘图样式
    setup_plot_style(
        style=config["visualization"]["style"],
        palette=config["visualization"]["palette"],
    )

    # 1. 加载数据
    logger.info("步骤1: 加载数据")
    df = load_telecom_data(config["data"]["input_file"])
    data_summary = get_data_summary(df)

    # 2. RFM分析
    logger.info("步骤2: RFM分析")
    rfm_analyzer = RFMAnalyzer(
        analysis_date=config["data"]["analysis_date"],
        n_bins=config["rfm"]["n_bins"],
        segment_rules=config["rfm"]["segments"],
    )

    df = rfm_analyzer.calculate_rfm(df)
    df = rfm_analyzer.segment_customers(df)
    rfm_summary = rfm_analyzer.get_segment_summary(df)

    logger.info(f"RFM分析完成:\n{rfm_summary}")

    # 3. 可视化RFM结果
    logger.info("步骤3: 生成RFM可视化")
    plot_rfm_distribution(df, output_path=f"{config['paths']['figures_dir']}rfm_distribution.png")
    plot_customer_segments(df["segment"], output_path=f"{config['paths']['figures_dir']}customer_segments.png")

    # 4. 流失预测模型
    logger.info("步骤4: 训练流失预测模型")
    predictor = ChurnPredictor(
        model_type=config["model"]["type"],
        **config["model"][config["model"]["type"]],
    )

    X, y = predictor.prepare_features(df)
    train_results = predictor.train(
        X,
        y,
        test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
    )

    # 5. 模型评估
    logger.info("步骤5: 模型评估")
    y_test = train_results["y_test"]
    X_test_scaled = train_results["X_test"]

    y_pred = predictor.model.predict(X_test_scaled)
    y_pred_proba = predictor.model.predict_proba(X_test_scaled)[:, 1]

    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
    logger.info(format_metrics_for_display(metrics))

    # 6. 可视化模型结果
    logger.info("步骤6: 生成模型可视化")
    plot_roc_curve(y_test, y_pred_proba, output_path=f"{config['paths']['figures_dir']}churn_prediction_roc.png")
    plot_confusion_matrix(y_test, y_pred, output_path=f"{config['paths']['figures_dir']}confusion_matrix.png")

    # 特征重要性(仅随机森林)
    feature_importance = None
    if config["model"]["type"] == "random_forest":
        feature_importance = predictor.get_feature_importance()
        logger.info(f"Top 10 重要特征:\n{feature_importance.head(10)}")
        plot_feature_importance(
            feature_importance["feature"].tolist(),
            feature_importance["importance"].values,
            output_path=f"{config['paths']['figures_dir']}feature_importance.png",
        )

    # 7. 保存模型
    logger.info("步骤7: 保存模型与结果")
    model_path = f"{config['paths']['models_dir']}{config['paths']['model_file']}"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(predictor, f)
    logger.info(f"模型已保存: {model_path}")

    # 8. 生成报告
    logger.info("步骤8: 生成实验报告")
    rfm_summary_dict = {
        "total_customers": int(len(df)),
        "segments": rfm_summary.to_dict(),
    }

    save_experiment_info(
        config["paths"]["output_dir"] + config["paths"]["experiment_info_file"],
        config,
        metrics,
        rfm_summary_dict,
    )

    generate_report(
        config["paths"]["reports_dir"] + config["paths"]["report_file"],
        config,
        data_summary,
        rfm_summary,
        metrics,
        feature_importance,
    )

    logger.info("=" * 60)
    logger.info("P04-Telecom 分析完成!")
    logger.info(f"输出目录: {config['paths']['output_dir']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P04-Telecom客户响应速度提升分析")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径(默认: configs/default.yaml)",
    )

    args = parser.parse_args()
    main(args.config)
