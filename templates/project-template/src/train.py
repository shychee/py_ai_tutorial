#!/usr/bin/env python3
"""
模型训练脚本 (Model Training Script)

Usage:
    python src/train.py --config configs/default.yaml
    python src/train.py --config configs/experiment.yaml --verbose
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """加载数据集

    Args:
        data_path: 数据文件路径

    Returns:
        加载的DataFrame
    """
    return pd.read_csv(data_path)


def preprocess_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """数据预处理

    Args:
        df: 原始数据
        config: 配置字典

    Returns:
        (X, y): 特征和标签
    """
    # TODO: 实现数据预处理逻辑
    # 1. 处理缺失值
    # 2. 处理异常值
    # 3. 特征编码
    # 4. 特征缩放
    pass


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分割数据集

    Args:
        X: 特征
        y: 标签
        config: 配置字典

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    train_ratio = config["data"]["train_ratio"]
    test_ratio = config["data"]["test_ratio"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Any:
    """训练模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        config: 配置字典
        logger: 日志记录器

    Returns:
        训练好的模型
    """
    model_type = config["model"]["type"]
    model_params = config["model"]["params"]

    logger.info(f"训练模型: {model_type}")
    logger.info(f"模型参数: {model_params}")

    if model_type == "random_forest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.fit(X_train, y_train)
    logger.info("模型训练完成")

    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger
) -> Dict[str, float]:
    """评估模型

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        logger: 日志记录器

    Returns:
        评估指标字典
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"测试准确率: {accuracy:.4f}")
    logger.info("\n分类报告:")
    logger.info(classification_report(y_test, y_pred))

    return {"accuracy": accuracy}


def save_model(
    model: Any,
    output_dir: Path,
    model_name: str = "model.pkl"
) -> None:
    """保存模型

    Args:
        model: 训练好的模型
        output_dir: 输出目录
        model_name: 模型文件名
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_name

    joblib.dump(model, model_path)
    print(f"模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练机器学习模型")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    log_level = "DEBUG" if args.verbose else config["output"]["log_level"]
    logger = setup_logging(log_level)

    logger.info("=" * 60)
    logger.info(f"项目: {config['project']['name']}")
    logger.info("=" * 60)

    # TODO: 实现完整的训练流程
    # 1. 加载数据
    # 2. 数据预处理
    # 3. 分割数据集
    # 4. 训练模型
    # 5. 评估模型
    # 6. 保存模型

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
