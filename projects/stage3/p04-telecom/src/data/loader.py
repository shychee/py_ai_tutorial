"""
数据加载模块
加载 Kaggle Telco Customer Churn 数据集
"""

from pathlib import Path
from typing import Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_telco_data(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """加载 Kaggle Telco Customer Churn 数据集

    Args:
        file_path: 数据文件路径
        encoding: 文件编码

    Returns:
        加载并做基础清洗后的 DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {file_path}\n"
            "请从 Kaggle 下载: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )

    logger.info(f"加载数据: {file_path}")
    df = pd.read_csv(file_path, encoding=encoding)
    logger.info(f"加载 {len(df)} 行, {len(df.columns)} 列")

    required = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必需列: {missing}")

    # TotalCharges 有空字符串，转为数值
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Churn 转为 0/1
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info(f"流失率: {df['Churn'].mean():.2%}")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """获取数据摘要"""
    return {
        "total_records": len(df),
        "num_features": len(df.columns),
        "churn_rate": df["Churn"].mean(),
        "missing_values": df.isnull().sum().sum(),
        "tenure_range": f"{df['tenure'].min()}-{df['tenure'].max()} 月",
        "monthly_charges_range": f"${df['MonthlyCharges'].min():.0f}-${df['MonthlyCharges'].max():.0f}",
    }
