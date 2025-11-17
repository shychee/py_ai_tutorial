"""
数据加载模块
负责从文件加载原始数据
"""

from pathlib import Path
from typing import Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_telecom_data(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """加载通讯公司客户数据

    Args:
        file_path: 数据文件路径
        encoding: 文件编码

    Returns:
        pd.DataFrame: 加载的数据

    Raises:
        FileNotFoundError: 文件不存在
        pd.errors.ParserError: 文件格式错误

    Examples:
        >>> df = load_telecom_data("data/telecom_customer_data.csv")
        >>> print(f"加载了{len(df)}条客户记录")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        error_msg = f"数据文件不存在: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        logger.info(f"开始加载数据: {file_path}")
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"成功加载{len(df)}行, {len(df.columns)}列数据")

        # 基本数据验证
        required_columns = [
            "customer_id",
            "registration_date",
            "last_transaction_date",
            "transaction_count",
            "total_amount",
            "churn",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")

        # 转换日期列
        date_columns = ["registration_date", "last_transaction_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.debug(f"已将{col}转换为日期类型")

        return df

    except pd.errors.ParserError as e:
        logger.error(f"文件格式错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> dict:
    """获取数据摘要信息

    Args:
        df: 数据DataFrame

    Returns:
        dict: 包含数据摘要的字典

    Examples:
        >>> summary = get_data_summary(df)
        >>> print(f"流失率: {summary['churn_rate']:.2%}")
    """
    summary = {
        "total_records": len(df),
        "num_features": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "churn_rate": df["churn"].mean() if "churn" in df.columns else None,
        "date_range": {
            "min_registration": df["registration_date"].min()
            if "registration_date" in df.columns
            else None,
            "max_registration": df["registration_date"].max()
            if "registration_date" in df.columns
            else None,
            "latest_transaction": df["last_transaction_date"].max()
            if "last_transaction_date" in df.columns
            else None,
        },
    }

    logger.info(f"数据摘要: {summary['total_records']}条记录, 流失率: {summary['churn_rate']:.2%}")
    return summary
