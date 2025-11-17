"""
日志配置模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """配置并返回logger实例

    Args:
        name: logger名称
        log_file: 日志文件路径,若为None则不写入文件
        level: 日志级别(DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: 日志格式字符串
        console: 是否同时输出到控制台

    Returns:
        logging.Logger: 配置好的logger实例

    Examples:
        >>> logger = setup_logger("my_module", "logs/app.log")
        >>> logger.info("这是一条信息日志")
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 设置日志格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # 添加文件handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 添加控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
