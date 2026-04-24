#!/usr/bin/env python3
"""
Stage 3 数据集状态检查与下载指引

Usage:
    python scripts/data/download-stage3.py
    python scripts/data/download-stage3.py --verify-only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage3"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
END = "\033[0m"

DATASETS = [
    {
        "id": "P01",
        "name": "朝阳医院销售数据",
        "filename": "hospital_sales.csv",
        "source": "included",
        "rows": 1000,
    },
    {
        "id": "P02",
        "name": "服装零售销售数据",
        "filename": "clothing_retail.csv",
        "source": "included",
        "rows": 2000,
    },
    {
        "id": "P03",
        "name": "银行营销数据 (UCI)",
        "filename": "bank_marketing.csv",
        "source": "included",
        "rows": 45211,
    },
    {
        "id": "P04",
        "name": "电信客户流失数据 (Kaggle)",
        "filename": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "source": "kaggle",
        "rows": 7043,
        "kaggle_dataset": "blastchar/telco-customer-churn",
        "download_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    },
]

COMING_SOON = ["P05 零售超市分析", "P06 滴滴运营分析", "P07 淘宝用户行为", "P08 航空客户价值", "P09 信贷审批"]


def check_file(dataset: dict) -> bool:
    path = DATA_DIR / dataset["filename"]
    if not path.exists():
        return False
    row_count = sum(1 for _ in open(path, encoding="utf-8")) - 1
    if abs(row_count - dataset["rows"]) > dataset["rows"] * 0.1:
        print(f"  {YELLOW}行数异常: 期望 ~{dataset['rows']}, 实际 {row_count}{END}")
        return False
    return True


def print_kaggle_instructions(dataset: dict) -> None:
    print(f"\n  {BLUE}下载方式 1: 浏览器{END}")
    print(f"  访问 {dataset['download_url']}")
    print(f"  下载 CSV 文件并放到 data/stage3/ 目录")
    print(f"\n  {BLUE}下载方式 2: Kaggle CLI{END}")
    print(f"  pip install kaggle")
    print(f"  kaggle datasets download -d {dataset['kaggle_dataset']} -p data/stage3/ --unzip")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 数据集状态检查")
    parser.add_argument("--verify-only", action="store_true", help="仅检查状态")
    args = parser.parse_args()

    print(f"{BOLD}{'=' * 50}{END}")
    print(f"{BOLD}Stage 3 数据集状态检查{END}")
    print(f"{BOLD}{'=' * 50}{END}")
    print(f"数据目录: {DATA_DIR}\n")

    ok_count = 0
    missing = []

    for ds in DATASETS:
        status = check_file(ds)
        icon = f"{GREEN}✓{END}" if status else f"{RED}✗{END}"
        source_tag = "仓库内" if ds["source"] == "included" else "Kaggle"
        print(f"  {icon} {ds['id']}: {ds['name']} [{source_tag}]")

        if status:
            path = DATA_DIR / ds["filename"]
            size_kb = path.stat().st_size / 1024
            if size_kb > 1024:
                print(f"    文件: {ds['filename']} ({size_kb/1024:.1f} MB)")
            else:
                print(f"    文件: {ds['filename']} ({size_kb:.0f} KB)")
            ok_count += 1
        else:
            missing.append(ds)
            if ds["source"] == "kaggle":
                print_kaggle_instructions(ds)
            elif ds["source"] == "included":
                print(f"    {RED}文件缺失! 请重新 git checkout data/stage3/{ds['filename']}{END}")

    print(f"\n  {YELLOW}Coming Soon:{END}")
    for name in COMING_SOON:
        print(f"    - {name}")

    print(f"\n{BOLD}{'=' * 50}{END}")
    print(f"可用: {ok_count}/{len(DATASETS)}  |  缺失: {len(missing)}/{len(DATASETS)}")

    if missing:
        print(f"\n{YELLOW}提示: 缺失的数据集需要手动下载后才能运行对应项目{END}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}所有数据集就绪!{END}")


if __name__ == "__main__":
    main()
