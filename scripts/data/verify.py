#!/usr/bin/env python3
"""
数据完整性验证脚本 (Data Integrity Verification Script)

验证已下载数据集的完整性（文件存在性、校验和、大小）。

Usage:
    python scripts/data/verify.py --stage 3
    python scripts/data/verify.py --stage all
    python scripts/data/verify.py --checksums-only
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataVerifier:
    """数据验证器"""

    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = data_dir
        self.config_path = config_path
        self.datasets_config: List[Dict] = []

    def load_config(self, stage_filter: Optional[str] = None):
        """加载数据集配置"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if stage_filter and stage_filter != "all":
                self.datasets_config = [
                    ds for ds in config.get("datasets", [])
                    if ds["stage_id"] == f"stage{stage_filter}"
                ]
            else:
                self.datasets_config = config.get("datasets", [])

    def calculate_checksum(self, file_path: Path) -> str:
        """计算文件SHA256校验和"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"   ❌ 无法计算校验和: {e}")
            return ""

    def verify_file(
        self,
        file_path: Path,
        expected_checksum: str,
        expected_size_mb: Optional[int] = None,
        checksums_only: bool = False
    ) -> Dict[str, any]:
        """验证文件完整性"""
        result = {
            "exists": file_path.exists(),
            "checksum_valid": False,
            "size_valid": False,
            "actual_size_mb": 0,
            "actual_checksum": "",
        }

        if not result["exists"]:
            return result

        # 检查文件大小
        actual_size_bytes = file_path.stat().st_size
        result["actual_size_mb"] = actual_size_bytes / (1024 ** 2)

        if expected_size_mb:
            # 允许±10%误差
            size_tolerance = expected_size_mb * 0.1
            result["size_valid"] = abs(result["actual_size_mb"] - expected_size_mb) <= size_tolerance

        # 检查校验和
        if not checksums_only and expected_checksum != "PLACEHOLDER_CHECKSUM_TO_BE_GENERATED":
            result["actual_checksum"] = self.calculate_checksum(file_path)
            result["checksum_valid"] = (result["actual_checksum"] == expected_checksum)
        elif expected_checksum == "PLACEHOLDER_CHECKSUM_TO_BE_GENERATED":
            result["checksum_valid"] = True  # 跳过占位符

        return result

    def verify_dataset(self, dataset_config: Dict, checksums_only: bool = False) -> Dict[str, any]:
        """验证单个数据集"""
        dataset_id = dataset_config["id"]
        stage_id = dataset_config["stage_id"]
        stage_num = stage_id.replace("stage", "")

        result = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_config["name"],
            "files_total": len(dataset_config["files"]),
            "files_verified": 0,
            "files_missing": 0,
            "files_invalid": 0,
            "file_details": [],
        }

        for file_info in dataset_config["files"]:
            filename = file_info["filename"]
            file_path = self.data_dir / stage_id / filename
            expected_checksum = file_info["checksum_sha256"]
            expected_size_mb = file_info.get("size_mb")

            file_result = self.verify_file(
                file_path,
                expected_checksum,
                expected_size_mb,
                checksums_only
            )

            if not file_result["exists"]:
                result["files_missing"] += 1
                status = "❌ 缺失"
            elif not file_result["checksum_valid"] or not file_result["size_valid"]:
                result["files_invalid"] += 1
                status = "⚠️  无效"
            else:
                result["files_verified"] += 1
                status = "✅ 完整"

            result["file_details"].append({
                "filename": filename,
                "status": status,
                "exists": file_result["exists"],
                "checksum_valid": file_result["checksum_valid"],
                "size_valid": file_result["size_valid"],
                "actual_size_mb": file_result["actual_size_mb"],
            })

        return result

    def verify_all(self, checksums_only: bool = False, verbose: bool = True) -> Dict[str, any]:
        """验证所有数据集"""
        summary = {
            "total_datasets": len(self.datasets_config),
            "verified_datasets": 0,
            "missing_datasets": 0,
            "invalid_datasets": 0,
            "total_files": 0,
            "verified_files": 0,
            "missing_files": 0,
            "invalid_files": 0,
            "dataset_results": [],
        }

        if verbose:
            print("=" * 70)
            print("🔍 数据完整性验证 (Data Integrity Verification)")
            print("=" * 70)
            print(f"数据目录: {self.data_dir}")
            print(f"数据集数量: {summary['total_datasets']}")
            print()

        for dataset_config in self.datasets_config:
            result = self.verify_dataset(dataset_config, checksums_only)
            summary["dataset_results"].append(result)

            summary["total_files"] += result["files_total"]
            summary["verified_files"] += result["files_verified"]
            summary["missing_files"] += result["files_missing"]
            summary["invalid_files"] += result["files_invalid"]

            if result["files_missing"] == 0 and result["files_invalid"] == 0:
                summary["verified_datasets"] += 1
            elif result["files_missing"] == result["files_total"]:
                summary["missing_datasets"] += 1
            else:
                summary["invalid_datasets"] += 1

            if verbose:
                # 打印数据集验证结果
                status_icon = "✅" if result["files_missing"] == 0 and result["files_invalid"] == 0 else "❌"
                print(f"{status_icon} {result['dataset_name']} ({result['dataset_id']})")

                for file_detail in result["file_details"]:
                    print(f"   {file_detail['status']} {file_detail['filename']}")
                    if file_detail["exists"]:
                        print(f"      大小: {file_detail['actual_size_mb']:.2f}MB")
                        if not file_detail["checksum_valid"] and not checksums_only:
                            print(f"      ⚠️  校验和不匹配")
                        if not file_detail["size_valid"]:
                            print(f"      ⚠️  文件大小异常")

                print()

        if verbose:
            print("=" * 70)
            print("📊 验证总结 (Summary)")
            print("=" * 70)
            print(f"数据集: {summary['verified_datasets']}/{summary['total_datasets']} 完整")
            if summary['missing_datasets'] > 0:
                print(f"        {summary['missing_datasets']} 个数据集完全缺失")
            if summary['invalid_datasets'] > 0:
                print(f"        {summary['invalid_datasets']} 个数据集部分缺失或损坏")
            print()
            print(f"文件:   {summary['verified_files']}/{summary['total_files']} 完整")
            if summary['missing_files'] > 0:
                print(f"        {summary['missing_files']} 个文件缺失")
            if summary['invalid_files'] > 0:
                print(f"        {summary['invalid_files']} 个文件损坏")
            print("=" * 70)

            if summary["missing_files"] > 0:
                print()
                print("💡 提示: 运行以下命令下载缺失的数据集:")
                stages_with_missing = set()
                for ds_result in summary["dataset_results"]:
                    if ds_result["files_missing"] > 0:
                        dataset_id = ds_result["dataset_id"]
                        stage_num = dataset_id.split("-")[1].replace("S", "")
                        stages_with_missing.add(stage_num)

                for stage_num in sorted(stages_with_missing):
                    print(f"   python scripts/data/download-stage{stage_num}.py")

        return summary

    def print_statistics(self):
        """打印数据集统计信息"""
        print("=" * 70)
        print("📊 数据集统计 (Dataset Statistics)")
        print("=" * 70)
        print()

        by_stage: Dict[str, List] = {}
        for ds in self.datasets_config:
            stage_id = ds["stage_id"]
            if stage_id not in by_stage:
                by_stage[stage_id] = []
            by_stage[stage_id].append(ds)

        for stage_id in sorted(by_stage.keys()):
            datasets = by_stage[stage_id]
            stage_num = stage_id.replace("stage", "")

            total_size_mb = sum(
                file_info["size_mb"]
                for ds in datasets
                for file_info in ds["files"]
            )
            total_files = sum(len(ds["files"]) for ds in datasets)

            print(f"阶段{stage_num}:")
            print(f"   数据集数量: {len(datasets)}")
            print(f"   文件数量:   {total_files}")
            print(f"   总大小:     {total_size_mb / 1024:.2f}GB ({total_size_mb:.0f}MB)")
            print()

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="验证数据集完整性"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["3", "4", "5", "all"],
        help="指定验证的学习阶段（3/4/5/all）",
    )
    parser.add_argument(
        "--checksums-only",
        action="store_true",
        help="仅检查文件存在性和大小，跳过校验和计算（快速模式）",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示数据集统计信息",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="数据存储目录（默认: ./data）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "content" / "datasets.yaml",
        help="数据集配置文件路径",
    )
    args = parser.parse_args()

    # 初始化验证器
    verifier = DataVerifier(
        data_dir=args.data_dir,
        config_path=args.config,
    )

    # 加载配置
    if not args.config.exists():
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    verifier.load_config(stage_filter=args.stage)

    # 统计模式
    if args.stats:
        verifier.print_statistics()
        sys.exit(0)

    # 验证模式
    summary = verifier.verify_all(
        checksums_only=args.checksums_only,
        verbose=True
    )

    # 返回状态码
    if summary["missing_files"] > 0 or summary["invalid_files"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
