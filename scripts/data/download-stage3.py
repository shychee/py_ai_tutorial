#!/usr/bin/env python3
"""
阶段3数据下载脚本 (Stage 3 Data Download Script)

下载阶段3（机器学习与数据挖掘）所需的9个数据集。

Usage:
    python scripts/data/download-stage3.py
    python scripts/data/download-stage3.py --dataset DS-S3-P01-HOSPITAL
    python scripts/data/download-stage3.py --verify-only
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import yaml

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DatasetDownloader:
    """数据集下载器"""

    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = data_dir
        self.config_path = config_path
        self.datasets_config: List[Dict] = []

    def load_config(self):
        """加载数据集配置"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.datasets_config = [
                ds for ds in config.get("datasets", [])
                if ds["stage_id"] == "stage3"
            ]

    def calculate_checksum(self, file_path: Path) -> str:
        """计算文件SHA256校验和"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def verify_file(self, file_path: Path, expected_checksum: str) -> bool:
        """验证文件完整性"""
        if not file_path.exists():
            return False

        if expected_checksum == "PLACEHOLDER_CHECKSUM_TO_BE_GENERATED":
            print(f"   ⚠️  校验和未设置，跳过验证: {file_path.name}")
            return True

        actual_checksum = self.calculate_checksum(file_path)
        return actual_checksum == expected_checksum

    def download_file(self, url: str, dest_path: Path) -> bool:
        """下载文件（使用urllib）"""
        try:
            import urllib.request
            print(f"   📥 下载中: {url}")
            print(f"   ⬇️  保存到: {dest_path}")

            # 创建目标目录
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 下载文件
            urllib.request.urlretrieve(url, dest_path)
            print(f"   ✅ 下载完成: {dest_path.name}")
            return True

        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            return False

    def download_dataset(self, dataset_id: str) -> bool:
        """下载单个数据集"""
        # 查找数据集配置
        dataset_config = None
        for ds in self.datasets_config:
            if ds["id"] == dataset_id:
                dataset_config = ds
                break

        if not dataset_config:
            print(f"❌ 未找到数据集: {dataset_id}")
            return False

        print(f"\n📦 数据集: {dataset_config['name']} ({dataset_id})")
        print(f"   项目: {dataset_config['project_id']}")
        print(f"   描述: {dataset_config['description'][:60]}...")

        # 下载文件
        for file_info in dataset_config["files"]:
            filename = file_info["filename"]
            file_path = self.data_dir / "stage3" / filename
            expected_checksum = file_info["checksum_sha256"]

            # 检查文件是否已存在且完整
            if file_path.exists():
                print(f"   📄 文件已存在: {filename}")
                if self.verify_file(file_path, expected_checksum):
                    print(f"   ✅ 校验通过，跳过下载")
                    continue
                else:
                    print(f"   ⚠️  校验失败，重新下载")

            # 下载文件
            download_url = dataset_config["source"]["url"]

            # 注意：实际实现时，这里需要真实的下载URL
            # 目前使用placeholder标记需要手动处理
            if "github.com" in download_url or "releases/download" in download_url:
                success = self.download_file(download_url, file_path)
                if not success:
                    # 尝试镜像URL
                    mirror_url = dataset_config["source"].get("mirror_url")
                    if mirror_url:
                        print(f"   🔄 尝试镜像地址...")
                        success = self.download_file(mirror_url, file_path)

                if success and expected_checksum != "PLACEHOLDER_CHECKSUM_TO_BE_GENERATED":
                    # 验证下载的文件
                    if self.verify_file(file_path, expected_checksum):
                        print(f"   ✅ 文件验证通过")
                    else:
                        print(f"   ❌ 文件验证失败，校验和不匹配")
                        return False
            else:
                print(f"   ⚠️  数据集尚未发布，请访问: {download_url}")
                print(f"   💡 提示: 数据集将在教程正式发布时提供下载链接")
                # 创建占位符文件（用于开发测试）
                self._create_placeholder_file(file_path, file_info)
                return True

        return True

    def _create_placeholder_file(self, file_path: Path, file_info: Dict):
        """创建占位符CSV文件（用于开发测试）"""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 简单CSV占位符
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 占位符数据文件\n")
            f.write(f"# 文件名: {file_info['filename']}\n")
            f.write(f"# 大小: {file_info['size_mb']}MB\n")
            f.write(f"# 行数: {file_info['rows']}\n")
            f.write(f"# 列数: {file_info['columns']}\n")
            f.write("# 此文件为占位符，实际数据将在教程发布时提供\n")

        print(f"   📝 创建占位符文件: {file_path}")

    def download_all(self) -> bool:
        """下载所有阶段3数据集"""
        print("=" * 60)
        print("📚 阶段3数据下载 (Stage 3 Data Download)")
        print("=" * 60)
        print(f"数据目录: {self.data_dir}/stage3")
        print(f"数据集数量: {len(self.datasets_config)}")
        print()

        success_count = 0
        for dataset in self.datasets_config:
            if self.download_dataset(dataset["id"]):
                success_count += 1

        print("\n" + "=" * 60)
        print(f"✅ 下载完成: {success_count}/{len(self.datasets_config)} 个数据集")
        print("=" * 60)

        return success_count == len(self.datasets_config)

    def verify_all(self) -> bool:
        """验证所有已下载的数据集"""
        print("=" * 60)
        print("🔍 数据验证 (Data Verification)")
        print("=" * 60)
        print()

        verified_count = 0
        missing_count = 0

        for dataset in self.datasets_config:
            dataset_id = dataset["id"]
            print(f"📦 {dataset['name']} ({dataset_id})")

            for file_info in dataset["files"]:
                filename = file_info["filename"]
                file_path = self.data_dir / "stage3" / filename
                expected_checksum = file_info["checksum_sha256"]

                if not file_path.exists():
                    print(f"   ❌ 文件缺失: {filename}")
                    missing_count += 1
                elif self.verify_file(file_path, expected_checksum):
                    print(f"   ✅ 文件完整: {filename}")
                    verified_count += 1
                else:
                    print(f"   ❌ 校验失败: {filename}")

        print("\n" + "=" * 60)
        print(f"验证结果: {verified_count} 个文件完整, {missing_count} 个文件缺失")
        print("=" * 60)

        return missing_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="下载阶段3（机器学习与数据挖掘）数据集"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="指定下载单个数据集（例如: DS-S3-P01-HOSPITAL）",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证已下载的数据集，不下载",
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

    # 初始化下载器
    downloader = DatasetDownloader(
        data_dir=args.data_dir,
        config_path=args.config,
    )

    # 加载配置
    if not args.config.exists():
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    downloader.load_config()

    # 验证模式
    if args.verify_only:
        success = downloader.verify_all()
        sys.exit(0 if success else 1)

    # 下载模式
    if args.dataset:
        # 下载单个数据集
        success = downloader.download_dataset(args.dataset)
        sys.exit(0 if success else 1)
    else:
        # 下载所有数据集
        success = downloader.download_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
