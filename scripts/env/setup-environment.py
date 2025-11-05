#!/usr/bin/env python3
"""
环境自动化配置脚本 (Environment Setup Automation Script)

自动检测平台并配置Python虚拟环境、安装依赖。

Usage:
    python scripts/env/setup-environment.py --stage stage3
    python scripts/env/setup-environment.py --stage all
    python scripts/env/setup-environment.py --stage stage4 --gpu
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class EnvironmentSetup:
    """环境配置自动化"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.venv_path = project_root / ".venv"

    def check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
            print("   需要Python ≥3.9")
            return False

    def check_uv_installed(self) -> bool:
        """检查uv是否已安装"""
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ uv已安装: {version}")
                return True
        except FileNotFoundError:
            pass

        print("❌ uv未安装")
        return False

    def install_uv(self) -> bool:
        """安装uv包管理器"""
        print("\n📦 安装uv包管理器...")

        import platform
        system = platform.system()

        try:
            if system in ["Darwin", "Linux"]:
                # macOS/Linux
                cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
                subprocess.run(cmd, shell=True, check=True)
            elif system == "Windows":
                # Windows
                cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
                subprocess.run(cmd, shell=True, check=True)
            else:
                print(f"❌ 不支持的操作系统: {system}")
                return False

            print("✅ uv安装完成")
            print("   请运行以下命令激活uv:")
            if system in ["Darwin", "Linux"]:
                print("   source $HOME/.cargo/env")
            return True

        except Exception as e:
            print(f"❌ uv安装失败: {e}")
            return False

    def create_venv(self, python_version: str = "3.11") -> bool:
        """创建虚拟环境"""
        if self.venv_path.exists():
            print(f"✅ 虚拟环境已存在: {self.venv_path}")
            return True

        print(f"\n🐍 创建虚拟环境 (Python {python_version})...")

        try:
            subprocess.run(
                ["uv", "venv", "--python", python_version],
                cwd=self.project_root,
                check=True
            )
            print(f"✅ 虚拟环境创建成功: {self.venv_path}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 虚拟环境创建失败: {e}")
            return False

    def install_dependencies(self, stage: str, use_gpu: bool = False) -> bool:
        """安装依赖"""
        print(f"\n📦 安装{stage}依赖...")

        # 确定安装包
        if stage == "all":
            packages = "[all]"
        elif stage == "stage3":
            packages = "[stage3]"
        elif stage == "stage4":
            if use_gpu:
                # 检测GPU类型
                try:
                    from detect_platform import PlatformDetector
                    detector = PlatformDetector()
                    gpu_info = detector.detect_gpu()

                    if gpu_info["nvidia_available"]:
                        packages = "[stage4-gpu]"
                        print("   检测到NVIDIA GPU，使用CUDA版本")
                    elif gpu_info["mps_available"]:
                        packages = "[stage4-mps]"
                        print("   检测到Apple GPU，使用MPS版本")
                    else:
                        packages = "[stage4-cpu]"
                        print("   ⚠️  未检测到GPU，使用CPU版本")
                except:
                    packages = "[stage4-cpu]"
                    print("   无法检测GPU，使用CPU版本")
            else:
                packages = "[stage4-cpu]"
        elif stage == "stage5":
            packages = "[stage5]"
        elif stage == "dev":
            packages = "[dev]"
        elif stage == "docs":
            packages = "[docs]"
        else:
            print(f"❌ 不支持的阶段: {stage}")
            return False

        try:
            # 使用uv安装
            cmd = ["uv", "pip", "install", "-e", f".{packages}"]
            print(f"   执行: {' '.join(cmd)}")

            subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True
            )

            print(f"✅ {stage}依赖安装完成")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False

    def verify_installation(self, stage: str) -> bool:
        """验证安装"""
        print(f"\n🔍 验证{stage}环境...")

        # 获取Python解释器路径
        python_path = self.venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = self.venv_path / "Scripts" / "python.exe"

        if not python_path.exists():
            print("❌ 找不到虚拟环境Python解释器")
            return False

        # 验证核心包
        test_imports = []
        if stage in ["stage3", "all"]:
            test_imports.extend(["numpy", "pandas", "sklearn", "matplotlib"])
        if stage in ["stage4", "all"]:
            test_imports.extend(["torch"])
        if stage in ["stage5", "all"]:
            test_imports.extend(["langchain"])

        failed = []
        for package in test_imports:
            try:
                result = subprocess.run(
                    [str(python_path), "-c", f"import {package}; print({package}.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print(f"   ✅ {package}: {version}")
                else:
                    failed.append(package)
                    print(f"   ❌ {package}: 导入失败")
            except Exception as e:
                failed.append(package)
                print(f"   ❌ {package}: {e}")

        if failed:
            print(f"\n⚠️  以下包导入失败: {', '.join(failed)}")
            return False

        print("\n✅ 环境验证通过")
        return True

    def print_next_steps(self, stage: str):
        """打印后续步骤"""
        print("\n" + "=" * 70)
        print("🎉 环境配置完成！")
        print("=" * 70)
        print("\n📝 后续步骤:\n")

        print("1. 激活虚拟环境:")
        print("   source .venv/bin/activate  # macOS/Linux")
        print("   .venv\\Scripts\\Activate.ps1  # Windows PowerShell\n")

        print("2. 下载数据集:")
        if stage in ["stage3", "all"]:
            print("   python scripts/data/download-stage3.py")
        if stage in ["stage4", "all"]:
            print("   python scripts/data/download-stage4.py")
        if stage in ["stage5", "all"]:
            print("   python scripts/data/download-stage5.py")
        print()

        print("3. 验证数据:")
        print("   python scripts/data/verify.py --stage 3")
        print()

        print("4. 开始学习:")
        print("   jupyter lab")
        print("   # 在浏览器中打开 notebooks/stage3/")
        print()

        print("📖 更多信息:")
        print("   README.md - 项目概述")
        print("   docs/cross-platform/ - 跨平台配置指引")
        print("   IMPLEMENTATION_GUIDE.md - 实施指南")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="自动配置Python AI教程环境"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["stage3", "stage4", "stage5", "all", "dev", "docs"],
        help="要配置的学习阶段",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="安装GPU版本依赖（仅适用于stage4）",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="跳过虚拟环境创建（如果已存在）",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default="3.11",
        help="Python版本（默认: 3.11）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 Python AI教程 - 环境自动化配置")
    print("=" * 70)
    print(f"目标阶段: {args.stage}")
    print(f"项目路径: {PROJECT_ROOT}")
    print()

    # 初始化配置器
    setup = EnvironmentSetup(PROJECT_ROOT)

    # 1. 检查Python版本
    if not setup.check_python_version():
        sys.exit(1)

    # 2. 检查uv
    if not setup.check_uv_installed():
        print("\n💡 提示: 需要先安装uv包管理器")
        response = input("是否现在安装? (y/n): ")
        if response.lower() == 'y':
            if not setup.install_uv():
                sys.exit(1)
            print("\n请重启终端或运行以下命令后再次执行本脚本:")
            print("source $HOME/.cargo/env")
            sys.exit(0)
        else:
            print("\n请手动安装uv:")
            print("curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)

    # 3. 创建虚拟环境
    if not args.skip_venv:
        if not setup.create_venv(args.python_version):
            sys.exit(1)

    # 4. 安装依赖
    if not setup.install_dependencies(args.stage, args.gpu):
        sys.exit(1)

    # 5. 验证安装
    if not setup.verify_installation(args.stage):
        print("\n⚠️  环境验证失败，但依赖已安装")
        print("   请手动检查环境配置")

    # 6. 打印后续步骤
    setup.print_next_steps(args.stage)


if __name__ == "__main__":
    main()
