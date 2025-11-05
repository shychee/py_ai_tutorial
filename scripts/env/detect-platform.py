#!/usr/bin/env python3
"""
平台检测脚本 (Platform Detection Script)

检测当前运行环境的操作系统、硬件、Python版本、GPU支持等信息，
为用户推荐最佳的学习阶段和安装依赖。

Usage:
    python scripts/env/detect-platform.py
    python scripts/env/detect-platform.py --json
    python scripts/env/detect-platform.py --recommend
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


class PlatformDetector:
    """平台检测器"""

    def __init__(self):
        self.info: Dict[str, Any] = {}

    def detect_os(self) -> Dict[str, str]:
        """检测操作系统信息"""
        os_info = {
            "system": platform.system(),  # Darwin, Linux, Windows
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),  # x86_64, arm64, AMD64
            "processor": platform.processor(),
        }

        # 判断具体平台
        if os_info["system"] == "Darwin":
            # macOS
            if os_info["machine"] == "arm64":
                os_info["platform"] = "macOS-ARM64"
                os_info["platform_name"] = "macOS Apple Silicon (M1/M2/M3)"
            else:
                os_info["platform"] = "macOS-Intel"
                os_info["platform_name"] = "macOS Intel (x86_64)"
        elif os_info["system"] == "Linux":
            # Linux
            os_info["platform"] = "Linux"
            os_info["platform_name"] = "Linux (Ubuntu/CentOS/etc.)"
            # 检测WSL2
            try:
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        os_info["platform"] = "WSL2"
                        os_info["platform_name"] = "Windows WSL2"
            except FileNotFoundError:
                pass
        elif os_info["system"] == "Windows":
            # Windows Native
            os_info["platform"] = "Windows"
            os_info["platform_name"] = "Windows 10/11 Native"
        else:
            os_info["platform"] = "Unknown"
            os_info["platform_name"] = "Unknown Platform"

        return os_info

    def detect_python(self) -> Dict[str, str]:
        """检测Python版本信息"""
        python_info = {
            "version": platform.python_version(),
            "version_tuple": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        }

        # 检查Python版本是否符合要求
        major, minor, _ = sys.version_info[:3]
        python_info["meets_requirement"] = (major == 3 and minor >= 9)
        python_info["recommended"] = (major == 3 and minor >= 11)

        return python_info

    def detect_gpu(self) -> Dict[str, Any]:
        """检测GPU信息"""
        gpu_info = {
            "nvidia_available": False,
            "cuda_version": None,
            "mps_available": False,  # Apple Metal Performance Shaders
            "gpu_count": 0,
            "gpu_names": [],
        }

        # 检测NVIDIA GPU (CUDA)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_info["nvidia_available"] = True
                lines = result.stdout.strip().split("\n")
                gpu_info["gpu_count"] = len(lines)
                gpu_info["gpu_names"] = [line.split(",")[0].strip() for line in lines]

                # 获取CUDA版本
                cuda_result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=5
                )
                if "CUDA Version" in cuda_result.stdout:
                    import re
                    match = re.search(r"CUDA Version:\s+([\d.]+)", cuda_result.stdout)
                    if match:
                        gpu_info["cuda_version"] = match.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 检测Apple MPS (Metal Performance Shaders)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    gpu_info["mps_available"] = True
                    gpu_info["gpu_count"] = 1
                    gpu_info["gpu_names"] = ["Apple GPU (MPS)"]
            except ImportError:
                # PyTorch未安装，假设MPS可用
                gpu_info["mps_available"] = True
                gpu_info["gpu_count"] = 1
                gpu_info["gpu_names"] = ["Apple GPU (MPS, PyTorch not installed)"]

        return gpu_info

    def detect_memory(self) -> Dict[str, Any]:
        """检测内存信息"""
        memory_info = {
            "total_gb": 0,
            "available_gb": 0,
        }

        try:
            if platform.system() == "Linux" or platform.system() == "Darwin":
                # Linux/macOS
                if platform.system() == "Linux":
                    with open("/proc/meminfo", "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith("MemTotal:"):
                                memory_info["total_gb"] = int(line.split()[1]) / (1024 ** 2)
                            elif line.startswith("MemAvailable:"):
                                memory_info["available_gb"] = int(line.split()[1]) / (1024 ** 2)
                else:
                    # macOS
                    result = subprocess.run(
                        ["sysctl", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        mem_bytes = int(result.stdout.split(":")[1].strip())
                        memory_info["total_gb"] = mem_bytes / (1024 ** 3)
                        memory_info["available_gb"] = memory_info["total_gb"] * 0.7  # 估算
            elif platform.system() == "Windows":
                # Windows
                result = subprocess.run(
                    ["wmic", "OS", "get", "TotalVisibleMemorySize", "/value"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "=" in line:
                            mem_kb = int(line.split("=")[1])
                            memory_info["total_gb"] = mem_kb / (1024 ** 2)
                            memory_info["available_gb"] = memory_info["total_gb"] * 0.7
        except Exception:
            pass

        return memory_info

    def detect_disk(self) -> Dict[str, Any]:
        """检测磁盘空间信息"""
        disk_info = {
            "total_gb": 0,
            "available_gb": 0,
        }

        try:
            import shutil
            stat = shutil.disk_usage(Path.cwd())
            disk_info["total_gb"] = stat.total / (1024 ** 3)
            disk_info["available_gb"] = stat.free / (1024 ** 3)
        except Exception:
            pass

        return disk_info

    def detect_all(self) -> Dict[str, Any]:
        """检测所有信息"""
        self.info = {
            "os": self.detect_os(),
            "python": self.detect_python(),
            "gpu": self.detect_gpu(),
            "memory": self.detect_memory(),
            "disk": self.detect_disk(),
        }
        return self.info

    def get_recommendations(self) -> Dict[str, Any]:
        """根据检测结果推荐学习路径和依赖安装"""
        if not self.info:
            self.detect_all()

        recommendations = {
            "stages": [],
            "dependencies": [],
            "warnings": [],
            "setup_guides": [],
        }

        os_info = self.info["os"]
        python_info = self.info["python"]
        gpu_info = self.info["gpu"]
        memory_info = self.info["memory"]
        disk_info = self.info["disk"]

        # Python版本检查
        if not python_info["meets_requirement"]:
            recommendations["warnings"].append(
                f"⚠️  Python版本过低 ({python_info['version']})，需要Python ≥3.9。"
                f"请升级Python后再继续。"
            )
            return recommendations

        if not python_info["recommended"]:
            recommendations["warnings"].append(
                f"⚠️  Python版本 {python_info['version']} 可用，但推荐使用Python 3.11+以获得更好性能。"
            )

        # 内存检查
        if memory_info["total_gb"] < 8:
            recommendations["warnings"].append(
                f"⚠️  系统内存较低 ({memory_info['total_gb']:.1f}GB)，推荐至少8GB内存。"
                "部分项目可能运行缓慢。"
            )

        # 磁盘空间检查
        if disk_info["available_gb"] < 10:
            recommendations["warnings"].append(
                f"⚠️  磁盘空间不足 ({disk_info['available_gb']:.1f}GB可用)，"
                "推荐至少10GB可用空间。"
            )

        # Stage 3推荐 (机器学习，CPU即可)
        stage3_rec = {
            "stage": "stage3",
            "name": "机器学习与数据挖掘",
            "available": True,
            "reason": "✅ 可以在CPU上运行，适合所有平台",
            "install_command": 'uv pip install -e ".[stage3]"',
        }
        recommendations["stages"].append(stage3_rec)
        recommendations["dependencies"].append("stage3")

        # Stage 4推荐 (深度学习，推荐GPU)
        stage4_available = memory_info["total_gb"] >= 16 or gpu_info["nvidia_available"] or gpu_info["mps_available"]

        if gpu_info["nvidia_available"]:
            stage4_rec = {
                "stage": "stage4",
                "name": "深度学习",
                "available": True,
                "reason": f"✅ 检测到NVIDIA GPU ({gpu_info['gpu_names'][0]})，支持CUDA加速",
                "install_command": 'uv pip install -e ".[stage4-gpu]"',
                "gpu_type": "CUDA",
            }
            recommendations["dependencies"].append("stage4-gpu")
        elif gpu_info["mps_available"]:
            stage4_rec = {
                "stage": "stage4",
                "name": "深度学习",
                "available": True,
                "reason": "✅ 检测到Apple Silicon芯片，支持MPS加速",
                "install_command": 'uv pip install -e ".[stage4-mps]"',
                "gpu_type": "MPS",
            }
            recommendations["dependencies"].append("stage4-mps")
        elif memory_info["total_gb"] >= 16:
            stage4_rec = {
                "stage": "stage4",
                "name": "深度学习",
                "available": True,
                "reason": "⚠️  无GPU加速，使用CPU模式（训练速度较慢）",
                "install_command": 'uv pip install -e ".[stage4-cpu]"',
                "gpu_type": "CPU",
            }
            recommendations["dependencies"].append("stage4-cpu")
            recommendations["warnings"].append(
                "💡 Stage 4深度学习项目在CPU上训练较慢，推荐使用GPU或云端环境。"
            )
        else:
            stage4_rec = {
                "stage": "stage4",
                "name": "深度学习",
                "available": False,
                "reason": "❌ 内存不足且无GPU，不推荐本地运行Stage 4",
                "install_command": None,
            }
            recommendations["warnings"].append(
                "❌ Stage 4深度学习需要16GB+内存或GPU支持。"
                "建议使用Google Colab或云端GPU环境。"
            )

        recommendations["stages"].append(stage4_rec)

        # Stage 5推荐 (大模型，内存要求高)
        stage5_available = memory_info["total_gb"] >= 16

        if stage5_available:
            stage5_rec = {
                "stage": "stage5",
                "name": "AIGC与大模型",
                "available": True,
                "reason": "✅ 内存充足，可运行Stage 5（本地LLM推理需32GB+）",
                "install_command": 'uv pip install -e ".[stage5]"',
            }
            recommendations["dependencies"].append("stage5")
            if memory_info["total_gb"] < 32:
                recommendations["warnings"].append(
                    "💡 Stage 5使用API调用模式（OpenAI/DeepSeek）。"
                    "本地运行LLM需要32GB+内存。"
                )
        else:
            stage5_rec = {
                "stage": "stage5",
                "name": "AIGC与大模型",
                "available": False,
                "reason": "❌ 内存不足 (需要16GB+)，不推荐本地运行",
                "install_command": None,
            }
            recommendations["warnings"].append(
                "❌ Stage 5大模型开发需要16GB+内存。建议使用云端环境。"
            )

        recommendations["stages"].append(stage5_rec)

        # 推荐安装指引
        platform_key = os_info["platform"]
        setup_guide_map = {
            "macOS-Intel": "docs/cross-platform/setup-macos-intel.md",
            "macOS-ARM64": "docs/cross-platform/setup-macos-arm64.md",
            "Linux": "docs/cross-platform/setup-linux.md",
            "WSL2": "docs/cross-platform/setup-windows-wsl2.md",
            "Windows": "docs/cross-platform/setup-windows-native.md",
        }
        recommendations["setup_guides"].append(
            setup_guide_map.get(platform_key, "docs/cross-platform/troubleshooting.md")
        )

        return recommendations

    def print_report(self, show_recommendations: bool = False):
        """打印检测报告"""
        if not self.info:
            self.detect_all()

        print("=" * 60)
        print("🖥️  平台检测报告 (Platform Detection Report)")
        print("=" * 60)
        print()

        # 操作系统
        os_info = self.info["os"]
        print(f"📱 操作系统: {os_info['platform_name']}")
        print(f"   系统: {os_info['system']} {os_info['release']}")
        print(f"   架构: {os_info['machine']}")
        print()

        # Python
        python_info = self.info["python"]
        status = "✅" if python_info["recommended"] else ("⚠️ " if python_info["meets_requirement"] else "❌")
        print(f"🐍 Python版本: {status} {python_info['version']}")
        print(f"   实现: {python_info['implementation']}")
        print()

        # GPU
        gpu_info = self.info["gpu"]
        if gpu_info["nvidia_available"]:
            print(f"🎮 GPU: ✅ NVIDIA CUDA (版本 {gpu_info['cuda_version']})")
            for i, name in enumerate(gpu_info["gpu_names"]):
                print(f"   GPU {i}: {name}")
        elif gpu_info["mps_available"]:
            print(f"🎮 GPU: ✅ Apple MPS (Metal Performance Shaders)")
        else:
            print(f"🎮 GPU: ❌ 未检测到GPU加速")
        print()

        # 内存
        memory_info = self.info["memory"]
        mem_status = "✅" if memory_info["total_gb"] >= 16 else ("⚠️ " if memory_info["total_gb"] >= 8 else "❌")
        print(f"💾 内存: {mem_status} {memory_info['total_gb']:.1f}GB 总内存")
        if memory_info["available_gb"] > 0:
            print(f"   可用: {memory_info['available_gb']:.1f}GB")
        print()

        # 磁盘
        disk_info = self.info["disk"]
        disk_status = "✅" if disk_info["available_gb"] >= 20 else ("⚠️ " if disk_info["available_gb"] >= 10 else "❌")
        print(f"💿 磁盘空间: {disk_status} {disk_info['available_gb']:.1f}GB 可用")
        print(f"   总空间: {disk_info['total_gb']:.1f}GB")
        print()

        # 推荐
        if show_recommendations:
            print("=" * 60)
            print("📋 推荐学习路径 (Recommended Learning Path)")
            print("=" * 60)
            print()

            recommendations = self.get_recommendations()

            # 警告信息
            if recommendations["warnings"]:
                print("⚠️  注意事项:")
                for warning in recommendations["warnings"]:
                    print(f"   {warning}")
                print()

            # 阶段推荐
            print("📚 可用学习阶段:")
            for stage_rec in recommendations["stages"]:
                status = "✅" if stage_rec["available"] else "❌"
                print(f"   {status} {stage_rec['name']} ({stage_rec['stage']})")
                print(f"      {stage_rec['reason']}")
                if stage_rec["install_command"]:
                    print(f"      安装命令: {stage_rec['install_command']}")
                print()

            # 安装指引
            print("📖 推荐安装指引:")
            for guide in recommendations["setup_guides"]:
                print(f"   {guide}")
            print()

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="检测当前运行环境并推荐学习路径"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出JSON格式（用于程序调用）",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="显示推荐的学习路径和依赖安装",
    )
    args = parser.parse_args()

    detector = PlatformDetector()
    detector.detect_all()

    if args.json:
        # JSON输出
        output = {
            "detection": detector.info,
            "recommendations": detector.get_recommendations() if args.recommend else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # 人类可读输出
        detector.print_report(show_recommendations=args.recommend or True)


if __name__ == "__main__":
    main()
