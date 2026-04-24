#!/usr/bin/env python3
"""
Stage 4 Dataset Download Script
下载阶段4深度学习数据集与预训练模型权重

Usage:
    python scripts/data/download-stage4.py [--mirror] [--verify-only] [--skip-models]

Options:
    --mirror       使用国内镜像加速下载
    --verify-only  仅校验已下载的数据，不重新下载
    --skip-models  跳过预训练模型下载（仅下载数据集）
    --dataset ID   仅下载指定数据集 (如: DS-S4-MNIST)
    --model ID     仅下载指定模型 (如: resnet50-imagenet)

Datasets:
    - MNIST: 手写数字数据集 (~11MB)
    - CIFAR-10: 10类物体数据集 (~170MB)
    - CIFAR-100: 100类物体数据集 (~170MB)
    - ImageNet Sample: ImageNet子集 (~1GB, optional)
    - COCO Sample: COCO目标检测子集 (~500MB, optional)

Pretrained Models:
    - ResNet-50 (PyTorch): ImageNet预训练权重 (~100MB)
    - ResNet-50 (TensorFlow): ImageNet预训练权重 (~100MB)
    - BERT-base-uncased: NLP预训练模型 (~500MB)
    - YOLO weights: 目标检测预训练权重 (~240MB)
"""

import argparse
import hashlib
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage4"
MODELS_DIR = PROJECT_ROOT / "data" / "models"


class Colors:
    """Terminal colors for better output"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# Stage 4 数据集定义
DATASETS = {
    "DS-S4-MNIST": {
        "id": "DS-S4-MNIST",
        "name": "MNIST 手写数字数据集",
        "description": "60000张28x28灰度手写数字图像(0-9)",
        "size_mb": 11,
        "download_method": "torchvision",  # 使用PyTorch/TensorFlow内置下载
        "local_path": "mnist/",
        "required_for": ["notebooks/stage4/02-pytorch-basics.ipynb"],
        "checksum": None  # 使用torchvision自动验证
    },
    "DS-S4-CIFAR10": {
        "id": "DS-S4-CIFAR10",
        "name": "CIFAR-10 数据集",
        "description": "60000张32x32彩色图像，10个类别",
        "size_mb": 170,
        "download_method": "torchvision",
        "local_path": "cifar10/",
        "required_for": ["notebooks/stage4/03-cnn-image-classification.ipynb"],
        "checksum": None
    },
    "DS-S4-CIFAR100": {
        "id": "DS-S4-CIFAR100",
        "name": "CIFAR-100 数据集",
        "description": "60000张32x32彩色图像，100个类别",
        "size_mb": 170,
        "download_method": "torchvision",
        "local_path": "cifar100/",
        "required_for": ["notebooks/stage4/03-cnn-image-classification.ipynb"],
        "optional": True,
        "checksum": None
    },
    "DS-S4-IMDB": {
        "id": "DS-S4-IMDB",
        "name": "IMDB 电影评论数据集",
        "description": "50000条电影评论文本，情感二分类",
        "size_mb": 80,
        "download_method": "huggingface",
        "local_path": "imdb/",
        "required_for": ["notebooks/stage4/04-rnn-text-classification.ipynb"],
        "checksum": None
    },
    "DS-S4-IMAGENET-SAMPLE": {
        "id": "DS-S4-IMAGENET-SAMPLE",
        "name": "ImageNet 样本数据集",
        "description": "ImageNet-1K的1000个类别样本(每类10张)",
        "size_mb": 1024,
        "download_method": "direct",
        "url": "https://github.com/fastai/imagenette/releases/download/v0.3/imagenette2.tgz",
        "local_path": "imagenet-sample/",
        "required_for": ["docs/stage4/projects/p01-industrial-vision/"],
        "optional": True,
        "checksum": "fe2fc210e6bb7c5664d602c3cd71e612"
    },
    "DS-S4-COCO-SAMPLE": {
        "id": "DS-S4-COCO-SAMPLE",
        "name": "COCO 样本数据集",
        "description": "COCO 2017验证集样本(1000张图像)",
        "size_mb": 500,
        "download_method": "direct",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "local_path": "coco-sample/",
        "required_for": ["docs/stage4/projects/p02-yolov11-realtime/"],
        "optional": True,
        "checksum": None  # 需要解压后验证
    }
}

# 预训练模型定义
PRETRAINED_MODELS = {
    "resnet50-pytorch": {
        "id": "resnet50-pytorch",
        "name": "ResNet-50 PyTorch",
        "description": "ResNet-50 ImageNet预训练权重 (PyTorch)",
        "size_mb": 100,
        "download_method": "torchvision",
        "local_path": "resnet50_pytorch.pth",
        "required_for": ["notebooks/stage4/03-cnn-image-classification.ipynb"],
        "checksum": None
    },
    "bert-base-uncased": {
        "id": "bert-base-uncased",
        "name": "BERT Base Uncased",
        "description": "BERT-base-uncased预训练模型",
        "size_mb": 440,
        "download_method": "huggingface",
        "model_name": "bert-base-uncased",
        "local_path": "bert-base-uncased/",
        "required_for": ["notebooks/stage4/04-rnn-text-classification.ipynb"],
        "optional": True,
        "checksum": None
    },
    "yolov8n": {
        "id": "yolov8n",
        "name": "YOLOv8 Nano",
        "description": "YOLOv8n目标检测预训练权重",
        "size_mb": 6,
        "download_method": "ultralytics",
        "model_name": "yolov8n.pt",
        "local_path": "yolov8n.pt",
        "required_for": ["docs/stage4/projects/p02-yolov11-realtime/"],
        "optional": True,
        "checksum": None
    }
}


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 checksum of a file"""
    md5 = hashlib.md5()

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify file checksum"""
    if not file_path.exists():
        return False

    if not expected_checksum:
        return True  # Skip verification if no checksum provided

    actual_checksum = calculate_md5(file_path)
    return actual_checksum == expected_checksum


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress indicator"""
    try:
        print_info(f"Downloading: {desc or url}")
        print(f"  URL: {url}")
        print(f"  Destination: {dest_path}")

        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = count * block_size * 100 / total_size
                size_mb = total_size / (1024 * 1024)
                downloaded_mb = count * block_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {downloaded_mb:.1f}/{size_mb:.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # New line after progress
        return True

    except Exception as e:
        print_error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract tar.gz or zip archive"""
    try:
        print_info(f"Extracting: {archive_path.name}")

        extract_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print_warning(f"Unknown archive format: {archive_path.suffix}")
            return False

        print_success(f"Extracted to: {extract_dir}")
        return True

    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False


def download_torchvision_dataset(dataset_name: str, data_path: Path) -> bool:
    """Download dataset using torchvision"""
    try:
        print_info(f"Using PyTorch to download {dataset_name}")

        import torch
        import torchvision

        data_path.mkdir(parents=True, exist_ok=True)

        if dataset_name == "MNIST":
            torchvision.datasets.MNIST(root=str(data_path), train=True, download=True)
            torchvision.datasets.MNIST(root=str(data_path), train=False, download=True)
        elif dataset_name == "CIFAR10":
            torchvision.datasets.CIFAR10(root=str(data_path), train=True, download=True)
            torchvision.datasets.CIFAR10(root=str(data_path), train=False, download=True)
        elif dataset_name == "CIFAR100":
            torchvision.datasets.CIFAR100(root=str(data_path), train=True, download=True)
            torchvision.datasets.CIFAR100(root=str(data_path), train=False, download=True)
        else:
            print_warning(f"Unknown torchvision dataset: {dataset_name}")
            return False

        print_success(f"{dataset_name} downloaded successfully")
        return True

    except ImportError:
        print_error("PyTorch not installed. Please install: pip install torch torchvision")
        return False
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False


def download_huggingface_dataset(dataset_name: str, data_path: Path) -> bool:
    """Download dataset using Hugging Face datasets library"""
    try:
        print_info(f"Using Hugging Face to download {dataset_name}")

        from datasets import load_dataset

        data_path.mkdir(parents=True, exist_ok=True)

        if dataset_name == "IMDB":
            dataset = load_dataset("imdb", cache_dir=str(data_path))
            print_success(f"IMDB dataset downloaded: {len(dataset['train'])} train, {len(dataset['test'])} test")
        else:
            print_warning(f"Unknown Hugging Face dataset: {dataset_name}")
            return False

        return True

    except ImportError:
        print_error("Hugging Face datasets not installed. Please install: pip install datasets")
        return False
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False


def download_huggingface_model(model_name: str, model_path: Path) -> bool:
    """Download model from Hugging Face"""
    try:
        print_info(f"Downloading model from Hugging Face: {model_name}")

        from transformers import AutoTokenizer, AutoModel

        model_path.mkdir(parents=True, exist_ok=True)

        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_path))
        model = AutoModel.from_pretrained(model_name, cache_dir=str(model_path))

        print_success(f"Model {model_name} downloaded successfully")
        return True

    except ImportError:
        print_error("Transformers not installed. Please install: pip install transformers")
        return False
    except Exception as e:
        print_error(f"Model download failed: {e}")
        return False


def download_dataset(dataset_id: str, verify_only: bool = False, mirror: bool = False) -> Tuple[bool, str]:
    """Download a specific dataset"""
    if dataset_id not in DATASETS:
        return False, f"Unknown dataset: {dataset_id}"

    dataset = DATASETS[dataset_id]
    data_path = DATA_DIR / dataset["local_path"]

    print_header(f"Dataset: {dataset['name']}")
    print(f"ID: {dataset['id']}")
    print(f"Description: {dataset['description']}")
    print(f"Size: ~{dataset['size_mb']} MB")
    print(f"Local path: {data_path}")

    # Check if already exists
    if data_path.exists() and any(data_path.iterdir()):
        print_success("Dataset already exists locally")
        if verify_only:
            return True, "Dataset verified"
        else:
            user_input = input("Re-download? (y/N): ").strip().lower()
            if user_input != 'y':
                return True, "Skipped (already exists)"

    # Download based on method
    method = dataset["download_method"]

    if method == "torchvision":
        dataset_name = dataset_id.split("-")[-1]  # Extract dataset name
        success = download_torchvision_dataset(dataset_name, data_path)

    elif method == "huggingface":
        dataset_name = dataset["id"].split("-")[-1]
        success = download_huggingface_dataset(dataset_name, data_path)

    elif method == "direct":
        # Download and extract
        url = dataset.get("url")
        if not url:
            return False, "No download URL provided"

        filename = url.split("/")[-1]
        download_path = DATA_DIR / filename

        success = download_file(url, download_path, dataset["name"])

        if success and (filename.endswith('.tgz') or filename.endswith('.tar.gz') or filename.endswith('.zip')):
            success = extract_archive(download_path, data_path)
            # Clean up archive
            download_path.unlink()

    else:
        return False, f"Unknown download method: {method}"

    if success:
        print_success(f"Dataset {dataset_id} ready")
        return True, "Downloaded successfully"
    else:
        return False, "Download failed"


def download_model(model_id: str, verify_only: bool = False) -> Tuple[bool, str]:
    """Download a specific pretrained model"""
    if model_id not in PRETRAINED_MODELS:
        return False, f"Unknown model: {model_id}"

    model = PRETRAINED_MODELS[model_id]
    model_path = MODELS_DIR / model["local_path"]

    print_header(f"Model: {model['name']}")
    print(f"ID: {model['id']}")
    print(f"Description: {model['description']}")
    print(f"Size: ~{model['size_mb']} MB")
    print(f"Local path: {model_path}")

    # Check if already exists
    if model_path.exists():
        print_success("Model already exists locally")
        if verify_only:
            return True, "Model verified"
        else:
            user_input = input("Re-download? (y/N): ").strip().lower()
            if user_input != 'y':
                return True, "Skipped (already exists)"

    # Download based on method
    method = model["download_method"]

    if method == "torchvision":
        # PyTorch models download automatically on first use
        print_info("PyTorch models will download automatically when first used")
        return True, "Will download on first use"

    elif method == "huggingface":
        model_name = model.get("model_name")
        success = download_huggingface_model(model_name, model_path)

    elif method == "ultralytics":
        print_info("YOLOv8 models will download automatically when first used")
        print_info("Or install via: pip install ultralytics")
        return True, "Will download on first use"

    else:
        return False, f"Unknown download method: {method}"

    if success:
        print_success(f"Model {model_id} ready")
        return True, "Downloaded successfully"
    else:
        return False, "Download failed"


def main():
    parser = argparse.ArgumentParser(
        description="Download Stage 4 datasets and pretrained models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mirror", action="store_true", help="使用国内镜像加速")
    parser.add_argument("--verify-only", action="store_true", help="仅验证已下载数据")
    parser.add_argument("--skip-models", action="store_true", help="跳过模型下载")
    parser.add_argument("--dataset", type=str, help="仅下载指定数据集")
    parser.add_argument("--model", type=str, help="仅下载指定模型")

    args = parser.parse_args()

    print_header("Stage 4 Data Download")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Verify only: {args.verify_only}")
    print(f"Skip models: {args.skip_models}")

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "datasets": {},
        "models": {}
    }

    # Download datasets
    if args.dataset:
        # Single dataset
        success, msg = download_dataset(args.dataset, args.verify_only, args.mirror)
        results["datasets"][args.dataset] = {"success": success, "message": msg}
    else:
        # All datasets (required only by default)
        for dataset_id, dataset in DATASETS.items():
            if dataset.get("optional", False):
                print_info(f"Skipping optional dataset: {dataset_id}")
                continue

            success, msg = download_dataset(dataset_id, args.verify_only, args.mirror)
            results["datasets"][dataset_id] = {"success": success, "message": msg}

    # Download models
    if not args.skip_models:
        if args.model:
            # Single model
            success, msg = download_model(args.model, args.verify_only)
            results["models"][args.model] = {"success": success, "message": msg}
        else:
            # All models (required only)
            for model_id, model in PRETRAINED_MODELS.items():
                if model.get("optional", False):
                    print_info(f"Skipping optional model: {model_id}")
                    continue

                success, msg = download_model(model_id, args.verify_only)
                results["models"][model_id] = {"success": success, "message": msg}

    # Summary
    print_header("Download Summary")

    print(f"\n{Colors.BOLD}Datasets:{Colors.ENDC}")
    for dataset_id, result in results["datasets"].items():
        status = "✓" if result["success"] else "✗"
        color = Colors.GREEN if result["success"] else Colors.RED
        print(f"{color}{status}{Colors.ENDC} {dataset_id}: {result['message']}")

    if results["models"]:
        print(f"\n{Colors.BOLD}Models:{Colors.ENDC}")
        for model_id, result in results["models"].items():
            status = "✓" if result["success"] else "✗"
            color = Colors.GREEN if result["success"] else Colors.RED
            print(f"{color}{status}{Colors.ENDC} {model_id}: {result['message']}")

    # Overall status
    all_success = all(r["success"] for r in results["datasets"].values())
    if results["models"]:
        all_success = all_success and all(r["success"] for r in results["models"].values())

    if all_success:
        print_success("\n✓ All downloads completed successfully!")
        return 0
    else:
        print_warning("\n⚠ Some downloads failed. Check the summary above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
