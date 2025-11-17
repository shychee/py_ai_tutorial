#!/usr/bin/env python3
"""
YOLOv11 PyTorch Training Script
基于YOLOv11的目标检测模型训练脚本

Usage:
    # CPU训练
    python train.py --config ../configs/cpu.yaml --data ../configs/dataset.yaml --epochs 100

    # GPU训练
    python train.py --config ../configs/gpu.yaml --data ../configs/dataset.yaml --epochs 50 --device 0

    # 迁移学习
    python train.py --weights yolov11n.pt --data ../configs/dataset.yaml --epochs 50

Author: AI Tutorial Team
Date: 2025-11-17
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 检查Ultralytics是否可用
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("✓ Ultralytics YOLO library detected")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠ Ultralytics not installed. Using simplified training mode.")
    print("  Install: pip install ultralytics")


class YOLOv11Trainer:
    """YOLOv11训练器"""

    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.output_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"YOLOv11 Training Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

    def _get_device(self):
        """获取计算设备"""
        if self.args.device == 'cpu':
            return torch.device('cpu')
        elif torch.cuda.is_available():
            device_id = int(self.args.device) if self.args.device.isdigit() else 0
            return torch.device(f'cuda:{device_id}')
        else:
            print("⚠ GPU not available, using CPU")
            return torch.device('cpu')

    def load_config(self):
        """加载配置文件"""
        # 加载数据配置
        with open(self.args.data, 'r') as f:
            self.data_config = yaml.safe_load(f)

        # 加载训练配置（如果提供）
        if self.args.config:
            with open(self.args.config, 'r') as f:
                self.train_config = yaml.safe_load(f)
        else:
            self.train_config = {}

        print("✓ Configuration loaded")
        print(f"  Dataset: {self.data_config.get('path', 'N/A')}")
        print(f"  Classes: {self.data_config.get('nc', 'N/A')}")

    def build_model(self):
        """构建模型"""
        if ULTRALYTICS_AVAILABLE:
            # 使用Ultralytics官方实现
            if self.args.weights and Path(self.args.weights).exists():
                print(f"✓ Loading pretrained weights: {self.args.weights}")
                self.model = YOLO(self.args.weights)
            else:
                print(f"✓ Creating new YOLOv11n model")
                self.model = YOLO('yolov11n.yaml')  # 创建新模型
        else:
            # 简化版本（仅供演示）
            print("⚠ Using simplified model (for demonstration)")
            self.model = self._create_simple_model()
            self.model.to(self.device)

    def _create_simple_model(self):
        """创建简化的演示模型"""
        class SimplifiedYOLO(nn.Module):
            def __init__(self, num_classes=80):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.head = nn.Conv2d(64, num_classes + 5, 1)  # 5 = x,y,w,h,conf

            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x

        return SimplifiedYOLO(num_classes=self.data_config.get('nc', 80))

    def train(self):
        """执行训练"""
        if ULTRALYTICS_AVAILABLE:
            # 使用Ultralytics训练
            results = self.model.train(
                data=self.args.data,
                epochs=self.args.epochs,
                imgsz=self.args.imgsz,
                batch=self.args.batch_size,
                device=self.args.device,
                workers=self.args.workers,
                project=str(self.output_dir.parent),
                name=self.output_dir.name,
                exist_ok=True,
                pretrained=bool(self.args.weights),
                optimizer=self.args.optimizer,
                lr0=self.args.lr0,
                lrf=self.args.lrf,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                warmup_epochs=self.args.warmup_epochs,
                patience=self.args.patience,
                save=True,
                save_period=self.args.save_period,
                val=True,
                plots=True,
                verbose=True
            )

            print("\n" + "="*60)
            print("Training Complete!")
            print("="*60)
            print(f"Best weights: {results.save_dir / 'weights' / 'best.pt'}")
            print(f"Last weights: {results.save_dir / 'weights' / 'last.pt'}")
            print(f"Results: {results.save_dir}")
            print("="*60 + "\n")

            return results
        else:
            # 简化训练循环（仅供演示）
            self._simplified_training()

    def _simplified_training(self):
        """简化的训练循环（演示用）"""
        print("\n" + "="*60)
        print("Running Simplified Training (Demo Mode)")
        print("="*60)
        print("Note: This is a simplified version for demonstration.")
        print("For production use, install: pip install ultralytics")
        print("="*60 + "\n")

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr0)
        criterion = nn.MSELoss()  # 简化的损失函数

        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss = 0.0

            # 模拟训练批次
            pbar = tqdm(range(10), desc=f'Epoch {epoch+1}/{self.args.epochs}')
            for batch_idx in pbar:
                # 模拟数据
                images = torch.randn(self.args.batch_size, 3, self.args.imgsz, self.args.imgsz).to(self.device)
                targets = torch.randn(self.args.batch_size, 64, 85).to(self.device)  # 简化

                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs.mean(), targets.mean())  # 简化损失

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / 10
            print(f'Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}')

            # 保存检查点
            if (epoch + 1) % self.args.save_period == 0:
                checkpoint_path = self.weights_dir / f'epoch_{epoch+1}.pt'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'  Saved checkpoint: {checkpoint_path}')

        # 保存最终权重
        final_path = self.weights_dir / 'last.pt'
        torch.save(self.model.state_dict(), final_path)
        print(f"\n✓ Training complete. Weights saved to: {final_path}")

    def validate(self):
        """验证模型"""
        if ULTRALYTICS_AVAILABLE:
            results = self.model.val()
            print("\nValidation Results:")
            print(f"  mAP@0.5: {results.box.map50:.4f}")
            print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
            return results
        else:
            print("⚠ Validation not available in simplified mode")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11 Training')

    # 基础参数
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件路径')
    parser.add_argument('--config', type=str, default=None, help='训练配置文件路径')
    parser.add_argument('--weights', type=str, default='', help='预训练权重路径')
    parser.add_argument('--output-dir', type=str, default='../outputs/yolov11_training', help='输出目录')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/0/1/...)')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')

    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器 (SGD/Adam/AdamW)')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率比例')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='预热轮数')

    # 其他参数
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--save-period', type=int, default=10, help='保存检查点间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建训练器
    trainer = YOLOv11Trainer(args)

    # 加载配置
    trainer.load_config()

    # 构建模型
    trainer.build_model()

    # 开始训练
    trainer.train()

    # 验证模型
    trainer.validate()


if __name__ == '__main__':
    main()
