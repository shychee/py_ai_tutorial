# 项目P02: 基于YOLOv11的视频实时检测系统

**阶段**: Stage 4 - 深度学习
**难度**: ⭐⭐⭐⭐ 中高等
**预计时间**: 8-12 小时
**框架**: PyTorch + TensorFlow (双框架实现)

---

## 📋 项目概述

本项目实现一个基于 YOLOv11 的实时视频目标检测系统，支持从摄像头、视频文件或图像进行目标检测。项目提供 **PyTorch** 和 **TensorFlow** 两种框架的完整实现，便于学习和对比两种框架的差异。

### 核心功能

- ✅ 实时视频流检测（摄像头/视频文件）
- ✅ 图像批量检测
- ✅ 自定义数据集训练
- ✅ 预训练模型微调
- ✅ 检测结果可视化（边界框、标签、置信度）
- ✅ 性能优化（FP16混合精度、批处理）
- ✅ 导出为 ONNX/TFLite（边缘设备部署）

### 学习目标

完成本项目后，你将能够：

1. ✅ 理解 YOLO 系列目标检测算法的核心原理
2. ✅ 掌握 PyTorch 和 TensorFlow 的目标检测实现差异
3. ✅ 能够使用预训练模型进行迁移学习
4. ✅ 掌握实时视频处理与性能优化技巧
5. ✅ 能够将模型部署到边缘设备（Jetson/RaspberryPi）
6. ✅ 理解目标检测的评估指标（mAP、Precision、Recall）

---

## 🎯 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **智能安防** | 入侵检测、异常行为识别 | 监控摄像头实时报警 |
| **自动驾驶** | 行人、车辆、交通标志检测 | 辅助驾驶系统 |
| **工业质检** | 产品缺陷检测 | 生产线自动化检测 |
| **零售分析** | 客流统计、货架分析 | 智能零售系统 |
| **医疗影像** | 病灶检测 | 辅助诊断系统 |

---

## 🏗️ 项目结构

```
p02-yolov11-realtime/
├── README.md                    # 本文档
├── configs/                     # 配置文件
│   ├── cpu.yaml                # CPU运行配置
│   ├── gpu.yaml                # GPU运行配置
│   └── dataset.yaml            # 数据集配置
│
├── pytorch/                     # PyTorch实现
│   ├── train.py                # 训练脚本
│   ├── detect.py               # 检测脚本
│   ├── export.py               # 模型导出
│   ├── models/                 # 模型定义
│   │   ├── yolov11.py         # YOLOv11架构
│   │   └── blocks.py          # 基础模块
│   ├── utils/                  # 工具函数
│   │   ├── dataset.py         # 数据加载
│   │   ├── metrics.py         # 评估指标
│   │   └── visualization.py   # 可视化工具
│   └── requirements.txt        # PyTorch依赖
│
├── tensorflow/                  # TensorFlow实现
│   ├── train.py                # 训练脚本
│   ├── detect.py               # 检测脚本
│   ├── export.py               # 模型导出
│   ├── models/                 # 模型定义
│   │   ├── yolov11.py         # YOLOv11架构
│   │   └── blocks.py          # 基础模块
│   ├── utils/                  # 工具函数
│   │   ├── dataset.py         # 数据加载
│   │   ├── metrics.py         # 评估指标
│   │   └── visualization.py   # 可视化工具
│   └── requirements.txt        # TensorFlow依赖
│
├── notebooks/                   # Jupyter Notebooks
│   ├── 01-pytorch-demo.ipynb   # PyTorch示例
│   ├── 02-tensorflow-demo.ipynb# TensorFlow示例
│   └── 03-comparison.ipynb     # 框架对比分析
│
├── data/                        # 数据目录
│   ├── images/                 # 图像文件
│   ├── labels/                 # 标注文件（YOLO格式）
│   ├── videos/                 # 视频文件
│   └── coco/                   # COCO数据集（可选）
│
├── outputs/                     # 输出目录
│   ├── weights/                # 训练权重
│   ├── predictions/            # 检测结果
│   └── logs/                   # 训练日志
│
└── tests/                       # 单元测试
    ├── test_pytorch.py
    └── test_tensorflow.py
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.8+ | 3.10+ |
| **CPU** | 4核 | 8核+ |
| **内存** | 8GB | 16GB+ |
| **GPU** | 无（可运行） | NVIDIA GTX 1060 6GB+ |
| **硬盘** | 5GB | 20GB+ |

### 安装依赖

#### PyTorch 版本

```bash
cd pytorch/

# 基础依赖
pip install -r requirements.txt

# 或手动安装
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install pyyaml
pip install tqdm
pip install matplotlib seaborn
```

#### TensorFlow 版本

```bash
cd tensorflow/

# 基础依赖
pip install -r requirements.txt

# 或手动安装
pip install tensorflow>=2.13
pip install opencv-python
pip install pillow
pip install pyyaml
pip install tqdm
pip install matplotlib seaborn
```

### 快速推理（使用预训练模型）

#### PyTorch

```bash
cd pytorch/

# 检测单张图像
python detect.py --source ../data/images/bus.jpg \
                 --weights yolov11n.pt \
                 --conf 0.25 \
                 --save

# 检测视频
python detect.py --source ../data/videos/traffic.mp4 \
                 --weights yolov11n.pt

# 实时摄像头检测
python detect.py --source 0 \
                 --weights yolov11n.pt
```

#### TensorFlow

```bash
cd tensorflow/

# 检测单张图像
python detect.py --source ../data/images/bus.jpg \
                 --weights yolov11n.h5 \
                 --conf 0.25 \
                 --save

# 检测视频
python detect.py --source ../data/videos/traffic.mp4 \
                 --weights yolov11n.h5
```

---

## 📚 详细教程

### 1. 数据准备

#### 1.1 使用COCO数据集（推荐）

```bash
# 下载COCO数据集（约18GB）
bash scripts/download_coco.sh

# 或使用样本数据集（约500MB）
python scripts/download_coco_sample.py
```

**COCO数据集结构**:
```
data/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

#### 1.2 自定义数据集

**步骤1**: 准备图像和标注

```
data/custom/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

**步骤2**: YOLO标注格式

每个 `.txt` 文件对应一张图像，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有值归一化到 [0, 1]。

**示例** (`img1.txt`):
```
0 0.5 0.5 0.3 0.4    # 类别0，中心(0.5, 0.5)，宽0.3，高0.4
1 0.7 0.3 0.2 0.2    # 类别1
```

**步骤3**: 创建数据集配置 (`configs/dataset.yaml`)

```yaml
# 数据集路径
path: ./data/custom
train: images/train
val: images/val

# 类别数量
nc: 80

# 类别名称
names:
  0: person
  1: bicycle
  2: car
  # ... 更多类别
```

#### 1.3 标注工具推荐

- **LabelImg**: https://github.com/heartexlabs/labelImg (支持YOLO格式)
- **CVAT**: https://cvat.org/ (在线标注平台)
- **Roboflow**: https://roboflow.com/ (自动转换格式)

---

### 2. 模型训练

#### 2.1 PyTorch训练

```bash
cd pytorch/

# 从头训练（小模型，适合CPU）
python train.py \
    --config ../configs/cpu.yaml \
    --data ../configs/dataset.yaml \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --weights '' \
    --device cpu

# 迁移学习（推荐，使用GPU）
python train.py \
    --config ../configs/gpu.yaml \
    --data ../configs/dataset.yaml \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 640 \
    --weights yolov11n.pt \
    --device 0
```

**训练参数说明**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 100-300 |
| `--batch-size` | 批次大小 | CPU: 8-16, GPU: 32-64 |
| `--imgsz` | 输入图像尺寸 | 640 (标准), 1280 (高精度) |
| `--weights` | 预训练权重 | yolov11n.pt (Nano), yolov11s.pt (Small) |
| `--device` | 计算设备 | cpu / 0 (GPU 0) |

#### 2.2 TensorFlow训练

```bash
cd tensorflow/

# 迁移学习
python train.py \
    --config ../configs/gpu.yaml \
    --data ../configs/dataset.yaml \
    --epochs 50 \
    --batch-size 32 \
    --imgsz 640 \
    --weights yolov11n.h5
```

#### 2.3 训练监控

**TensorBoard可视化**:

```bash
# 启动TensorBoard
tensorboard --logdir ../outputs/logs

# 访问 http://localhost:6006
```

**关键指标**:
- **Train Loss**: 训练损失（应持续下降）
- **Val Loss**: 验证损失（不应持续上升）
- **mAP@0.5**: 平均精度（越高越好，>0.5为良好）
- **Precision**: 精确率
- **Recall**: 召回率

---

### 3. 模型评估

#### 3.1 计算mAP

```bash
# PyTorch
python pytorch/evaluate.py \
    --weights ../outputs/weights/best.pt \
    --data ../configs/dataset.yaml \
    --imgsz 640

# TensorFlow
python tensorflow/evaluate.py \
    --weights ../outputs/weights/best.h5 \
    --data ../configs/dataset.yaml \
    --imgsz 640
```

**输出示例**:
```
Class    Images  Targets  Precision  Recall  mAP@0.5  mAP@0.5:0.95
all       5000    36335      0.725   0.681    0.731        0.562
person    5000    10777      0.842   0.789    0.856        0.672
car       5000     4852      0.731   0.698    0.748        0.578
...
```

#### 3.2 混淆矩阵

```bash
python pytorch/utils/metrics.py \
    --weights ../outputs/weights/best.pt \
    --data ../configs/dataset.yaml \
    --plot-confusion-matrix
```

---

### 4. 实时检测

#### 4.1 图像检测

```bash
# 单张图像
python pytorch/detect.py \
    --source ../data/images/sample.jpg \
    --weights ../outputs/weights/best.pt \
    --save \
    --conf 0.25

# 批量图像
python pytorch/detect.py \
    --source ../data/images/ \
    --weights ../outputs/weights/best.pt \
    --save-txt  # 保存检测结果为txt
```

#### 4.2 视频检测

```bash
python pytorch/detect.py \
    --source ../data/videos/traffic.mp4 \
    --weights ../outputs/weights/best.pt \
    --save \
    --view-img  # 实时预览
```

#### 4.3 摄像头实时检测

```bash
# 默认摄像头
python pytorch/detect.py \
    --source 0 \
    --weights ../outputs/weights/best.pt \
    --view-img

# 指定摄像头（如USB摄像头）
python pytorch/detect.py \
    --source 1 \
    --weights ../outputs/weights/best.pt
```

#### 4.4 RTSP流检测（IP摄像头）

```bash
python pytorch/detect.py \
    --source rtsp://username:password@192.168.1.100:554/stream \
    --weights ../outputs/weights/best.pt \
    --view-img
```

---

### 5. 模型导出与部署

#### 5.1 导出为ONNX（跨平台推理）

```bash
# PyTorch
python pytorch/export.py \
    --weights ../outputs/weights/best.pt \
    --include onnx \
    --simplify

# TensorFlow
python tensorflow/export.py \
    --weights ../outputs/weights/best.h5 \
    --include onnx
```

**ONNX优势**:
- ✅ 跨框架兼容（PyTorch → TensorFlow → ONNX Runtime）
- ✅ 硬件加速（TensorRT、OpenVINO）
- ✅ 部署灵活（服务器、边缘设备、浏览器）

#### 5.2 导出为TFLite（移动端/边缘设备）

```bash
python tensorflow/export.py \
    --weights ../outputs/weights/best.h5 \
    --include tflite \
    --int8  # INT8量化，提升速度
```

**TFLite部署场景**:
- Android应用（Kotlin/Java）
- iOS应用（Swift）
- Raspberry Pi
- Coral TPU

#### 5.3 TensorRT加速（NVIDIA GPU）

```bash
# 转换为TensorRT引擎
python pytorch/export.py \
    --weights ../outputs/weights/best.pt \
    --include engine \
    --device 0 \
    --half  # FP16混合精度

# TensorRT推理
python pytorch/detect.py \
    --source ../data/videos/traffic.mp4 \
    --weights ../outputs/weights/best.engine \
    --device 0
```

---

## 🔬 进阶主题

### 6.1 数据增强

**内置增强**（train.py自动应用）:
- Mosaic拼接
- MixUp混合
- 随机缩放
- 随机翻转
- 色彩抖动

**自定义增强** (`pytorch/utils/augmentation.py`):

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussianBlur(p=0.3),
    A.RandomRotate90(p=0.5),
], bbox_params=A.BboxParams(format='yolo'))
```

### 6.2 超参数调优

**关键超参数**:

```yaml
# configs/hyperparams.yaml
lr0: 0.01              # 初始学习率
lrf: 0.01              # 最终学习率（lr0 * lrf）
momentum: 0.937        # SGD动量
weight_decay: 0.0005   # 权重衰减（L2正则化）
warmup_epochs: 3       # 预热轮数
box: 7.5               # 边界框损失权重
cls: 0.5               # 分类损失权重
dfl: 1.5               # DFL损失权重
```

**自动调优**（使用Optuna）:

```bash
python pytorch/tune.py \
    --data ../configs/dataset.yaml \
    --epochs 50 \
    --trials 100
```

### 6.3 多GPU训练

```bash
# PyTorch DistributedDataParallel
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    pytorch/train.py \
    --data ../configs/dataset.yaml \
    --batch-size 128 \
    --device 0,1,2,3

# TensorFlow MirroredStrategy
python tensorflow/train.py \
    --data ../configs/dataset.yaml \
    --batch-size 128 \
    --strategy mirrored
```

### 6.4 小目标检测优化

```yaml
# configs/small_object.yaml
# 提高输入分辨率
imgsz: 1280

# 使用更多尺度
scales: [0.5, 0.75, 1.0, 1.25, 1.5]

# 增加小目标数据增强
mosaic: 1.0
mixup: 0.5
copy_paste: 0.3
```

---

## 📊 性能基准

### 模型对比（COCO val2017）

| 模型 | 大小 | mAP@0.5 | 速度(CPU) | 速度(GPU) | 参数量 |
|------|------|---------|----------|----------|--------|
| YOLOv11n | 3.2MB | 37.3% | 45ms | 1.2ms | 2.6M |
| YOLOv11s | 9.4MB | 44.9% | 78ms | 1.8ms | 9.4M |
| YOLOv11m | 20.1MB | 49.7% | 125ms | 2.5ms | 20.1M |
| YOLOv11l | 25.3MB | 52.2% | 156ms | 3.1ms | 25.3M |
| YOLOv11x | 56.9MB | 54.7% | 234ms | 4.3ms | 56.9M |

**测试环境**: Intel i7-12700K, NVIDIA RTX 3090, Batch Size=1

### 框架对比

| 指标 | PyTorch | TensorFlow | 备注 |
|------|---------|-----------|------|
| **训练速度** | ⚡⚡⚡⚡ | ⚡⚡⚡ | PyTorch略快 |
| **推理速度** | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ | TFLite移动端更快 |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PyTorch API更直观 |
| **部署** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | TF部署生态更完善 |
| **社区** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PyTorch学术界主流 |

---

## 🐛 故障排查

### Q1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 1. 减小批次大小
--batch-size 8

# 2. 减小输入尺寸
--imgsz 416

# 3. 使用梯度累积
--accumulate 4

# 4. 使用混合精度训练
--amp
```

### Q2: 检测速度慢

**解决方案**:
```bash
# 1. 使用更小的模型
--weights yolov11n.pt

# 2. 降低输入分辨率
--imgsz 320

# 3. 使用TensorRT加速
python pytorch/export.py --include engine --half

# 4. 批量处理（视频）
--batch-size 8
```

### Q3: 训练不收敛

**解决方案**:
1. 检查数据标注是否正确
2. 增加训练轮数: `--epochs 300`
3. 调整学习率: `--lr0 0.001`
4. 使用预训练权重: `--weights yolov11n.pt`
5. 检查数据增强是否过度

### Q4: mAP很低

**原因分析**:
- 数据集质量差（标注错误）
- 训练不充分（轮数太少）
- 类别不平衡
- 超参数不合适

**解决方案**:
```bash
# 1. 增加训练轮数
--epochs 300

# 2. 使用类别权重平衡
--cls-weight auto

# 3. 数据增强
--augment

# 4. 超参数调优
python pytorch/tune.py --trials 50
```

---

## 📖 参考资料

### 必读

- [YOLOv11 官方文档](https://docs.ultralytics.com/)
- [COCO数据集](https://cocodataset.org/)
- [目标检测评估指标](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

### 推荐阅读

- [YOLOv1 论文](https://arxiv.org/abs/1506.02640)
- [YOLOv11 改进点](https://github.com/ultralytics/ultralytics)
- [目标检测综述](https://arxiv.org/abs/1905.05055)

### 相关教程

- [PyTorch目标检测教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [TensorFlow目标检测API](https://github.com/tensorflow/models/tree/master/research/object_detection)

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 待完成功能

- [ ] 多类别NMS优化
- [ ] 实例分割（Mask YOLO）
- [ ] 3D目标检测
- [ ] 视频目标追踪（DeepSORT）
- [ ] 模型量化（INT8）
- [ ] Web前端界面（Streamlit）

---

## 📝 更新日志

### v1.0.0 (2025-11-17)
- ✅ 初始版本发布
- ✅ PyTorch实现
- ✅ TensorFlow实现
- ✅ CPU/GPU配置
- ✅ 完整文档

---

## 📄 许可证

本项目采用 MIT 许可证，仅供学习使用。

YOLO算法遵循 [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。

---

**上一个项目**: [P01: 工业视觉检测](../p01-industrial-vision/README.md)
**下一个项目**: [P03: OCR票据识别](../p03-ocr/README.md)
**返回**: [Stage 4 目录](../../README.md)
