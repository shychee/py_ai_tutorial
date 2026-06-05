# 术语中英对照表 (Glossary)

本教程涵盖机器学习、深度学习、大模型等领域的常用术语。首次出现的技术术语将同时标注中英文,便于查阅和理解。

---

## A

### 激活函数 (Activation Function)
在神经网络中引入非线性变换的函数,如ReLU、Sigmoid、Tanh等。

### 注意力机制 (Attention Mechanism)
让模型专注于输入的重要部分的机制,是Transformer架构的核心组件。

### 自编码器 (Autoencoder)
一种无监督学习的神经网络,学习数据的压缩表示并重构原始输入。

---

## B

### 批处理 (Batch Processing)
同时处理多个样本以提高训练效率和稳定性的方法。

### 批标准化 (Batch Normalization, BN)
标准化每个批次的激活值,加速训练收敛并提高模型稳定性。

### BERT (Bidirectional Encoder Representations from Transformers)
双向Transformer编码器,是预训练语言模型的代表。

### 偏差-方差权衡 (Bias-Variance Tradeoff)
模型复杂度与泛化能力之间的平衡:简单模型高偏差(欠拟合),复杂模型高方差(过拟合)。

### 反向传播 (Backpropagation)
通过链式法则计算梯度,从输出层向输入层传播误差的算法。

---

## C

### 分类 (Classification)
预测离散类别标签的监督学习任务,如图像分类、文本分类。

### 聚类 (Clustering)
将相似数据分组的无监督学习方法,如K-means、DBSCAN、层次聚类。

### 卷积神经网络 (Convolutional Neural Network, CNN)
专门处理网格结构数据(如图像)的神经网络,使用卷积层提取空间特征。

### 交叉熵 (Cross-Entropy)
衡量两个概率分布差异的损失函数,常用于分类任务。

### 交叉验证 (Cross-Validation)
将数据分成多个子集,轮流作为验证集评估模型性能的技术。

---

## D

### 数据增强 (Data Augmentation)
通过变换(旋转、翻转、裁剪等)扩充训练数据,提高模型泛化能力。

### 决策树 (Decision Tree)
通过树状结构进行决策的模型,可用于分类和回归。

### 深度学习 (Deep Learning)
使用多层神经网络学习数据表示的机器学习方法。

### 降维 (Dimensionality Reduction)
减少特征数量的技术,如PCA、t-SNE、UMAP。

### Dropout
训练时随机丢弃部分神经元的正则化技术,防止过拟合。

---

## E

### 嵌入 (Embedding)
将离散对象(如单词、类别)映射到连续向量空间的表示方法。

### 集成学习 (Ensemble Learning)
组合多个模型提升预测性能的方法,如随机森林、XGBoost、投票法。

### Epoch
训练过程中完整遍历一次整个训练集的过程。

### 评估指标 (Evaluation Metrics)
衡量模型性能的指标,如准确率、精确率、召回率、F1分数、AUC等。

---

## F

### 特征工程 (Feature Engineering)
从原始数据中创建、选择和转换特征以提升模型性能的过程。

### 特征缩放 (Feature Scaling)
标准化特征范围的技术,如Min-Max归一化、Z-score标准化。

### 微调 (Fine-tuning)
在预训练模型基础上针对特定任务继续训练的迁移学习方法。

---

## G

### 生成对抗网络 (Generative Adversarial Network, GAN)
由生成器和判别器对抗训练的生成模型,可生成逼真的图像、文本等。

### GPT (Generative Pre-trained Transformer)
生成式预训练Transformer,是大语言模型的代表架构。

### 梯度 (Gradient)
损失函数对模型参数的导数,指示参数更新方向。

### 梯度下降 (Gradient Descent)
通过沿梯度负方向迭代更新参数以最小化损失函数的优化算法。

### 梯度消失/爆炸 (Gradient Vanishing/Exploding)
深层网络训练中梯度变得极小或极大的问题,影响模型收敛。

---

## H

### 超参数 (Hyperparameter)
模型训练前需要设定的参数,如学习率、批大小、隐藏层数等。

### 超参数调优 (Hyperparameter Tuning)
寻找最优超参数组合的过程,常用网格搜索、随机搜索、贝叶斯优化。

---

## I

### 推理 (Inference)
使用训练好的模型对新数据进行预测的过程。

### 迁移学习 (Transfer Learning)
将在一个任务上学到的知识应用到另一个相关任务的方法。

---

## K

### K近邻 (K-Nearest Neighbors, KNN)
基于最近邻样本进行分类或回归的简单算法。

### K-means
将数据分成K个簇的聚类算法。

---

## L

### 学习率 (Learning Rate)
控制参数更新步长的超参数,影响训练速度和稳定性。

### 线性回归 (Linear Regression)
建立特征与连续目标变量之间线性关系的回归模型。

### 逻辑回归 (Logistic Regression)
用于二分类的线性模型,通过Sigmoid函数输出概率。

### LoRA (Low-Rank Adaptation)
大模型的参数高效微调方法,通过低秩矩阵分解减少可训练参数。

### 损失函数 (Loss Function)
衡量模型预测与真实标签差异的函数,如MSE、交叉熵。

---

## M

### 机器学习 (Machine Learning)
让计算机从数据中学习模式而无需显式编程的技术。

### 平均绝对误差 (Mean Absolute Error, MAE)
回归任务中预测值与真实值绝对差的平均,衡量预测误差。

### 均方误差 (Mean Squared Error, MSE)
回归任务中预测值与真实值平方差的平均,对大误差更敏感。

### 模型 (Model)
从数据中学习的数学表示,用于预测或决策。

---

## N

### 神经网络 (Neural Network)
由多层神经元组成的模型,模拟生物神经系统的计算结构。

### 归一化 (Normalization)
将数据缩放到特定范围的技术,如[0, 1]或均值0、方差1。

### NumPy
Python科学计算的基础库,提供高效的多维数组操作。

---

## O

### 目标检测 (Object Detection)
检测图像中多个对象位置和类别的计算机视觉任务,如YOLO、Faster R-CNN。

### 一热编码 (One-Hot Encoding)
将类别变量转换为二进制向量的编码方式。

### 优化器 (Optimizer)
更新模型参数的算法,如SGD、Adam、AdamW、RMSprop。

### 过拟合 (Overfitting)
模型在训练集上表现好但泛化能力差的现象,通常由模型过于复杂引起。

---

## P

### Pandas
Python数据分析库,提供DataFrame结构和数据操作工具。

### 主成分分析 (Principal Component Analysis, PCA)
通过线性变换降维的无监督学习方法,保留最大方差。

### 精确率 (Precision)
分类任务中正例预测的准确度: TP / (TP + FP)。

### 预训练模型 (Pre-trained Model)
在大规模数据上训练好的模型,可用于迁移学习或微调。

### 提示工程 (Prompt Engineering)
设计有效的输入提示以引导大语言模型生成期望输出的技术。

### PyTorch
Meta开发的深度学习框架,以动态计算图和灵活性著称。

---

## R

### 随机森林 (Random Forest)
集成多个决策树的分类和回归模型,通过投票或平均提升性能。

### RAG (Retrieval-Augmented Generation)
检索增强生成,结合外部知识库和生成模型的技术,提升大模型的准确性。

### 召回率 (Recall)
分类任务中识别出的正例比例: TP / (TP + FN)。

### 循环神经网络 (Recurrent Neural Network, RNN)
处理序列数据的神经网络,通过循环连接保持时间依赖。

### 正则化 (Regularization)
防止过拟合的技术,如L1/L2正则化、Dropout、数据增强。

### 回归 (Regression)
预测连续数值的监督学习任务,如房价预测、股票预测。

### 强化学习 (Reinforcement Learning)
通过与环境交互学习最优策略的机器学习方法。

### ResNet (Residual Network)
使用残差连接解决梯度消失问题的深度卷积网络。

### ROC曲线 (Receiver Operating Characteristic Curve)
展示分类器在不同阈值下真正例率和假正例率关系的曲线。

---

## S

### Scikit-learn
Python机器学习库,提供丰富的传统机器学习算法和工具。

### 语义分割 (Semantic Segmentation)
为图像中每个像素分配类别标签的计算机视觉任务。

### 序列到序列 (Sequence-to-Sequence, Seq2Seq)
处理输入序列生成输出序列的模型,如机器翻译、文本摘要。

### 随机梯度下降 (Stochastic Gradient Descent, SGD)
每次使用单个或小批样本计算梯度的优化算法,比批量梯度下降更快。

### 监督学习 (Supervised Learning)
从标注数据中学习输入到输出映射的机器学习方法。

### 支持向量机 (Support Vector Machine, SVM)
寻找最优决策边界的分类算法,通过最大化间隔提升泛化能力。

---

## T

### TensorFlow
Google开发的深度学习框架,以静态计算图和生产部署能力著称。

### 训练集 (Training Set)
用于训练模型的数据子集。

### Transformer
基于自注意力机制的架构,是现代NLP和多模态模型的基础。

---

## U

### 欠拟合 (Underfitting)
模型过于简单,无法捕捉数据模式的现象。

### 无监督学习 (Unsupervised Learning)
从无标注数据中发现模式的机器学习方法,如聚类、降维。

---

## V

### 验证集 (Validation Set)
用于调整超参数和模型选择的数据子集,不参与训练。

### VGG (Visual Geometry Group)
使用小卷积核堆叠的经典卷积网络架构。

---

## W

### 权重 (Weight)
神经网络中连接神经元的可学习参数。

### 词嵌入 (Word Embedding)
将单词映射到连续向量空间的表示,如Word2Vec、GloVe。

---

## X

### XGBoost (Extreme Gradient Boosting)
基于梯度提升决策树的高效集成学习算法,在结构化数据竞赛中表现优异。

---

## Y

### YOLO (You Only Look Once)
实时目标检测算法,通过单次前向传播同时预测多个对象的位置和类别。

---

## Z

### 零样本学习 (Zero-Shot Learning)
模型在未见过特定任务训练样本的情况下进行预测的能力。

---

## 符号

### η (Eta)
学习率的常用符号。

### λ (Lambda)
正则化系数的常用符号。

### σ (Sigma)
Sigmoid函数或标准差的常用符号。

---

## 缩写速查

| 缩写 | 全称 | 中文 |
|------|------|------|
| AI | Artificial Intelligence | 人工智能 |
| AUC | Area Under Curve | 曲线下面积 |
| BERT | Bidirectional Encoder Representations from Transformers | 双向Transformer编码器 |
| BN | Batch Normalization | 批标准化 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| CV | Computer Vision | 计算机视觉 |
| DL | Deep Learning | 深度学习 |
| DNN | Deep Neural Network | 深度神经网络 |
| EDA | Exploratory Data Analysis | 探索性数据分析 |
| GAN | Generative Adversarial Network | 生成对抗网络 |
| GPT | Generative Pre-trained Transformer | 生成式预训练Transformer |
| GPU | Graphics Processing Unit | 图形处理单元 |
| KNN | K-Nearest Neighbors | K近邻 |
| LLM | Large Language Model | 大语言模型 |
| LoRA | Low-Rank Adaptation | 低秩适应 |
| LSTM | Long Short-Term Memory | 长短期记忆网络 |
| MAE | Mean Absolute Error | 平均绝对误差 |
| ML | Machine Learning | 机器学习 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| MSE | Mean Squared Error | 均方误差 |
| NLP | Natural Language Processing | 自然语言处理 |
| OCR | Optical Character Recognition | 光学字符识别 |
| PCA | Principal Component Analysis | 主成分分析 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| RMSE | Root Mean Squared Error | 均方根误差 |
| RNN | Recurrent Neural Network | 循环神经网络 |
| ROC | Receiver Operating Characteristic | 受试者工作特征 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| SVM | Support Vector Machine | 支持向量机 |
| TPU | Tensor Processing Unit | 张量处理单元 |

---

**提示**: 本术语表持续更新中。如发现遗漏或需要补充的术语,欢迎提Issue或PR。

**最后更新**: 2025-11-12
**维护者**: py_ai_tutorial团队
