# 学习前提 (Prerequisites)

本教程面向**有3-5年Python后端开发经验的工程师**,旨在系统学习AI/ML技能并应用到实际业务场景。在开始学习前,请确认你满足以下基础要求。

---

## 📋 必备技能 (Required)

### 1. Python编程基础

**要求水平**: 熟练使用Python进行日常开发

**必备知识**:
- ✅ Python基本语法(变量、循环、条件、函数)
- ✅ 面向对象编程(类、继承、多态)
- ✅ 常用数据结构(列表、字典、集合、元组)
- ✅ 文件I/O操作
- ✅ 异常处理
- ✅ 模块和包管理
- ✅ 虚拟环境使用(venv/conda)
- ✅ pip包安装

**推荐版本**: Python 3.9+ (本教程使用3.11+)

**自测问题**:
- 能否快速写出读取CSV文件并处理每一行的代码?
- 是否理解装饰器、生成器、上下文管理器的用法?
- 能否独立调试Python程序并定位错误?

**如果不满足**: 建议先学习[《Python编程:从入门到实践》](https://www.ituring.com.cn/book/2784)或[Real Python教程](https://realpython.com/)

---

### 2. 数学基础

**要求水平**: 高中数学 + 基础线性代数/概率统计

**必备知识**:
- ✅ 基础代数(方程、函数、对数、指数)
- ✅ 向量和矩阵概念(加法、乘法、转置)
- ✅ 概率基础(概率分布、期望、方差)
- ✅ 基础统计(均值、中位数、标准差)
- ✅ 导数概念(梯度下降需要理解导数含义)

**推荐资源**:
- [3Blue1Brown - 线性代数的本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (强烈推荐!)
- [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer) (统计学直观讲解)

**自测问题**:
- 是否理解矩阵乘法的含义?
- 能否解释什么是正态分布?
- 是否理解导数表示函数的变化率?

**如果不满足**: 
- **不用担心!** 本教程会通过Python代码和可视化帮你理解数学概念
- 建议边学边补,重点理解直觉而非推导
- 可以先学习阶段3(传统ML),再根据需要补充数学

---

### 3. 命令行基础

**要求水平**: 能够使用终端执行基本操作

**必备知识**:
- ✅ 基本命令:`cd`, `ls`, `mkdir`, `rm`, `cp`, `mv`
- ✅ 查看文件内容:`cat`, `less`, `head`, `tail`
- ✅ 进程管理:`ps`, `kill`, `top`
- ✅ 环境变量设置:`export`, `PATH`
- ✅ 文本编辑器(vim/nano)基本使用

**自测问题**:
- 能否快速切换目录并创建/删除文件?
- 是否知道如何查看当前Python版本?
- 能否激活Python虚拟环境?

**如果不满足**: 建议花1-2小时学习[Linux命令行基础教程](https://www.runoob.com/linux/linux-command-manual.html)

---

### 4. Git版本控制

**要求水平**: 能够克隆仓库、提交代码、查看历史

**必备知识**:
- ✅ `git clone`, `git pull`, `git push`
- ✅ `git add`, `git commit`, `git status`
- ✅ `git log`, `git diff`
- ✅ 基本分支操作:`git branch`, `git checkout`

**自测问题**:
- 能否从GitHub克隆本教程仓库?
- 是否知道如何查看提交历史?
- 能否回退到之前的版本?

**如果不满足**: 建议学习[Pro Git书籍](https://git-scm.com/book/zh/v2)(前3章即可)

---

## 🎯 推荐技能 (Optional but Helpful)

以下技能不是必须的,但会让你的学习更顺畅:

### 1. Jupyter Notebook使用经验
- 了解Notebook单元格的运行方式
- 熟悉魔法命令(`%timeit`, `%matplotlib inline`)
- **如不熟悉**: 第一个教程会介绍,无需提前学习

### 2. 数据分析经验
- 使用过Pandas处理表格数据
- 使用过Matplotlib/Seaborn绘图
- **如不熟悉**: 阶段3会从头教起

### 3. SQL基础
- 了解SELECT、WHERE、JOIN等基本查询
- **如不熟悉**: 不影响学习,部分数据处理概念与SQL类似

### 4. Web开发基础
- 了解HTTP协议、RESTful API
- **如不熟悉**: 不影响ML学习,但有助于理解阶段5的API集成

### 5. Docker基础
- 了解容器概念和基本命令
- **如不熟悉**: 不影响学习,本教程提供原生环境配置方案

---

## 💻 硬件要求

### 阶段3(传统机器学习) - 最低要求
- **CPU**: 双核2.0GHz+ (推荐4核+)
- **内存**: 8GB RAM (推荐16GB)
- **存储**: 20GB可用空间
- **GPU**: 不需要
- **操作系统**: macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10/11

### 阶段4(深度学习) - 推荐配置
- **CPU**: 4核3.0GHz+
- **内存**: 16GB RAM (推荐32GB)
- **存储**: 50GB可用空间
- **GPU**: 可选,但强烈推荐
  - NVIDIA GPU (GTX 1060 6GB+, RTX 3060+, 或云端GPU)
  - 8GB+ 显存(推荐)
  - CUDA 11.8+支持
- **操作系统**: Linux首选,macOS/Windows可用

### 阶段5(大模型) - 云端推荐
- **本地**: 运行推理和API调用即可(CPU足够)
- **云端**: 推荐使用Google Colab / Kaggle / AWS SageMaker
- **GPU**: T4/V100级别(云端免费GPU足够)

**💡 提示**: 
- 阶段3可在任何笔记本电脑上完成,无需GPU
- 阶段4提供CPU和GPU两套代码,CPU可运行但较慢
- 阶段5推荐使用云端环境,本地无GPU也能学习

---

## 📚 外部资源链接

### Python学习资源
- [Python官方文档](https://docs.python.org/zh-cn/3/)
- [Real Python](https://realpython.com/) - 高质量Python教程
- [Python Tutor](https://pythontutor.com/) - 可视化代码执行

### 数学与统计
- [Khan Academy - 线性代数](https://www.khanacademy.org/math/linear-algebra)
- [3Blue1Brown - 数学之美](https://www.3blue1brown.com/)
- [Seeing Theory](https://seeing-theory.brown.edu/) - 概率统计可视化

### 机器学习入门
- [机器学习速成课程](https://developers.google.com/machine-learning/crash-course?hl=zh-cn) - Google官方
- [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning) - 经典入门课
- [Scikit-learn文档](https://scikit-learn.org/stable/) - 必备参考文档

### 深度学习
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [TensorFlow指南](https://www.tensorflow.org/tutorials?hl=zh-cn)
- [Deep Learning Book](https://www.deeplearningbook.org/) - Ian Goodfellow著

### 数据集资源
- [Kaggle Datasets](https://www.kaggle.com/datasets) - 丰富的公开数据集
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) - 经典数据集
- [HuggingFace Datasets](https://huggingface.co/datasets) - NLP和多模态数据

### 社区与论坛
- [Stack Overflow](https://stackoverflow.com/questions/tagged/python) - 技术问答
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - ML社区
- [GitHub](https://github.com/) - 开源项目和代码示例

---

## 🚦 自我评估

在开始学习前,请诚实回答以下问题:

| 问题 | 是 | 否 |
|------|----|----|
| 我能独立编写和调试Python脚本 | ☐ | ☐ |
| 我理解函数、类、模块的概念 | ☐ | ☐ |
| 我熟悉命令行基本操作 | ☐ | ☐ |
| 我能克隆Git仓库并查看文件 | ☐ | ☐ |
| 我理解向量和矩阵的基本概念 | ☐ | ☐ |
| 我知道什么是正态分布 | ☐ | ☐ |
| 我能安装Python包并创建虚拟环境 | ☐ | ☐ |

**如果有6个或更多"是"**: ✅ 你已经准备好开始学习了!

**如果有3-5个"是"**: ⚠️ 建议先补充不足的部分,或边学边补

**如果少于3个"是"**: ❌ 建议先系统学习Python和基础数学,再回来学习AI

---

## 🎯 开始你的学习之旅

确认前提条件后,按以下步骤开始:

1. **环境配置**: 根据你的操作系统,参考[跨平台配置指引](./cross-platform/)
   - [macOS (Intel)](./cross-platform/setup-macos-intel.md)
   - [macOS (Apple Silicon)](./cross-platform/setup-macos-arm64.md)
   - [Linux](./cross-platform/setup-linux.md)
   - [Windows原生](./cross-platform/setup-windows-native.md)
   - [WSL2](./cross-platform/setup-windows-wsl2.md) (推荐Windows用户)
   - [云端GPU](./cross-platform/setup-cloud-gpu.md)

2. **查看学习路径**: 阅读[学习路线图](./learning-path.md)了解完整学习计划

3. **快速开始**: 运行[快速入门Notebook](../notebooks/stage3/00-quick-start.ipynb)体验机器学习

4. **深入学习**: 从[阶段3第一个模块](./stage3/01-scientific-computing/)开始系统学习

---

**💪 记住**: 学习AI不需要数学博士学位,关键是动手实践和持续学习。祝你学习愉快!

---

**最后更新**: 2025-11-12
**维护者**: py_ai_tutorial团队
