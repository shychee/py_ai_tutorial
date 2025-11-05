# 项目模板 (Project Template)

这是一个标准的项目模板结构，用于创建新的教程项目。

## 目录结构

```
project-name/
├── README.md                   # 项目说明文档
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认配置
│   └── experiment.yaml        # 实验配置
├── src/                        # 源代码
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载
│   ├── model.py               # 模型定义
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数
├── notebooks/                  # Jupyter Notebook
│   └── analysis.ipynb         # 分析Notebook
├── tests/                      # 单元测试
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_model.py
├── outputs/                    # 输出目录
│   ├── models/                # 保存的模型
│   ├── figures/               # 可视化图表
│   ├── reports/               # 分析报告
│   └── logs/                  # 日志文件
├── requirements.txt            # Python依赖
└── .gitignore                 # Git忽略文件
```

## 使用方式

### 1. 创建新项目

```bash
# 复制模板
cp -r templates/project-template docs/stage3/projects/p01-new-project

# 重命名并修改README.md
```

### 2. 配置项目

编辑 `configs/default.yaml`:

```yaml
project:
  name: "项目名称"
  stage: "stage3"
  project_id: "P01"

data:
  path: "data/stage3/dataset.csv"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001

output:
  models_dir: "outputs/models"
  figures_dir: "outputs/figures"
  reports_dir: "outputs/reports"
```

### 3. 运行项目

```bash
# 训练模型
python src/train.py --config configs/default.yaml

# 评估模型
python src/evaluate.py --config configs/default.yaml --model outputs/models/best_model.pkl

# 运行Notebook
jupyter notebook notebooks/analysis.ipynb
```

## 代码质量标准

### PEP 8 规范

- 使用4个空格缩进
- 每行不超过88个字符（black默认）
- 导入顺序：标准库 → 第三方库 → 本地模块

### 类型注解

```python
from typing import List, Dict, Optional, Tuple

def load_data(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """加载数据集

    Args:
        file_path: 数据文件路径
        nrows: 读取的行数，None表示读取全部

    Returns:
        加载的DataFrame
    """
    return pd.read_csv(file_path, nrows=nrows)
```

### 文档字符串

使用Google风格的docstring:

```python
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: Dict[str, any]
) -> sklearn.base.BaseEstimator:
    """训练机器学习模型

    Args:
        X_train: 训练特征，shape (n_samples, n_features)
        y_train: 训练标签，shape (n_samples,)
        model_params: 模型超参数字典

    Returns:
        训练好的模型实例

    Raises:
        ValueError: 如果输入数据形状不匹配

    Example:
        >>> model = train_model(X_train, y_train, {"n_estimators": 100})
        >>> predictions = model.predict(X_test)
    """
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model
```

### 单元测试

```python
import pytest
import numpy as np
from src.data_loader import load_data

def test_load_data():
    """测试数据加载功能"""
    df = load_data("data/test.csv")
    assert not df.empty
    assert len(df.columns) > 0

def test_load_data_with_nrows():
    """测试限制行数的数据加载"""
    df = load_data("data/test.csv", nrows=100)
    assert len(df) == 100
```

## 评估标准

### 代码质量 (30%)

- [ ] 通过black格式化检查
- [ ] 通过ruff代码检查
- [ ] 通过mypy类型检查
- [ ] 单元测试覆盖率≥80%

### 功能完整性 (40%)

- [ ] 数据加载与预处理正确
- [ ] 模型训练收敛
- [ ] 评估指标落在预期范围
- [ ] 可视化图表清晰

### 文档完善度 (20%)

- [ ] README.md完整
- [ ] 代码注释充分
- [ ] 函数文档字符串完整
- [ ] 实验报告详细

### 可复现性 (10%)

- [ ] 设置随机种子
- [ ] 配置文件完整
- [ ] 依赖版本锁定
- [ ] 输出指标可复现

## 参考资源

- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [pytest Documentation](https://docs.pytest.org/)
