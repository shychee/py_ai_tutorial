# 文档工具脚本

本目录包含用于构建和部署 MkDocs 文档的工具脚本。

## 📋 脚本列表

### serve.sh - 本地文档服务器

启动本地 MkDocs 开发服务器，支持热重载。

**用法**:
```bash
# 方式 1: 直接运行脚本
./scripts/docs/serve.sh

# 方式 2: 手动启动
source .venv/bin/activate
mkdocs serve
```

**访问**: http://localhost:8000

**特性**:
- 自动检测虚拟环境
- 自动安装缺失的依赖
- 支持实时预览（修改文件后自动刷新）

## 🚀 快速开始

### 1. 安装文档依赖

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装文档依赖
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-jupyter jieba

# 或使用 uv
uv pip install -e ".[docs]"
```

### 2. 本地预览文档

```bash
# 方式 1: 使用脚本（推荐）
./scripts/docs/serve.sh

# 方式 2: 直接运行 mkdocs
mkdocs serve

# 方式 3: 指定端口
mkdocs serve --dev-addr=localhost:8001
```

### 3. 构建静态站点

```bash
# 构建到 site/ 目录
mkdocs build

# 构建并清理旧文件
mkdocs build --clean

# 查看构建结果
ls -la site/
```

### 4. 部署到 GitHub Pages

```bash
# 自动构建并部署到 gh-pages 分支
mkdocs gh-deploy

# 带清理和详细输出
mkdocs gh-deploy --clean --verbose
```

**注意**: 推送到 `main` 分支会自动触发 GitHub Actions 部署，无需手动运行此命令。

## 🔧 常见问题

### Q: 提示 "plugin not installed"

A: 安装缺失的插件：
```bash
pip install mkdocs-git-revision-date-localized-plugin mkdocs-jupyter
```

### Q: 中文搜索不工作

A: 安装 jieba 分词库：
```bash
pip install jieba
```

### Q: 依赖冲突 (PyTorch/TensorFlow)

A: 不要使用 `uv run pip install`，而是在激活的虚拟环境中直接安装：
```bash
source .venv/bin/activate
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-jupyter jieba
```

### Q: 如何添加新页面？

A:
1. 在 `docs/` 目录下创建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加导航项
3. 保存后自动重载（如果运行了 `mkdocs serve`）

### Q: 如何预览 Jupyter Notebooks？

A:
1. 将 `.ipynb` 文件放在 `notebooks/` 目录
2. 在 `mkdocs.yml` 的 `nav` 中引用
3. MkDocs 会自动渲染（需要 mkdocs-jupyter 插件）

## 📚 相关文档

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)

## 🆘 获取帮助

如有问题，请：
- 查看 [mkdocs.yml](../../mkdocs.yml) 配置文件
- 提交 [GitHub Issue](https://github.com/shychee/py_ai_tutorial/issues)
- 发送邮件至 shychee96@gmail.com
