# GitHub Pages 配置指南

## 当前状态

✅ 文档已构建完成（本地 site/ 目录）
✅ GitHub Actions 工作流已配置（.github/workflows/deploy-docs.yml）
✅ gh-pages 分支已存在
✅ 所有代码已推送到 GitHub

## 需要手动配置的步骤

由于 GitHub Pages 需要在仓库设置中启用,请按以下步骤操作:

### 1. 访问仓库设置

打开浏览器访问: https://github.com/shychee/py_ai_tutorial/settings/pages

### 2. 配置 GitHub Pages

在 "Build and deployment" 部分:

- **Source**: 选择 `Deploy from a branch`
- **Branch**: 选择 `gh-pages` 分支
- **Folder**: 选择 `/ (root)`

### 3. 保存并等待部署

点击 "Save" 后,GitHub 会自动开始部署。通常需要 1-3 分钟。

### 4. 验证部署

部署完成后,访问: https://shychee.github.io/py_ai_tutorial/

你应该能看到完整的文档网站。

## 本地预览文档

如果想在本地预览文档,可以使用以下方法:

### 方法 1: 使用便捷脚本（推荐）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行文档服务器
./scripts/docs/serve.sh
```

### 方法 2: 直接运行 mkdocs

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动开发服务器（支持热重载）
mkdocs serve
```

访问: http://localhost:8000

## 自动部署机制

每次推送到 `main` 分支时,GitHub Actions 会自动:

1. 安装依赖（mkdocs-material, mkdocs-jupyter, jieba 等）
2. 构建文档（mkdocs build）
3. 部署到 gh-pages 分支（mkdocs gh-deploy）

你可以在这里查看部署历史:
https://github.com/shychee/py_ai_tutorial/actions/workflows/deploy-docs.yml

## 故障排查

### 问题 1: 404 错误

**原因**: GitHub Pages 可能还未在仓库设置中启用

**解决方案**: 按照上面的步骤 1-3 配置 GitHub Pages

### 问题 2: 样式丢失或页面显示异常

**原因**: `site_url` 配置不正确

**解决方案**: 检查 mkdocs.yml 中的 `site_url` 是否为:
```yaml
site_url: https://shychee.github.io/py_ai_tutorial/
```

### 问题 3: 中文搜索不工作

**原因**: jieba 分词库未安装

**解决方案**:
```bash
source .venv/bin/activate
pip install jieba
```

### 问题 4: GitHub Actions 部署失败

**原因**: 可能是权限问题

**解决方案**:
1. 检查仓库设置 → Actions → General → Workflow permissions
2. 确保选择了 "Read and write permissions"

## 文档结构

```
docs/                       # 文档源文件（Markdown）
├── index.md               # 首页
├── quickstart.md          # 快速开始
├── prerequisites.md       # 先修要求
├── learning-path.md       # 学习路线
├── cross-platform/        # 跨平台配置指引
│   ├── setup-macos-intel.md
│   ├── setup-macos-arm64.md
│   ├── setup-linux.md
│   ├── setup-windows-native.md
│   ├── setup-windows-wsl2.md
│   ├── setup-cloud-gpu.md
│   └── troubleshooting.md
├── stage3/                # 阶段3：机器学习
├── stage4/                # 阶段4：深度学习
└── stage5/                # 阶段5：AIGC与大模型

site/                       # 构建后的静态网站（自动生成，不要提交）
```

## 相关文档

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [本项目文档工具说明](scripts/docs/README.md)

## 联系支持

如有问题,请:
- 查看 [scripts/docs/README.md](scripts/docs/README.md) 中的常见问题
- 提交 [GitHub Issue](https://github.com/shychee/py_ai_tutorial/issues)
- 发送邮件至 shychee96@gmail.com
