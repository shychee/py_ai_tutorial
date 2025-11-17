#!/usr/bin/env bash
# 启动 MkDocs 文档服务器
# 用法: ./scripts/docs/serve.sh

set -e

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告: 虚拟环境未激活"
    echo "请先运行: source .venv/bin/activate"
    exit 1
fi

# 检查 mkdocs 是否已安装
if ! command -v mkdocs &> /dev/null; then
    echo "📦 安装文档依赖..."
    pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-jupyter jieba
fi

# 启动文档服务器
echo "🚀 启动文档服务器..."
echo "📚 访问: http://localhost:8000"
echo "按 Ctrl+C 停止服务器"
echo ""
mkdocs serve
