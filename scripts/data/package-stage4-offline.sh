#!/bin/bash
#
# Package Stage 4 Offline Data
# 打包 Stage 4 离线数据与模型权重
#
# Usage:
#   bash scripts/data/package-stage4-offline.sh [--data-only] [--models-only]
#
# Options:
#   --data-only    仅打包数据集
#   --models-only  仅打包模型
#   --output DIR   指定输出目录（默认: offline/）
#
# Output:
#   offline/stage4-data.tar.gz    (~2GB, 包含 MNIST/CIFAR-10/IMDB)
#   offline/stage4-models.tar.gz  (~500MB, 包含 ResNet-50)
#

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Directories
DATA_DIR="$PROJECT_ROOT/data/stage4"
MODELS_DIR="$PROJECT_ROOT/data/models"
OUTPUT_DIR="$PROJECT_ROOT/offline"

# Parse arguments
PACKAGE_DATA=true
PACKAGE_MODELS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            PACKAGE_DATA=true
            PACKAGE_MODELS=false
            shift
            ;;
        --models-only)
            PACKAGE_DATA=false
            PACKAGE_MODELS=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
print_header() {
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Main
print_header "Stage 4 Offline Data Packaging"

echo "Project root: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Package datasets
if [ "$PACKAGE_DATA" = true ]; then
    print_header "Packaging Datasets"

    # Check if data exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        print_info "Run: python scripts/data/download-stage4.py"
        exit 1
    fi

    # Create temporary directory for packaging
    TMP_DIR=$(mktemp -d)
    mkdir -p "$TMP_DIR/stage4"

    # Copy datasets
    print_info "Copying datasets to temporary directory..."

    # MNIST
    if [ -d "$DATA_DIR/mnist" ]; then
        print_info "  - MNIST (~11MB)"
        cp -r "$DATA_DIR/mnist" "$TMP_DIR/stage4/"
    else
        print_warning "  - MNIST not found, skipping"
    fi

    # CIFAR-10
    if [ -d "$DATA_DIR/cifar10" ]; then
        print_info "  - CIFAR-10 (~170MB)"
        cp -r "$DATA_DIR/cifar10" "$TMP_DIR/stage4/"
    else
        print_warning "  - CIFAR-10 not found, skipping"
    fi

    # IMDB
    if [ -d "$DATA_DIR/imdb" ]; then
        print_info "  - IMDB (~80MB)"
        cp -r "$DATA_DIR/imdb" "$TMP_DIR/stage4/"
    else
        print_warning "  - IMDB not found, skipping"
    fi

    # Create checksums file
    print_info "Generating checksums..."
    cd "$TMP_DIR"
    find stage4 -type f -exec shasum -a 256 {} \; > checksums.txt
    mv checksums.txt stage4/

    # Create tarball
    OUTPUT_FILE="$OUTPUT_DIR/stage4-data.tar.gz"
    print_info "Creating archive: $OUTPUT_FILE"

    tar -czf "$OUTPUT_FILE" stage4/

    # Calculate size
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    print_success "Dataset archive created: $OUTPUT_FILE ($SIZE)"

    # Cleanup
    rm -rf "$TMP_DIR"
fi

# Package models
if [ "$PACKAGE_MODELS" = true ]; then
    print_header "Packaging Pretrained Models"

    # Check if models directory exists
    if [ ! -d "$MODELS_DIR" ]; then
        print_warning "Models directory not found: $MODELS_DIR"
        print_info "Models will be downloaded automatically on first use"
        print_info "Skipping model packaging"
    else
        # Create temporary directory for packaging
        TMP_DIR=$(mktemp -d)
        mkdir -p "$TMP_DIR/models"

        # Copy models
        print_info "Copying models to temporary directory..."

        # ResNet-50
        if [ -f "$MODELS_DIR/resnet50_pytorch.pth" ]; then
            print_info "  - ResNet-50 PyTorch (~100MB)"
            cp "$MODELS_DIR/resnet50_pytorch.pth" "$TMP_DIR/models/"
        else
            print_warning "  - ResNet-50 not found (will download on first use)"
        fi

        # BERT (optional)
        if [ -d "$MODELS_DIR/bert-base-uncased" ]; then
            print_info "  - BERT base uncased (~440MB)"
            cp -r "$MODELS_DIR/bert-base-uncased" "$TMP_DIR/models/"
        else
            print_warning "  - BERT not found (optional)"
        fi

        # YOLOv8 (optional)
        if [ -f "$MODELS_DIR/yolov8n.pt" ]; then
            print_info "  - YOLOv8 Nano (~6MB)"
            cp "$MODELS_DIR/yolov8n.pt" "$TMP_DIR/models/"
        else
            print_warning "  - YOLOv8 not found (optional)"
        fi

        # Create checksums file
        print_info "Generating checksums..."
        cd "$TMP_DIR"
        find models -type f -exec shasum -a 256 {} \; > checksums.txt
        mv checksums.txt models/

        # Create tarball
        OUTPUT_FILE="$OUTPUT_DIR/stage4-models.tar.gz"
        print_info "Creating archive: $OUTPUT_FILE"

        tar -czf "$OUTPUT_FILE" models/

        # Calculate size
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        print_success "Models archive created: $OUTPUT_FILE ($SIZE)"

        # Cleanup
        rm -rf "$TMP_DIR"
    fi
fi

# Summary
print_header "Packaging Complete"

if [ "$PACKAGE_DATA" = true ] && [ -f "$OUTPUT_DIR/stage4-data.tar.gz" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/stage4-data.tar.gz" | cut -f1)
    print_success "Datasets: $OUTPUT_DIR/stage4-data.tar.gz ($SIZE)"
fi

if [ "$PACKAGE_MODELS" = true ] && [ -f "$OUTPUT_DIR/stage4-models.tar.gz" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/stage4-models.tar.gz" | cut -f1)
    print_success "Models: $OUTPUT_DIR/stage4-models.tar.gz ($SIZE)"
fi

echo ""
print_info "To use offline packages:"
echo "  1. Transfer files to target system"
echo "  2. Extract: tar -xzf stage4-data.tar.gz -C data/"
echo "  3. Extract: tar -xzf stage4-models.tar.gz -C data/"
echo "  4. Verify: python scripts/data/verify.py --stage 4"

print_success "✓ All packaging tasks completed successfully!"
