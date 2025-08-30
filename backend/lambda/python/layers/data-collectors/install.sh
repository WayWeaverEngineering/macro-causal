#!/bin/bash
set -euo pipefail

# Script to install Python dependencies for Lambda layer
echo "Installing Python dependencies..."

# Create virtual environment
python3.10 -m venv create_layer
source create_layer/bin/activate

# Install dependencies
pip install -r requirements.txt --platform=manylinux2014_x86_64 --only-binary=:all: --target ./create_layer/lib/python3.10/site-packages

echo "Cleaning up unnecessary files to reduce layer size..."

# Remove Python cache files
find ./create_layer/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -name "*.pyc" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -name "*.pyo" -delete 2>/dev/null || true

# Remove package metadata (dist-info and egg-info)
find ./create_layer/lib/python3.10/site-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove test files and documentation
find ./create_layer/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -name "*.md" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -name "*.txt" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages -name "*.rst" -delete 2>/dev/null || true

# Remove unnecessary pandas files (keep only essential components)
find ./create_layer/lib/python3.10/site-packages/pandas -name "*.pyx" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/pandas -name "*.pxd" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/pandas -name "*.pxi" -delete 2>/dev/null || true

# Remove numpy test files
find ./create_layer/lib/python3.10/site-packages/numpy -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/numpy -name "*.pyx" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/numpy -name "*.pxd" -delete 2>/dev/null || true

# Remove boto3 documentation and examples
find ./create_layer/lib/python3.10/site-packages/boto3 -name "*.md" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/boto3 -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true

# Remove botocore documentation and examples
find ./create_layer/lib/python3.10/site-packages/botocore -name "*.md" -delete 2>/dev/null || true
find ./create_layer/lib/python3.10/site-packages/botocore -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true

# Remove yfinance documentation
find ./create_layer/lib/python3.10/site-packages/yfinance -name "*.md" -delete 2>/dev/null || true

# Remove requests documentation
find ./create_layer/lib/python3.10/site-packages/requests -name "*.md" -delete 2>/dev/null || true

# Remove python-dateutil documentation
find ./create_layer/lib/python3.10/site-packages/dateutil -name "*.md" -delete 2>/dev/null || true

# Remove empty directories
find ./create_layer/lib/python3.10/site-packages -type d -empty -delete 2>/dev/null || true

# Show final layer size and contents
echo "Layer cleanup completed!"
echo "Final layer size: $(du -sh ./create_layer/lib/python3.10/site-packages | cut -f1)"
echo "Number of packages: $(ls ./create_layer/lib/python3.10/site-packages | wc -l)"
echo "Key packages installed:"
ls ./create_layer/lib/python3.10/site-packages | grep -E "(numpy|pandas|boto3|requests|yfinance|dateutil)"