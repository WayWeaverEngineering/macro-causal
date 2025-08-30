#!/bin/bash
set -euo pipefail

# Script to build Python Lambda layer with dependencies
# This script installs Python packages to the correct directory structure for Lambda layers

LAYER_DIR="python-dependencies"
SITE_PACKAGES_DIR="${LAYER_DIR}/python/lib/python3.10/site-packages"
REQUIREMENTS_FILE="requirements.txt"

echo "Building Python Lambda layer..."

# Clean up any existing layer
rm -rf "${LAYER_DIR}"

# Create directory structure
mkdir -p "${SITE_PACKAGES_DIR}"

# Install dependencies to the layer directory
echo "Installing Python dependencies..."
pip install -r "${REQUIREMENTS_FILE}" -t "${SITE_PACKAGES_DIR}" --platform manylinux2014_x86_64 --only-binary=all

# Clean up unnecessary files to reduce layer size
echo "Cleaning up layer..."
find "${SITE_PACKAGES_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${SITE_PACKAGES_DIR}" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find "${SITE_PACKAGES_DIR}" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Verify key packages are installed
echo "Verifying packages..."
ls -la "${SITE_PACKAGES_DIR}" | grep -E "(numpy|pandas|requests)"

echo "Python Lambda layer built successfully!"
echo "Layer directory: ${LAYER_DIR}"
echo "Total size: $(du -sh ${LAYER_DIR} | cut -f1)"
