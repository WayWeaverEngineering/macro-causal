#!/bin/bash
set -euo pipefail

# Master script to build all Lambda layers
echo "=== Lambda Layers Build Script ==="
echo "This script will build all Python Lambda layers in the current directory"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to install dependencies for a layer
install_layer_dependencies() {
    local layer_name="$1"
    local layer_dir="$2"
    
    echo "Installing Python dependencies for $layer_name..."
    
    # Create directory structure
    mkdir -p "$layer_dir/create_layer/lib/python3.10/site-packages"
    
    # Install dependencies
    echo "Installing dependencies from $layer_dir/requirements.txt..."
    pip install -r "$layer_dir/requirements.txt" --target "$layer_dir/create_layer/lib/python3.10/site-packages" --quiet
    
    echo "Cleaning up unnecessary files to reduce layer size..."
    
    # Remove Python cache files
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -name "*.pyc" -delete 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove package metadata (dist-info and egg-info)
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove test files and documentation
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -name "*.md" -delete 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -name "*.txt" -delete 2>/dev/null || true
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -name "*.rst" -delete 2>/dev/null || true
    
    # Remove unnecessary pandas files (keep only essential components)
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/pandas" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/pandas" -name "*.pyx" -delete 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/pandas" -name "*.pxd" -delete 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/pandas" -name "*.pxi" -delete 2>/dev/null || true
    fi
    
    # Remove numpy test files
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/numpy" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/numpy" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/numpy" -name "*.pyx" -delete 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/numpy" -name "*.pxd" -delete 2>/dev/null || true
    fi
    
    # Remove boto3 documentation and examples
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/boto3" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/boto3" -name "*.md" -delete 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/boto3" -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # Remove botocore documentation and examples
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/botocore" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/botocore" -name "*.md" -delete 2>/dev/null || true
        find "$layer_dir/create_layer/lib/python3.10/site-packages/botocore" -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true
    fi
    
    # Remove yfinance documentation
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/yfinance" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/yfinance" -name "*.md" -delete 2>/dev/null || true
    fi
    
    # Remove requests documentation
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/requests" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/requests" -name "*.md" -delete 2>/dev/null || true
    fi
    
    # Remove python-dateutil documentation
    if [ -d "$layer_dir/create_layer/lib/python3.10/site-packages/dateutil" ]; then
        find "$layer_dir/create_layer/lib/python3.10/site-packages/dateutil" -name "*.md" -delete 2>/dev/null || true
    fi
    
    # Remove empty directories
    find "$layer_dir/create_layer/lib/python3.10/site-packages" -type d -empty -delete 2>/dev/null || true
    
    # Show final layer size and contents
    echo "Layer cleanup completed for $layer_name!"
    local layer_size=$(du -sh "$layer_dir/create_layer/lib/python3.10/site-packages" | cut -f1)
    local package_count=$(ls "$layer_dir/create_layer/lib/python3.10/site-packages" | wc -l)
    echo "✓ $layer_name layer built successfully (${layer_size}, ${package_count} packages)"
}

# Function to package a layer
package_layer() {
    local layer_name="$1"
    local layer_dir="$2"
    
    echo "Packaging layer: $layer_name..."
    
    # Create python directory and copy lib
    mkdir -p "$layer_dir/python"
    cp -r "$layer_dir/create_layer/lib" "$layer_dir/python/"
    
    # Create zip file
    cd "$layer_dir"
    zip -r layer_content.zip python > /dev/null 2>&1
    cd "$SCRIPT_DIR"
    
    # Clean up intermediate artifacts immediately after packaging
    echo "Cleaning up intermediate artifacts for $layer_name..."
    rm -rf "$layer_dir/create_layer" "$layer_dir/python" 2>/dev/null || true
    
    local zip_size=$(du -h "$layer_dir/layer_content.zip" | cut -f1)
    echo "✓ $layer_name packaged successfully (${zip_size})"
}

# Function to build a single layer
build_layer() {
    local layer_name="$1"
    local layer_dir="$2"
    
    echo "=== Building layer: $layer_name ==="
    
    # Check if the layer directory exists
    if [ ! -d "$layer_dir" ]; then
        echo "Error: Layer directory '$layer_dir' does not exist"
        return 1
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "$layer_dir/requirements.txt" ]; then
        echo "Error: requirements.txt not found in $layer_dir"
        return 1
    fi
    
    # Install dependencies
    if ! install_layer_dependencies "$layer_name" "$layer_dir"; then
        echo "Error: Failed to install dependencies for $layer_name"
        return 1
    fi
    
    # Package the layer
    if ! package_layer "$layer_name" "$layer_dir"; then
        echo "Error: Failed to package $layer_name"
        return 1
    fi
    
    echo ""
    
    return 0
}

# Function to clean up build artifacts
cleanup_layer() {
    local layer_name="$1"
    local layer_dir="$2"
    
    # Remove build directories and files (quietly)
    rm -rf "$layer_dir/create_layer" "$layer_dir/python" "$layer_dir/layer_content.zip" 2>/dev/null || true
}

# Main execution
main() {
    echo "Starting build process..."
    echo "Working directory: $SCRIPT_DIR"
    echo ""
    
    # Get all subdirectories (layer folders)
    local layer_dirs=()
    for dir in */; do
        if [ -d "$dir" ]; then
            layer_dirs+=("$dir")
        fi
    done
    
    if [ ${#layer_dirs[@]} -eq 0 ]; then
        echo "Error: No layer directories found"
        exit 1
    fi
    
    echo "Found ${#layer_dirs[@]} layer(s) to build:"
    for dir in "${layer_dirs[@]}"; do
        echo "  - ${dir%/}"
    done
    echo ""
    
    # Build each layer
    local failed_layers=()
    for dir in "${layer_dirs[@]}"; do
        local layer_name="${dir%/}"
        local layer_dir="$SCRIPT_DIR/$dir"
        
        if build_layer "$layer_name" "$layer_dir"; then
            # Success message is already printed in the build function
            :
        else
            echo "✗ Failed to build $layer_name"
            failed_layers+=("$layer_name")
        fi
    done
    
    echo ""
    echo "=== Build Summary ==="
    
    if [ ${#failed_layers[@]} -eq 0 ]; then
        echo "✓ All layers built successfully!"
        echo ""
        echo "All layer packages created successfully!"
    else
        echo "✗ Some layers failed to build:"
        for layer in "${failed_layers[@]}"; do
            echo "  - $layer"
        done
        echo ""
        echo "Please check the error messages above and fix the issues."
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean, -c     Clean up build artifacts before building"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              Build all layers"
    echo "  $0 --clean      Clean and build all layers"
}

# Parse command line arguments
CLEAN_FIRST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c)
            CLEAN_FIRST=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute main function
if [ "$CLEAN_FIRST" = true ]; then
    echo "Cleaning up existing build artifacts..."
    for dir in */; do
        if [ -d "$dir" ]; then
            local layer_name="${dir%/}"
            cleanup_layer "$layer_name" "$SCRIPT_DIR/$dir"
        fi
    done
    echo "Cleanup completed."
    echo ""
fi

main
