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
    
    # Define installation directory - now directly in python folder
    local installation_dir="$layer_dir/python"
    
    echo "Installing Python dependencies for $layer_name..."
    
    # Create python directory structure
    mkdir -p "$installation_dir"
    
    # Install dependencies directly into the python folder
    echo "Installing dependencies from $layer_dir/requirements.txt..."
    pip install -r "$layer_dir/requirements.txt" --target "$installation_dir" --quiet
    
    # Show final layer size and contents
    local layer_size=$(du -sh "$installation_dir" | cut -f1)
    local package_count=$(ls "$installation_dir" | wc -l)
    echo "✓ $layer_name layer built successfully (${layer_size}, ${package_count} packages)"
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
    
    echo ""
    
    return 0
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
    echo "All layer python folders created successfully!"
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
echo "  --clean, -c     (Deprecated - python folders are always preserved)"
echo "  --help, -h      Show this help message"
echo ""
echo "Examples:"
echo "  $0              Build all layers"
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
    echo "Note: --clean option is not needed since python folders are preserved."
    echo ""
fi

main
