#!/bin/bash
set -euo pipefail

# Script to clean up all Lambda layer build artifacts
echo "=== Lambda Layers Cleanup Script ==="
echo "This script will clean up all build artifacts from Lambda layers"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to clean up a single layer
cleanup_layer() {
    local layer_name="$1"
    local layer_dir="$2"
    
    echo "Cleaning up artifacts for layer: $layer_name"
    
    # Remove build directories and files
    local items_removed=0
    
    # Remove create_layer directory (intermediate build artifacts)
    if [ -d "$layer_dir/create_layer" ]; then
        rm -rf "$layer_dir/create_layer"
        echo "  ✓ Removed create_layer directory"
        ((items_removed++))
    fi
    
    # Remove python directory (intermediate packaging artifacts)
    if [ -d "$layer_dir/python" ]; then
        rm -rf "$layer_dir/python"
        echo "  ✓ Removed python directory"
        ((items_removed++))
    fi
    
    # Remove layer_content.zip file (final package)
    if [ -f "$layer_dir/layer_content.zip" ]; then
        rm -f "$layer_dir/layer_content.zip"
        echo "  ✓ Removed layer_content.zip"
        ((items_removed++))
    fi
    
    if [ $items_removed -eq 0 ]; then
        echo "  ℹ No artifacts found to clean up"
    fi
}

# Main execution
main() {
    echo "Starting cleanup process..."
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
        echo "No layer directories found to clean up"
        exit 0
    fi
    
    echo "Found ${#layer_dirs[@]} layer(s) to clean up:"
    for dir in "${layer_dirs[@]}"; do
        echo "  - ${dir%/}"
    done
    echo ""
    
    # Clean up each layer
    local cleaned_layers=0
    for dir in "${layer_dirs[@]}"; do
        local layer_name="${dir%/}"
        local layer_dir="$SCRIPT_DIR/$dir"
        
        cleanup_layer "$layer_name" "$layer_dir"
        ((cleaned_layers++))
        echo ""
    done
    
    echo "=== Cleanup Summary ==="
    echo "✓ Cleaned up artifacts for $cleaned_layers layer(s)"
    echo "✓ All build artifacts removed successfully"
    echo ""
    echo "Layer directories now contain only:"
    echo "  - requirements.txt files"
    echo "  - README.md (if present)"
    echo ""
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "This script removes all build artifacts from Lambda layers:"
    echo "  - create_layer/ directories"
    echo "  - python/ directories"
    echo "  - layer_content.zip files"
    echo ""
    echo "Example:"
    echo "  $0              Clean up all layer artifacts"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
main
