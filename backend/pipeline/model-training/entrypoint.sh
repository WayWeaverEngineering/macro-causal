#!/bin/bash

# Entrypoint script for Ray model training container

set -e

echo "Starting Ray model training container..."

# Get environment variables
EXECUTION_ID=${EXECUTION_ID:-"unknown"}
GOLD_BUCKET=${GOLD_BUCKET:-""}
ARTIFACTS_BUCKET=${ARTIFACTS_BUCKET:-""}
MODEL_REGISTRY_TABLE=${MODEL_REGISTRY_TABLE:-""}

# Validate required environment variables
if [ -z "$GOLD_BUCKET" ]; then
    echo "Error: GOLD_BUCKET environment variable is required"
    exit 1
fi

if [ -z "$ARTIFACTS_BUCKET" ]; then
    echo "Error: ARTIFACTS_BUCKET environment variable is required"
    exit 1
fi

if [ -z "$MODEL_REGISTRY_TABLE" ]; then
    echo "Error: MODEL_REGISTRY_TABLE environment variable is required"
    exit 1
fi

echo "Environment variables:"
echo "  EXECUTION_ID: $EXECUTION_ID"
echo "  GOLD_BUCKET: $GOLD_BUCKET"
echo "  ARTIFACTS_BUCKET: $ARTIFACTS_BUCKET"
echo "  MODEL_REGISTRY_TABLE: $MODEL_REGISTRY_TABLE"

# Run the training script
echo "Starting model training..."
python main.py \
    --execution-id "$EXECUTION_ID" \
    --gold-bucket "$GOLD_BUCKET" \
    --artifacts-bucket "$ARTIFACTS_BUCKET" \
    --model-registry-table "$MODEL_REGISTRY_TABLE"

echo "Model training completed successfully"
