#!/bin/bash

# Entrypoint script for Hybrid Causal Inference Training Pipeline

set -e

echo "Starting Hybrid Causal Inference Training Pipeline..."

# Get environment variables
EXECUTION_ID=${EXECUTION_ID:-"unknown"}
GOLD_BUCKET=${GOLD_BUCKET:-""}
ARTIFACTS_BUCKET=${ARTIFACTS_BUCKET:-""}

# Validate required environment variables
if [ -z "$GOLD_BUCKET" ]; then
    echo "Error: GOLD_BUCKET environment variable is required"
    exit 1
fi

if [ -z "$ARTIFACTS_BUCKET" ]; then
    echo "Error: ARTIFACTS_BUCKET environment variable is required"
    exit 1
fi

echo "Environment variables:"
echo "  EXECUTION_ID: $EXECUTION_ID"
echo "  GOLD_BUCKET: $GOLD_BUCKET"
echo "  ARTIFACTS_BUCKET: $ARTIFACTS_BUCKET"

# Check if we're in testing mode
if [ "$TESTING" = "true" ]; then
    echo "Running in testing mode..."
    python test_training.py
    exit $?
fi

# Run the training script
echo "Starting hybrid causal inference training..."
python main.py \
    --execution-id "$EXECUTION_ID" \
    --gold-bucket "$GOLD_BUCKET" \
    --artifacts-bucket "$ARTIFACTS_BUCKET"

echo "Hybrid causal inference training completed successfully"
