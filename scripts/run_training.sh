#!/bin/bash

# Script to run 2-stage training: pre-training followed by fine-tuning

echo "Starting 2-stage training pipeline..."

# Stage 1: Pre-training
echo "========================================="
echo "Stage 1: Running pre-training..."
echo "========================================="
python train.py --config pre-training

# Check if pre-training completed successfully
if [ $? -eq 0 ]; then
    echo "Pre-training completed successfully!"
    
    # Stage 2: Fine-tuning
    echo "========================================="
    echo "Stage 2: Running fine-tuning..."
    echo "========================================="
    python train.py --config fine-tuning
    
    if [ $? -eq 0 ]; then
        echo "Fine-tuning completed successfully!"
        echo "2-stage training pipeline completed!"
    else
        echo "Fine-tuning failed!"
        exit 1
    fi
else
    echo "Pre-training failed!"
    exit 1
fi 