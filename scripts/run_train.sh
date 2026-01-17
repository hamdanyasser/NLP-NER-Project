#!/bin/bash

# Training script for BiLSTM-CRF NER model

echo "=========================================="
echo "BiLSTM-CRF NER Training"
echo "=========================================="
echo ""

# Default values
MODEL="bilstm_crf"
CONFIG="config/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --baseline)
            MODEL="baseline"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model baseline|bilstm_crf] [--config path/to/config.yaml] [--baseline]"
            exit 1
            ;;
    esac
done

echo "Model: $MODEL"
echo "Config: $CONFIG"
echo ""

# Run training
python -m src.training.train --config "$CONFIG" --model "$MODEL"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
