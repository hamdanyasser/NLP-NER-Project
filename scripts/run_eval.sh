#!/bin/bash

# Evaluation script for BiLSTM-CRF NER model

echo "=========================================="
echo "BiLSTM-CRF NER Evaluation"
echo "=========================================="
echo ""

# Default values
MODEL="bilstm_crf"
CONFIG="config/config.yaml"
CHECKPOINT=""

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
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --baseline)
            MODEL="baseline"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model baseline|bilstm_crf] [--config path/to/config.yaml] [--checkpoint path/to/model.pt] [--baseline]"
            exit 1
            ;;
    esac
done

echo "Model: $MODEL"
echo "Config: $CONFIG"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo ""

# Run evaluation
if [ -n "$CHECKPOINT" ]; then
    python -m src.training.eval --config "$CONFIG" --model "$MODEL" --checkpoint "$CHECKPOINT"
else
    python -m src.training.eval --config "$CONFIG" --model "$MODEL"
fi

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
