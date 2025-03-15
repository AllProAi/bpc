#!/bin/bash
set -e

echo "Starting bin-picking pose estimation service..."

# Run model on startup
if [ "$1" = "serve" ]; then
    echo "Starting inference API server..."
    exec python -m src.inference.api_server
elif [ "$1" = "evaluate" ]; then
    echo "Running evaluation on test data..."
    exec python -m src.evaluation.evaluate --config /app/config.yaml
else
    echo "Running inference with default parameters..."
    exec python -m src.inference.predict --config /app/config.yaml "$@"
fi 