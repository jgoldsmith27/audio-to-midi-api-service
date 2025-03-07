#!/bin/bash

# Script to start the Basic Pitch Converter API with proper settings
echo "Starting Basic Pitch Converter API..."

# Set environment variables for TensorFlow
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
# Explicitly disable GPU/CUDA for Render compatibility
export CUDA_VISIBLE_DEVICES="-1"

# Configure TensorFlow for CPU optimization
export TF_INTRA_OP_PARALLELISM_THREADS=2
export TF_INTER_OP_PARALLELISM_THREADS=1
export OMP_NUM_THREADS=2

# Set service configuration
export ALLOW_ANONYMOUS_ACCESS=${ALLOW_ANONYMOUS_ACCESS:-"true"}

# Get the PORT from environment or use default
PORT=${PORT:-8000}

# Print environment configuration for debugging
echo "Environment configuration:"
echo "- TF_CPP_MIN_LOG_LEVEL: $TF_CPP_MIN_LOG_LEVEL"
echo "- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "- ALLOW_ANONYMOUS_ACCESS: $ALLOW_ANONYMOUS_ACCESS"
echo "- PORT: $PORT"

# Start with gunicorn with proper timeouts
echo "Starting server on port $PORT with extended timeouts..."
exec gunicorn -k uvicorn.workers.UvicornWorker \
  -w 1 \
  --threads 2 \
  main:app \
  --bind 0.0.0.0:$PORT \
  --timeout 600 \
  --graceful-timeout 600 \
  --keep-alive 120 