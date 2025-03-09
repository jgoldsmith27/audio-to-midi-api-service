#!/bin/bash

# Script to start the Basic Pitch Converter API with proper settings
echo "Starting Basic Pitch Converter API..."

# Set environment variables if not already set
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
export ALLOW_ANONYMOUS_ACCESS=${ALLOW_ANONYMOUS_ACCESS:-"true"}

# Get the PORT from environment or use default
# Render sets PORT automatically, so we need to use that
PORT=${PORT:-8000}

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