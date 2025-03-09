FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Basic Pitch with TensorFlow support
# Use a specific version to avoid compatibility issues
RUN pip install --no-cache-dir basic-pitch==0.4.0 tensorflow==2.12.0

# Copy the application code
COPY . .

# Environment variables (these can be overridden at runtime)
ENV ALLOW_ANONYMOUS_ACCESS=false
ENV ALLOW_INTERNAL_NETWORK=false

# Expose the port that the application will run on
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--threads", "8", "--timeout", "600", "--bind", "0.0.0.0:8000", "main:app"] 