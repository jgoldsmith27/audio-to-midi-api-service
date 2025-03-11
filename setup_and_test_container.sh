#!/bin/bash
# Script to setup and test the Basic Pitch Docker container

# Print commands as they are executed
set -x

# Check for test audio file
echo "Checking for test audio file..."
python download_test_audio.py

# Create sample_audio directory if it doesn't exist
mkdir -p sample_audio

# Build the Docker container
echo "Building Docker container..."
docker-compose build

# Run the test container
echo "Running test container..."
docker-compose run test

# If tests pass, start the service
echo "Starting the Basic Pitch API service..."
docker-compose up -d api

# Check if the service is running
echo "Checking if service is running..."
sleep 10
curl -I http://localhost:8000/health

echo "Done! The API service should now be running at http://localhost:8000"
echo "Use Ctrl+C to stop the service"

# Follow the logs
docker-compose logs -f api 