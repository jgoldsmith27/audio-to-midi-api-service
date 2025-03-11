#!/bin/bash
# Deployment script for Basic Pitch API service

# Print commands as they are executed
set -x

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please create a .env file with the required environment variables."
    exit 1
fi

# Load environment variables
source .env

# Pull latest code (uncomment if deploying from Git)
# git pull origin main

# Build the Docker container
echo "Building Docker container..."
docker-compose -f docker-compose.prod.yml build

# Stop and remove existing containers
echo "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start the service
echo "Starting the Basic Pitch API service..."
docker-compose -f docker-compose.prod.yml up -d

# Check if the service is running
echo "Checking if service is running..."
sleep 10
curl -I http://localhost:8000/health

echo "Deployment completed successfully!"
echo "The API service is running at http://localhost:8000"

# Optional: Follow logs
if [ "$1" == "--follow-logs" ]; then
    docker-compose -f docker-compose.prod.yml logs -f
fi 