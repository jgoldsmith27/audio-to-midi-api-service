#!/bin/bash

# Script to initialize a new GitHub repository for the Audio to MIDI API Service
echo "Setting up a new GitHub repository for the Audio to MIDI API Service"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed."
    echo "Please install it from https://cli.github.com/ and try again."
    exit 1
fi

# Check if user is authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "You need to authenticate with GitHub first."
    echo "Run 'gh auth login' and follow the prompts."
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for Audio to MIDI API Service"
else
    echo "Git repository already initialized."
    git add .
    git commit -m "Update files for Audio to MIDI API Service"
fi

# Create a new repository on GitHub
echo "Creating a new repository on GitHub..."
echo "Please enter the name for your GitHub repository (default: audio-to-midi-api-service):"
read repo_name
repo_name=${repo_name:-"audio-to-midi-api-service"}

echo "Please enter a description for your repository (default: 'API service for converting audio to MIDI using Basic Pitch'):"
read repo_description
repo_description=${repo_description:-"API service for converting audio to MIDI using Basic Pitch"}

echo "Should the repository be public or private? (default: public)"
read repo_visibility
repo_visibility=${repo_visibility:-"public"}

# Create the repository
gh repo create "$repo_name" --"$repo_visibility" --description "$repo_description" --source=. --push

echo "GitHub repository created and code pushed!"
echo "Repository URL: https://github.com/$(gh api user | jq -r '.login')/$repo_name"

echo "Next steps for deployment on Render:"
echo "1. Go to https://dashboard.render.com/new/web-service"
echo "2. Connect your GitHub repository"
echo "3. Configure the deployment settings as specified in the README.md" 