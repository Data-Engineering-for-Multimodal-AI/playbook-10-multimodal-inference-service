#!/bin/bash

# GitHub Setup Script for Multimodal Inference Service
# This script prepares the repository for GitHub push

echo "Setting up repository for GitHub..."

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Add all files to git
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Multimodal Inference Service

- Complete PyTorch + FastAPI implementation
- ResNet18 vision encoder with MLP fusion
- RLBench data compatibility
- FastAPI service with health checks and metrics
- Docker containerization
- Canary deployment simulation
- Production-ready architecture"

echo ""
echo "Repository is ready for GitHub!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/yourusername/multimodal-inference-service.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Optional: Set up GitHub Actions, issues templates, etc."