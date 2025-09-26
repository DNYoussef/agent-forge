#!/bin/bash

# Script to push clean-main branch to new repository

echo "Setting up new repository..."

# Remove old origin
git remote remove origin

# Add new origin (you'll create this repo on GitHub first)
git remote add origin https://github.com/DNYoussef/agent-forge.git

# Verify we're on clean-main
echo "Current branch:"
git branch --show-current

# Push to new repository
echo "Pushing to new repository..."
git push -u origin clean-main:main

echo "Done! Your new repository is set up with a clean history."
echo "The main branch now contains all your code without the large file history."