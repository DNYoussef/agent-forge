#!/bin/bash

echo "Setting up agent-forger repository..."

# Remove old origin
git remote remove origin 2>/dev/null

# Add new origin
git remote add origin https://github.com/DNYoussef/agent-forger.git

echo "Current remotes:"
git remote -v

echo "Current branch:"
git branch --show-current

# Push to the new repository
echo "Pushing to agent-forger..."
git push -u origin truly-clean:main

if [ $? -eq 0 ]; then
    echo "✅ Success! Your code is now on GitHub at:"
    echo "   https://github.com/DNYoussef/agent-forger"
    echo ""
    echo "Next steps:"
    echo "1. You can now delete the old agent-forge repository"
    echo "2. Your new repo has all 8 phases with 3D visualizations"
    echo "3. No large file history - clean and fast!"
else
    echo "❌ Push failed. Make sure you created the repository on GitHub first:"
    echo "   Go to: https://github.com/new"
    echo "   Repository name: agent-forger"
    echo "   Then run this script again."
fi