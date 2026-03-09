#!/bin/bash
# Setup script for automatic versioning with commitizen

set -e

echo "🚀 Setting up automatic versioning..."

# Check if commitizen is installed
if ! command -v cz &> /dev/null; then
    echo "📦 Installing commitizen..."
    pip install commitizen
fi

# Make hooks executable
chmod +x .git/hooks/pre-push 2>/dev/null || true

# Install pre-commit (optional but recommended)
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit framework..."
    pip install pre-commit
fi

# Install pre-commit hooks
pre-commit install --hook-type commit-msg 2>/dev/null || true
pre-commit install --hook-type pre-push 2>/dev/null || true

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Write commits in conventional format (e.g., 'feat: add feature')"
echo "2. Push with: git push"
echo "3. Version auto-bumps and tags are created automatically"
echo ""
echo "📖 See VERSIONING.md for detailed instructions"
