# Setup script for automatic versioning with commitizen (Windows)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up automatic versioning..." -ForegroundColor Cyan

# Check if commitizen is installed
try {
    cz version > $null 2>&1
} catch {
    Write-Host "📦 Installing commitizen..." -ForegroundColor Yellow
    pip install commitizen
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "1. Write commits in conventional format (e.g., 'feat: add feature')"
Write-Host "2. Push with: git push"
Write-Host "3. Version auto-bumps and tags are created automatically"
