#!/usr/bin/env pwsh
# Run local CI checks

Write-Host "Running tests..." -ForegroundColor Cyan
uv run pytest tests --cov=src/coordinatus --cov-report=term-missing --cov-report=html
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nRunning ruff..." -ForegroundColor Cyan
uv run ruff check --fix
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ruff checks failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nRunning ty..." -ForegroundColor Cyan
uv run ty check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ty checks failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nExecuting notebooks..." -ForegroundColor Cyan
Get-ChildItem notebooks/*.ipynb | ForEach-Object {
    Write-Host "  Running $($_.Name)..." -ForegroundColor Gray
    uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 --inplace $_.FullName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Notebook $($_.Name) failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`nAll checks passed! ✓" -ForegroundColor Green
