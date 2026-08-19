param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$Python = "python"

function Show-Help {
    Write-Host ""
    Write-Host "Alvos disponiveis:"
    Write-Host "  setup             Instala o pacote em modo desenvolvimento com extras dev"
    Write-Host "  test              Roda a suite offline (default)"
    Write-Host "  test-network      Roda testes marcados como @pytest.mark.network"
    Write-Host "  test-all          Roda tudo (offline + network + slow)"
    Write-Host "  lint              Roda ruff check"
    Write-Host "  notebook          Abre jupyter lab na pasta notebooks/"
    Write-Host "  clean             Remove __pycache__, .pytest_cache, .ruff_cache"
    Write-Host ""
}

function Invoke-Setup {
    & $Python -m pip install -e ".[dev,notebook]"
}

function Invoke-Test {
    & $Python -m pytest tests/
}

function Invoke-TestNetwork {
    & $Python -m pytest tests/ -m network -p no:cacheprovider
}

function Invoke-TestAll {
    & $Python -m pytest tests/ -m ""
}

function Invoke-Lint {
    & $Python -m ruff check src/ tests/
}

function Invoke-Notebook {
    & $Python -m jupyter lab --notebook-dir=notebooks
}

function Invoke-Clean {
    Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force .ruff_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force reports -ErrorAction SilentlyContinue
    Remove-Item -Force .coverage -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    Write-Host "[OK] Limpeza concluida."
}

switch ($Task.ToLower()) {
    "help"          { Show-Help }
    "setup"         { Invoke-Setup }
    "test"          { Invoke-Test }
    "test-network"  { Invoke-TestNetwork }
    "test-all"      { Invoke-TestAll }
    "lint"          { Invoke-Lint }
    "notebook"      { Invoke-Notebook }
    "clean"         { Invoke-Clean }
    default {
        Write-Host "[ERRO] Alvo desconhecido: $Task"
        Show-Help
        exit 1
    }
}
