# =============================================================================
# projeto_01 - tasks.ps1
# =============================================================================
# Equivalente ao Makefile para PowerShell nativo do Windows.
#
# Uso:
#   .\tasks.ps1 help
#   .\tasks.ps1 setup
#   .\tasks.ps1 test
#   .\tasks.ps1 lint
#   .\tasks.ps1 notebook
#   .\tasks.ps1 clean
#   .\tasks.ps1 clean-data
#
# Requer PowerShell 5.1 ou superior e Python 3.12 disponivel via 'py -3.12'.
# =============================================================================

param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
$Pip = ".\.venv\Scripts\pip.exe"

function Show-Help {
    Write-Host "Alvos disponiveis:"
    Write-Host "  setup             Cria .venv com Python 3.12 e instala dependencias"
    Write-Host "  test              Roda pytest sem os testes de rede (default)"
    Write-Host "  test-network      Roda apenas os testes de integracao com a API IBGE"
    Write-Host "  test-all          Roda TODA a suite, incluindo testes de rede"
    Write-Host "  lint              Roda ruff (lint + format check)"
    Write-Host "  notebook          Sobe o Jupyter Lab"
    Write-Host "  download-loc      Baixa lista de municipios da API Localidades IBGE"
    Write-Host "  download-ppm      Baixa PPM tabela 3939 da API SIDRA"
    Write-Host "  download-all      Baixa localidades + PPM"
    Write-Host "  build-features    Roda pipeline de feature engineering (features.py)"
    Write-Host "  clean             Remove __pycache__, .pytest_cache, .ruff_cache"
    Write-Host "  clean-data        Remove data/interim/* e data/processed/*"
}

function Invoke-Setup {
    Write-Host "[INFO] Criando ambiente virtual com Python 3.12..."
    py -3.12 -m venv .venv

    Write-Host "[INFO] Atualizando pip..."
    & $Python -m pip install --upgrade pip

    Write-Host "[INFO] Instalando dependencias (modo editavel + notebook + dev)..."
    & $Pip install -e ".[notebook,dev]"

    Write-Host ""
    Write-Host "[OK] Ambiente criado em .venv\"
    Write-Host "     Ative com: .\.venv\Scripts\Activate.ps1"
}

function Invoke-Test {
    & $Python -m pytest tests/ -v -m "not network"
}

function Invoke-TestNetwork {
    & $Python -m pytest tests/ -v -m network
}

function Invoke-TestAll {
    & $Python -m pytest tests/ -v
}

function Invoke-DownloadLoc {
    & $Python -m rec_agro_br.dataset localidades
}

function Invoke-DownloadPPM {
    & $Python -m rec_agro_br.dataset ppm
}

function Invoke-DownloadAll {
    & $Python -m rec_agro_br.dataset all
}

function Invoke-BuildFeatures {
    & $Python -m rec_agro_br.features
}

function Invoke-Lint {
    & $Python -m ruff check src/ tests/
    & $Python -m ruff format --check src/ tests/
}

function Invoke-Notebook {
    & $Python -m jupyter lab --notebook-dir=notebooks/
}

function Invoke-Clean {
    Get-ChildItem -Path . -Include "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints" `
        -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    if (Test-Path "reports\coverage") {
        Remove-Item "reports\coverage" -Recurse -Force
    }
    if (Test-Path ".coverage") {
        Remove-Item ".coverage" -Force
    }
    Write-Host "[OK] Artefatos temporarios removidos"
}

function Invoke-CleanData {
    Get-ChildItem -Path "data\interim" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path "data\processed" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] Dados intermediarios e processados removidos (raw preservado)"
}

switch ($Task) {
    "help"           { Show-Help }
    "setup"          { Invoke-Setup }
    "test"           { Invoke-Test }
    "test-network"   { Invoke-TestNetwork }
    "test-all"       { Invoke-TestAll }
    "lint"           { Invoke-Lint }
    "notebook"       { Invoke-Notebook }
    "download-loc"   { Invoke-DownloadLoc }
    "download-ppm"   { Invoke-DownloadPPM }
    "download-all"   { Invoke-DownloadAll }
    "build-features" { Invoke-BuildFeatures }
    "clean"          { Invoke-Clean }
    "clean-data"     { Invoke-CleanData }
    default {
        Write-Host "[ERRO] Alvo desconhecido: $Task"
        Show-Help
        exit 1
    }
}
