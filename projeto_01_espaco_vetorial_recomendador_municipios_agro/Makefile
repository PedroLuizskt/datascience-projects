# =============================================================================
# projeto_01 — Makefile
# =============================================================================
# Uso (Linux/WSL/Git Bash):
#   make help         Lista todos os alvos disponíveis
#   make setup        Cria .venv e instala dependências
#   make test         Roda a suíte pytest
#   make lint         Roda ruff no código
#   make notebook     Sobe o Jupyter Lab
#   make clean        Remove artefatos temporários
#   make clean-data   Remove dados intermediários e processados (mantém raw)
#
# Usuários Windows nativos: use o script tasks.ps1 equivalente. Ver README.
# =============================================================================

.PHONY: help setup test lint notebook clean clean-data

# Detecção do sistema para escolher o python correto
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    PIP := .venv/Scripts/pip.exe
else
    PYTHON := .venv/bin/python
    PIP := .venv/bin/pip
endif

help:
	@echo "Alvos disponiveis:"
	@echo "  setup        Cria .venv com Python 3.12 e instala dependencias"
	@echo "  test         Roda pytest com cobertura"
	@echo "  lint         Roda ruff (lint + format check)"
	@echo "  notebook     Sobe o Jupyter Lab"
	@echo "  clean        Remove __pycache__, .pytest_cache, .ruff_cache"
	@echo "  clean-data   Remove data/interim/* e data/processed/* (preserva raw)"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[notebook,dev]"
	@echo ""
	@echo "[OK] Ambiente criado em .venv/"
	@echo "     Ative com: source .venv/bin/activate  (Linux/Mac)"
	@echo "                .venv\\Scripts\\Activate.ps1  (Windows)"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

notebook:
	$(PYTHON) -m jupyter lab --notebook-dir=notebooks/

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	@rm -rf reports/coverage .coverage
	@echo "[OK] Artefatos temporarios removidos"

clean-data:
	@rm -f data/interim/*.parquet data/interim/*.csv 2>/dev/null || true
	@rm -f data/processed/*.parquet data/processed/*.csv 2>/dev/null || true
	@echo "[OK] Dados intermediarios e processados removidos (raw preservado)"
