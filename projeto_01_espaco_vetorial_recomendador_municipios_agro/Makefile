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

.PHONY: help setup test test-network test-all lint notebook \
        download-loc download-ppm download-all build-features \
        vectorize vectorize-no-stem recommend clean clean-data

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
	@echo "  setup             Cria .venv com Python 3.12 e instala dependencias"
	@echo "  test              Roda pytest sem os testes de rede (default)"
	@echo "  test-network      Roda apenas os testes de integracao com a API IBGE"
	@echo "  test-all          Roda TODA a suite, incluindo testes de rede"
	@echo "  lint              Roda ruff (lint + format check)"
	@echo "  notebook          Sobe o Jupyter Lab"
	@echo "  download-loc      Baixa lista de municipios da API Localidades IBGE"
	@echo "  download-ppm      Baixa PPM tabela 3939 da API SIDRA"
	@echo "  download-all      Baixa localidades + PPM"
	@echo "  build-features    Roda pipeline de feature engineering (features.py)"
	@echo "  vectorize         Ajusta CountVectorizer e salva matriz esparsa"
	@echo "  vectorize-no-stem Idem, mas sem stemming (para comparacao)"
	@echo "  recommend NOME    Consulta o recomendador (uso: make recommend NOME=Cambuquira UF=MG)"
	@echo "  clean             Remove __pycache__, .pytest_cache, .ruff_cache"
	@echo "  clean-data        Remove data/interim/* e data/processed/* (preserva raw)"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[notebook,dev]"
	@echo ""
	@echo "[OK] Ambiente criado em .venv/"
	@echo "     Ative com: source .venv/bin/activate  (Linux/Mac)"
	@echo "                .venv\\Scripts\\Activate.ps1  (Windows)"

test:
	$(PYTHON) -m pytest tests/ -v -m "not network"

test-network:
	$(PYTHON) -m pytest tests/ -v -m network

test-all:
	$(PYTHON) -m pytest tests/ -v

download-loc:
	$(PYTHON) -m rec_agro_br.dataset localidades

download-ppm:
	$(PYTHON) -m rec_agro_br.dataset ppm

download-all:
	$(PYTHON) -m rec_agro_br.dataset all

build-features:
	$(PYTHON) -m rec_agro_br.features

vectorize:
	$(PYTHON) -m rec_agro_br.vectorize

vectorize-no-stem:
	$(PYTHON) -m rec_agro_br.vectorize --sem-stemming

recommend:
	@if [ -z "$(NOME)" ]; then \
		echo "[ERRO] Uso: make recommend NOME=Cambuquira UF=MG K=5"; \
		exit 1; \
	fi
	$(PYTHON) -m rec_agro_br.recommender "$(NOME)" $(if $(UF),--uf $(UF)) $(if $(K),--k $(K))

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
