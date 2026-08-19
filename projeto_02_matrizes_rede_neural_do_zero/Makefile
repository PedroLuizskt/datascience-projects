PYTHON := python

.PHONY: help setup test test-network test-all lint clean clean-data

help:
	@echo "Alvos disponiveis:"
	@echo "  setup             Instala o pacote em modo desenvolvimento com extras dev"
	@echo "  test              Roda a suite offline (default)"
	@echo "  test-network      Roda testes marcados como @pytest.mark.network"
	@echo "  test-all          Roda tudo (offline + network + slow)"
	@echo "  lint              Roda ruff check"
	@echo "  clean             Remove __pycache__, .pytest_cache, .ruff_cache"
	@echo "  clean-data        Remove tudo em data/{raw,interim,processed} (cuidado!)"

setup:
	$(PYTHON) -m pip install -e ".[dev,notebook]"

test:
	$(PYTHON) -m pytest tests/

test-network:
	$(PYTHON) -m pytest tests/ -m network -p no:cacheprovider

test-all:
	$(PYTHON) -m pytest tests/ -m ""

lint:
	$(PYTHON) -m ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache reports/ .coverage
	find . -type d -name '*.egg-info' -exec rm -rf {} +

clean-data:
	@echo "AVISO: isso vai apagar data/raw, data/interim e data/processed."
	@read -p "Confirma (yes/no): " conf && [ "$$conf" = "yes" ] || exit 1
	rm -rf data/raw/* data/interim/* data/processed/*
	touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
