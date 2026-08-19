"""Configurações centralizadas do projeto rna_matrizes.

Carrega variáveis de ambiente do arquivo ``.env`` (se existir), define os
caminhos canônicos do data lake local e expõe defaults de hiperparâmetros
que podem ser sobrescritos via ``.env`` ou via variáveis de ambiente.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


# =============================================================================
# Localização do projeto e carregamento de .env
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Raiz do projeto (dois níveis acima de src/rna_matrizes/config.py)."""

load_dotenv(PROJECT_ROOT / ".env", override=False)


# =============================================================================
# Data lake local (Cookiecutter Data Science layout)
# =============================================================================
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"


def ensure_directories() -> None:
    """Cria todos os diretórios do data lake, idempotente."""
    for d in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
              EXTERNAL_DATA_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Defaults de hiperparâmetros da rede
# =============================================================================
DEFAULT_TAXA_APRENDIZADO: float = float(os.getenv("RNA_TAXA_APRENDIZADO", "0.01"))
"""Taxa de aprendizado do gradiente descendente. Default: 0.01 (mesmo do DSA)."""

DEFAULT_NUM_ITERACOES: int = int(os.getenv("RNA_NUM_ITERACOES", "1000"))
"""Número máximo de iterações. Default: 1000 (mesmo do DSA)."""

DEFAULT_TOLERANCIA: float = float(os.getenv("RNA_TOLERANCIA", "1e-6"))
"""Tolerância para detecção de convergência: se |ΔBCE| < tolerância, para."""

DEFAULT_LIMIAR_DECISAO: float = float(os.getenv("RNA_LIMIAR_DECISAO", "0.5"))
"""Limiar de corte para conversão de probabilidade em classe binária."""


# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL: str = os.getenv("RNA_LOG_LEVEL", "INFO")


def get_logger(name: str) -> logging.Logger:
    """Fábrica de loggers com formato consistente para o pacote."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger
