"""Configuração central do projeto.

Este módulo consolida todos os paths, constantes e parâmetros configuráveis
do pipeline em um único ponto. Ele deve ser importado por qualquer outro
módulo do pacote que precise saber onde ler ou gravar arquivos, quais
tabelas SIDRA consultar, ou quais parâmetros de vetorização usar.

A filosofia é: constantes que raramente mudam ficam como atributos deste
módulo, parâmetros que o usuário pode querer ajustar por execução vão para
o arquivo `.env` (lido via `python-dotenv`).

Referências
-----------
API SIDRA do IBGE:
    https://servicodados.ibge.gov.br/api/docs/agregados?versao=3
Pesquisa da Pecuária Municipal (PPM), Tabela 3939:
    https://sidra.ibge.gov.br/tabela/3939
Divisão Territorial Brasileira (API Localidades):
    https://servicodados.ibge.gov.br/api/docs/localidades?versao=1
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# Detecção da raiz do projeto
# =============================================================================
# A raiz do projeto é o diretório que contém o arquivo pyproject.toml. Subimos
# pela árvore de diretórios a partir deste arquivo até encontrá-lo. Essa
# abordagem funciona independentemente de onde o pacote seja importado.

def _find_project_root() -> Path:
    """Sobe pela árvore de diretórios procurando o pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume dois níveis acima de src/rec_agro_br/config.py
    return current.parents[2]


PROJECT_ROOT: Path = _find_project_root()

# Carrega .env logo após determinar a raiz, para que variáveis de ambiente
# estejam disponíveis para o restante deste módulo.
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# Estrutura de diretórios (padrão Cookiecutter Data Science)
# =============================================================================
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
APOSTILA_DIR: Path = DOCS_DIR / "apostila"

# Cache do NLTK, se o download for feito localmente
NLTK_DATA_DIR: Path = PROJECT_ROOT / "nltk_data"


def ensure_directories() -> None:
    """Cria todos os diretórios de dados se ainda não existirem.

    Chamada idempotente. Útil no início do pipeline para garantir que
    a estrutura de escrita está pronta antes de qualquer download.
    """
    for directory in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Parâmetros de coleta na API SIDRA (Pesquisa da Pecuária Municipal)
# =============================================================================
# A PPM é publicada anualmente. A tabela 3939 traz o efetivo dos rebanhos por
# tipo (bovinos, bubalinos, suínos, matrizes, equinos, ovinos, caprinos,
# galináceos, galinhas, codornas) para cada município brasileiro.

SIDRA_API_BASE: str = "https://servicodados.ibge.gov.br/api/v3/agregados"
IBGE_LOCALIDADES_BASE: str = "https://servicodados.ibge.gov.br/api/v1/localidades"

# Tabela SIDRA 3939 — Efetivo dos rebanhos, por tipo de rebanho
PPM_TABLE_CODE: str = "3939"

# Variável 105 — Efetivo dos rebanhos (Cabeças)
PPM_VARIABLE_CODE: str = "105"

# Nível territorial 6 — Município
SIDRA_TERRITORIAL_LEVEL_MUNICIPIO: str = "6"

# Classificação 79 — Tipo de rebanho. Categoria 0 (all) traz todos os tipos.
SIDRA_CLASSIFICATION_TIPO_REBANHO: str = "79"

# Ano de referência da PPM. Se PPM_ANO estiver definido no .env, usa-se ele;
# caso contrário, deixa None e o pipeline pega o último disponível
# (a lógica de "last 1" da API SIDRA).
_ppm_ano_env: str | None = os.getenv("PPM_ANO")
PPM_ANO: int | None = int(_ppm_ano_env) if _ppm_ano_env else None

# =============================================================================
# Parâmetros gerais de execução
# =============================================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# =============================================================================
# Parâmetros de vetorização (a serem consumidos por src/rec_agro_br/vectorize.py
# na Fase 1.D). Estão aqui para centralização e para permitir experimentação
# controlada.
# =============================================================================
# Espelhamos o parâmetro max_features do CountVectorizer do projeto DSA original
# (5000). Com ~5570 municípios, esse teto é folgado para nosso vocabulário.
COUNT_VECTORIZER_MAX_FEATURES: int = 5000

# Stemmer para português brasileiro. O projeto DSA original usa PorterStemmer
# (inglês); nossa adaptação para dataset brasileiro exige o RSLPStemmer.
STEMMER_NAME: str = "rslp"

# Número default de vizinhos mais próximos a devolver pelo recomendador.
DEFAULT_TOP_K: int = 5


# =============================================================================
# Metadados do projeto (usados em relatórios e logs)
# =============================================================================
PROJECT_NAME: str = "rec-agro-br"
PROJECT_TITLE: str = (
    "Recomendador de Municípios Brasileiros por Perfil Agropecuário"
)
PROJECT_VERSION: str = "0.1.0a0"
