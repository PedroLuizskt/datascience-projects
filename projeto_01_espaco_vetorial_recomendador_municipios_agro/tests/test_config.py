"""Smoke tests do módulo `rec_agro_br.config`.

Estes testes validam a saúde básica do setup do projeto:

- O pacote `rec_agro_br` é importável.
- A raiz do projeto foi detectada corretamente (contém pyproject.toml).
- Os paths canônicos apontam para dentro da raiz do projeto.
- Constantes essenciais estão definidas com valores plausíveis.
- A criação idempotente de diretórios funciona.

Se qualquer teste aqui falhar, algo do ambiente ou da estrutura de pastas
está errado — o pipeline propriamente dito nem chegou a ser exercitado.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rec_agro_br import __version__
from rec_agro_br import config


# =============================================================================
# Importação e versão
# =============================================================================
class TestImport:
    """Testes de sanidade sobre a importação do pacote."""

    def test_pacote_importa(self) -> None:
        """O pacote rec_agro_br importa sem erros."""
        import rec_agro_br  # noqa: F401

    def test_versao_definida(self) -> None:
        """A versão está definida e segue formato semântico ou pré-lançamento."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Aceita formatos como "0.1.0", "0.1.0a0", "1.2.3.dev4"
        assert __version__[0].isdigit()


# =============================================================================
# Detecção de raiz e paths
# =============================================================================
class TestPaths:
    """Testes sobre os paths configurados."""

    def test_project_root_existe(self) -> None:
        """A raiz do projeto foi detectada e é um diretório existente."""
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()

    def test_project_root_contem_pyproject(self) -> None:
        """A raiz detectada contém o pyproject.toml (âncora canônica)."""
        assert (config.PROJECT_ROOT / "pyproject.toml").is_file()

    @pytest.mark.parametrize(
        "path_attr",
        [
            "DATA_DIR",
            "RAW_DATA_DIR",
            "INTERIM_DATA_DIR",
            "PROCESSED_DATA_DIR",
            "EXTERNAL_DATA_DIR",
            "REPORTS_DIR",
            "FIGURES_DIR",
            "NOTEBOOKS_DIR",
            "DOCS_DIR",
            "APOSTILA_DIR",
        ],
    )
    def test_path_e_pathlib(self, path_attr: str) -> None:
        """Todos os paths declarados são objetos Path do pathlib."""
        value = getattr(config, path_attr)
        assert isinstance(value, Path), f"{path_attr} não é um Path"

    @pytest.mark.parametrize(
        "path_attr",
        [
            "DATA_DIR",
            "INTERIM_DATA_DIR",
            "PROCESSED_DATA_DIR",
            "EXTERNAL_DATA_DIR",
            "REPORTS_DIR",
            "FIGURES_DIR",
            "NOTEBOOKS_DIR",
            "DOCS_DIR",
            "APOSTILA_DIR",
        ],
    )
    def test_path_dentro_da_raiz(self, path_attr: str) -> None:
        """Todos os paths canônicos apontam para dentro da raiz do projeto."""
        value: Path = getattr(config, path_attr)
        try:
            value.resolve().relative_to(config.PROJECT_ROOT.resolve())
        except ValueError:
            pytest.fail(
                f"{path_attr}={value} não está dentro de "
                f"PROJECT_ROOT={config.PROJECT_ROOT}"
            )


# =============================================================================
# Constantes de API SIDRA
# =============================================================================
class TestSidraConstantes:
    """Verifica que as constantes de acesso à API SIDRA estão sadias."""

    def test_url_base_https(self) -> None:
        """A URL base da API SIDRA é HTTPS e aponta para o IBGE."""
        assert config.SIDRA_API_BASE.startswith("https://")
        assert "ibge.gov.br" in config.SIDRA_API_BASE

    def test_url_localidades_https(self) -> None:
        """A URL base de Localidades é HTTPS e aponta para o IBGE."""
        assert config.IBGE_LOCALIDADES_BASE.startswith("https://")
        assert "ibge.gov.br" in config.IBGE_LOCALIDADES_BASE

    def test_tabela_ppm_e_string_numerica(self) -> None:
        """O código da tabela PPM é uma string com dígitos."""
        assert isinstance(config.PPM_TABLE_CODE, str)
        assert config.PPM_TABLE_CODE.isdigit()

    def test_variavel_ppm_e_string_numerica(self) -> None:
        """O código da variável PPM é uma string com dígitos."""
        assert isinstance(config.PPM_VARIABLE_CODE, str)
        assert config.PPM_VARIABLE_CODE.isdigit()

    def test_nivel_territorial_municipio(self) -> None:
        """Nível territorial de município na API SIDRA é '6'."""
        assert config.SIDRA_TERRITORIAL_LEVEL_MUNICIPIO == "6"

    def test_ppm_ano_valido_ou_none(self) -> None:
        """PPM_ANO é None ou um ano razoável (entre 1974 e ano corrente + 1)."""
        from datetime import datetime

        if config.PPM_ANO is not None:
            assert isinstance(config.PPM_ANO, int)
            assert 1974 <= config.PPM_ANO <= datetime.now().year + 1


# =============================================================================
# Parâmetros gerais
# =============================================================================
class TestParametrosGerais:
    """Verifica parâmetros gerais lidos do .env ou dos defaults."""

    def test_log_level_valido(self) -> None:
        """LOG_LEVEL é um nível de log reconhecido."""
        assert config.LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_http_timeout_positivo(self) -> None:
        """Timeout HTTP é um inteiro positivo razoável."""
        assert isinstance(config.HTTP_TIMEOUT, int)
        assert 1 <= config.HTTP_TIMEOUT <= 600

    def test_random_seed_e_inteiro(self) -> None:
        """A semente aleatória é um inteiro."""
        assert isinstance(config.RANDOM_SEED, int)


# =============================================================================
# Parâmetros de vetorização
# =============================================================================
class TestParametrosVetorizacao:
    """Verifica parâmetros de vetorização do recomendador."""

    def test_max_features_positivo(self) -> None:
        """max_features do CountVectorizer é positivo."""
        assert config.COUNT_VECTORIZER_MAX_FEATURES > 0

    def test_stemmer_configurado(self) -> None:
        """Stemmer configurado é uma string não vazia."""
        assert isinstance(config.STEMMER_NAME, str)
        assert len(config.STEMMER_NAME) > 0

    def test_top_k_default_positivo(self) -> None:
        """Top-K default do recomendador é positivo."""
        assert config.DEFAULT_TOP_K > 0


# =============================================================================
# Idempotência de ensure_directories
# =============================================================================
class TestEnsureDirectories:
    """Testes de idempotência da função de criação de diretórios."""

    def test_ensure_directories_e_idempotente(self) -> None:
        """Chamar ensure_directories duas vezes seguidas não gera erro."""
        config.ensure_directories()
        config.ensure_directories()

    def test_ensure_directories_cria_estrutura(self) -> None:
        """Após chamar ensure_directories, os diretórios de dados existem."""
        config.ensure_directories()
        assert config.RAW_DATA_DIR.exists()
        assert config.INTERIM_DATA_DIR.exists()
        assert config.PROCESSED_DATA_DIR.exists()
        assert config.EXTERNAL_DATA_DIR.exists()
        assert config.FIGURES_DIR.exists()


# =============================================================================
# Metadados do projeto
# =============================================================================
class TestMetadados:
    """Testes sobre metadados de identificação do projeto."""

    def test_project_name(self) -> None:
        assert config.PROJECT_NAME == "rec-agro-br"

    def test_project_title_nao_vazio(self) -> None:
        assert isinstance(config.PROJECT_TITLE, str)
        assert len(config.PROJECT_TITLE) > 0

    def test_project_version_bate_com_pacote(self) -> None:
        """A versão exposta pelo pacote bate com a definida no config."""
        assert __version__ == config.PROJECT_VERSION
