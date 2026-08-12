"""Fixtures compartilhadas para a suíte pytest.

As fixtures aqui são construídas para simular respostas *plausíveis* das
APIs do IBGE sem precisar acessar a rede. Elas foram desenhadas para
reproduzir a estrutura exata dos JSONs reais, o que garante que o parsing
implementado em `dataset.py` seja exercitado nos mesmos caminhos que
seriam ativados em produção.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# =============================================================================
# Fixtures da API de Localidades v1
# =============================================================================
@pytest.fixture
def loc_municipio_completo() -> dict[str, Any]:
    """Um município com toda a hierarquia territorial preenchida."""
    return {
        "id": 3550308,
        "nome": "São Paulo",
        "microrregiao": {
            "id": 35061,
            "nome": "São Paulo",
            "mesorregiao": {
                "id": 3515,
                "nome": "Metropolitana de São Paulo",
                "UF": {
                    "id": 35,
                    "sigla": "SP",
                    "nome": "São Paulo",
                    "regiao": {
                        "id": 3,
                        "sigla": "SE",
                        "nome": "Sudeste",
                    },
                },
            },
        },
    }


@pytest.fixture
def loc_municipio_cambuquira() -> dict[str, Any]:
    """Cambuquira/MG — município natal do autor, usado como caso concreto."""
    return {
        "id": 3111606,
        "nome": "Cambuquira",
        "microrregiao": {
            "id": 31035,
            "nome": "São Lourenço",
            "mesorregiao": {
                "id": 3110,
                "nome": "Sul/Sudoeste de Minas",
                "UF": {
                    "id": 31,
                    "sigla": "MG",
                    "nome": "Minas Gerais",
                    "regiao": {
                        "id": 3,
                        "sigla": "SE",
                        "nome": "Sudeste",
                    },
                },
            },
        },
    }


@pytest.fixture
def loc_municipio_sem_hierarquia() -> dict[str, Any]:
    """Município degenerado: só nome e id, sem hierarquia — testa robustez."""
    return {
        "id": 9999999,
        "nome": "Município Fantasma",
    }


@pytest.fixture
def loc_lista_ampla(
    loc_municipio_completo: dict[str, Any],
    loc_municipio_cambuquira: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lista com múltiplos municípios de regiões distintas."""
    porto_alegre = {
        "id": 4314902,
        "nome": "Porto Alegre",
        "microrregiao": {
            "id": 43022,
            "nome": "Porto Alegre",
            "mesorregiao": {
                "id": 4306,
                "nome": "Metropolitana de Porto Alegre",
                "UF": {
                    "id": 43,
                    "sigla": "RS",
                    "nome": "Rio Grande do Sul",
                    "regiao": {
                        "id": 4,
                        "sigla": "S",
                        "nome": "Sul",
                    },
                },
            },
        },
    }
    manaus = {
        "id": 1302603,
        "nome": "Manaus",
        "microrregiao": {
            "id": 13007,
            "nome": "Manaus",
            "mesorregiao": {
                "id": 1303,
                "nome": "Centro Amazonense",
                "UF": {
                    "id": 13,
                    "sigla": "AM",
                    "nome": "Amazonas",
                    "regiao": {
                        "id": 1,
                        "sigla": "N",
                        "nome": "Norte",
                    },
                },
            },
        },
    }
    return [loc_municipio_completo, loc_municipio_cambuquira, porto_alegre, manaus]


# =============================================================================
# Fixtures do SIDRA / PPM
# =============================================================================
@pytest.fixture
def sidra_ppm_dataframe_fake() -> pd.DataFrame:
    """DataFrame que imita a resposta do sidrapy para a tabela 3939.

    Estrutura equivalente ao que o sidrapy retorna com `header='n'` (sem
    a linha descritiva). Duas cidades × dois tipos de rebanho.
    """
    return pd.DataFrame(
        {
            "NC": ["6", "6", "6", "6"],
            "NN": ["Município"] * 4,
            "MC": ["3550308", "3550308", "3111606", "3111606"],
            "MN": ["São Paulo", "São Paulo", "Cambuquira", "Cambuquira"],
            "V": ["100", "50", "1000", "200"],
            "D1C": ["2023"] * 4,
            "D1N": ["2023"] * 4,
            "D2C": ["2670", "2680", "2670", "2680"],
            "D2N": ["Bovino", "Suíno", "Bovino", "Suíno"],
            "MU": ["Cabeças"] * 4,
        }
    )


class FakeSidraClient:
    """Cliente falso do sidrapy usado nos testes.

    Registra a última chamada em `last_call` para inspeção nos asserts,
    e devolve um DataFrame fixo. Permite testar `download_ppm_efetivo_rebanhos`
    sem tocar a rede.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe
        self.last_call: dict[str, Any] | None = None
        self.call_count: int = 0

    def get_table(self, **kwargs: Any) -> pd.DataFrame:
        self.last_call = kwargs
        self.call_count += 1
        return self._dataframe.copy()


@pytest.fixture
def fake_sidra_client(sidra_ppm_dataframe_fake: pd.DataFrame) -> FakeSidraClient:
    return FakeSidraClient(sidra_ppm_dataframe_fake)


class FakeSidraEmpty:
    """Sidra falso que retorna DataFrame vazio (simula falha silenciosa da API)."""

    def get_table(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


@pytest.fixture
def fake_sidra_empty() -> FakeSidraEmpty:
    return FakeSidraEmpty()


# =============================================================================
# Fixture de isolamento de diretórios de dados
# =============================================================================
@pytest.fixture
def isolated_data_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redireciona os diretórios de dados do `config` para um tmp isolado.

    Isso garante que os testes de download não escrevam em `data/raw/`
    do projeto real. O tmp_path é distinto por teste, portanto os testes
    ficam isolados uns dos outros também.
    """
    from rec_agro_br import config

    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", tmp_path / "external")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    config.ensure_directories()
    return tmp_path


# =============================================================================
# Localidades de conveniência: municípios sintéticos com múltiplas UFs
# =============================================================================
def _sintetizar_localidades_multi_uf(
    ufs: list[tuple[str, int]],
    municipios_por_uf: int = 3,
) -> list[dict[str, Any]]:
    """Gera lista sintética de municípios distribuídos entre UFs.

    Útil em testes que precisam validar o chunking por UF: passamos várias
    UFs e a fixture devolve municípios sintéticos com códigos IBGE plausíveis.

    Parameters
    ----------
    ufs : list of (sigla_uf, id_uf)
        Ex.: [("SP", 35), ("MG", 31), ("RS", 43)]
    municipios_por_uf : int
        Quantos municípios por UF gerar.
    """
    resultado: list[dict[str, Any]] = []
    id_seq = 1
    for sigla, id_uf in ufs:
        for i in range(municipios_por_uf):
            codigo = int(f"{id_uf}{i:05d}0")
            resultado.append(
                {
                    "id": codigo,
                    "nome": f"Municipio {sigla} {i}",
                    "microrregiao": {
                        "id": id_uf * 1000 + i,
                        "nome": f"Micro {sigla} {i}",
                        "mesorregiao": {
                            "id": id_uf * 100 + i,
                            "nome": f"Meso {sigla} {i}",
                            "UF": {
                                "id": id_uf,
                                "sigla": sigla,
                                "nome": f"UF {sigla}",
                                "regiao": {
                                    "id": 1 + (id_uf % 5),
                                    "sigla": "R",
                                    "nome": "Regiao Fake",
                                },
                            },
                        },
                    },
                }
            )
            id_seq += 1
    return resultado


@pytest.fixture
def localidades_gravadas_uma_uf(
    isolated_data_dirs: Path,
) -> pd.DataFrame:
    """Grava e retorna localidades sintéticas com uma única UF (SP)."""
    from rec_agro_br import dataset

    locs = _sintetizar_localidades_multi_uf([("SP", 35)], municipios_por_uf=3)
    df = dataset._localidades_to_dataframe(locs)
    df.to_parquet(dataset.get_localidades_interim_path(), index=False)
    return df


@pytest.fixture
def localidades_gravadas_multi_uf(
    isolated_data_dirs: Path,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Grava e retorna localidades sintéticas com 4 UFs distintas."""
    from rec_agro_br import dataset

    ufs = [("SP", 35), ("MG", 31), ("RS", 43), ("AM", 13)]
    locs = _sintetizar_localidades_multi_uf(ufs, municipios_por_uf=2)
    df = dataset._localidades_to_dataframe(locs)
    df.to_parquet(dataset.get_localidades_interim_path(), index=False)
    return df, ufs
