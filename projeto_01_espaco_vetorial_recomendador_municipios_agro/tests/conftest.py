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
class FakeSidraClient:
    """Cliente falso do sidrapy usado nos testes.

    Gera dinamicamente respostas plausíveis baseadas nos códigos IBGE
    de município recebidos em `ibge_territorial_code`. Isso permite que
    a validação de cobertura pós-download do dataset.py funcione
    corretamente em testes: como os códigos retornados são exatamente
    os solicitados, a cobertura é 100%.

    Cada chamada é registrada em `last_call` para inspeção e o total de
    chamadas em `call_count` — útil para validar o chunking.
    """

    # Mapeamento fixo tipos de rebanho reproduzindo estrutura real do SIDRA
    _REBANHOS_DEFAULT: tuple[tuple[str, str], ...] = (
        ("2670", "Bovino"),
        ("32794", "Suíno - total"),
    )

    def __init__(self, rebanhos: tuple[tuple[str, str], ...] | None = None) -> None:
        self._rebanhos = rebanhos or self._REBANHOS_DEFAULT
        self.last_call: dict[str, Any] | None = None
        self.call_count: int = 0

    def get_table(self, **kwargs: Any) -> pd.DataFrame:
        """Simula chamada SIDRA gerando DataFrame com os códigos solicitados.

        Ecoa de volta os códigos passados em `ibge_territorial_code` (CSV
        de códigos IBGE), gerando N linhas para cada município (uma por
        tipo de rebanho em `_rebanhos`).
        """
        self.last_call = kwargs
        self.call_count += 1

        codigos_csv = kwargs.get("ibge_territorial_code", "")
        codigos = [c.strip() for c in codigos_csv.split(",") if c.strip()]

        if not codigos:
            return pd.DataFrame()

        linhas = []
        for cod in codigos:
            for tipo_cod, tipo_nome in self._rebanhos:
                linhas.append(
                    {
                        "NC": "6",
                        "NN": "Município",
                        "MC": "24",
                        "MN": "Cabeças",
                        "V": "100",
                        "D1C": cod,
                        "D1N": f"Municipio {cod}",
                        "D2C": "2024",
                        "D2N": "2024",
                        "D3C": "105",
                        "D3N": "Efetivo dos rebanhos",
                        "D4C": tipo_cod,
                        "D4N": tipo_nome,
                    }
                )
        return pd.DataFrame(linhas)


@pytest.fixture
def fake_sidra_client() -> FakeSidraClient:
    return FakeSidraClient()


class FakeSidraEmpty:
    """Sidra falso que retorna DataFrame vazio (simula falha silenciosa da API)."""

    def get_table(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


@pytest.fixture
def fake_sidra_empty() -> FakeSidraEmpty:
    return FakeSidraEmpty()


# =============================================================================
# Fixtures para features.py (PPM em formato real do sidrapy)
# =============================================================================
def _ppm_row(
    municipio_id: int,
    municipio_nome_uf: str,
    tipo_rebanho_cod: str,
    tipo_rebanho_nome: str,
    valor: str,
    ano: str = "2024",
) -> dict[str, str]:
    """Constrói uma linha PPM no formato exato do sidrapy header='n'.

    Mapeamento (descoberto empiricamente na Fase 1.B):
      NC=nível código, NN=nível nome, MC=medida código, MN=medida nome,
      V=valor, D1=município, D2=ano, D3=variável, D4=tipo de rebanho.
    """
    return {
        "NC": "6",
        "NN": "Município",
        "MC": "24",
        "MN": "Cabeças",
        "V": valor,
        "D1C": str(municipio_id),
        "D1N": municipio_nome_uf,
        "D2C": ano,
        "D2N": ano,
        "D3C": "105",
        "D3N": "Efetivo dos rebanhos",
        "D4C": tipo_rebanho_cod,
        "D4N": tipo_rebanho_nome,
    }


@pytest.fixture
def ppm_raw_minimo() -> pd.DataFrame:
    """PPM sintética com 3 municípios × 3 tipos de rebanho.

    Inclui valor '-' (zero por convenção IBGE), whitespace irregular, e
    o tipo redundante 'Suíno - matrizes desuínos' para testar filtragem.
    """
    linhas = [
        # Município 1200013 (Acrelândia-AC): bovinocultura alta, suínos, avicultura
        _ppm_row(1200013, "Acrelândia (AC)", "2670", "Bovino", "50000"),
        _ppm_row(1200013, "Acrelândia (AC)", "32794", "Suíno - total", "3000"),
        _ppm_row(1200013, "Acrelândia (AC)", "32796", "Galináceos - total", "80000"),
        _ppm_row(1200013, "Acrelândia (AC)", "32795", "Suíno - matrizes desuínos", "400"),
        # Município 3111606 (Cambuquira-MG): bovinos médios, sem suínos
        _ppm_row(3111606, "Cambuquira (MG)", "2670", " Bovino ", "10000"),  # espaços!
        _ppm_row(3111606, "Cambuquira (MG)", "32796", "Galináceos - total", "5000"),
        _ppm_row(3111606, "Cambuquira (MG)", "32794", "Suíno - total", "-"),  # zero IBGE
        # Município 3550308 (São Paulo-SP): urbano, sem produção
        _ppm_row(3550308, "São Paulo (SP)", "2670", "Bovino", "-"),
        _ppm_row(3550308, "São Paulo (SP)", "32794", "Suíno - total", "-"),
    ]
    return pd.DataFrame(linhas)


@pytest.fixture
def localidades_minimo() -> pd.DataFrame:
    """Localidades sintéticas cobrindo os municípios da fixture ppm_raw_minimo.

    Inclui um município adicional (Manaus-AM) que não aparece na PPM, para
    exercitar o merge com municípios "órfãos" (sem dados PPM → recebem zero).
    """
    from rec_agro_br import dataset

    locs_raw = [
        {
            "id": 1200013,
            "nome": "Acrelândia",
            "microrregiao": {
                "id": 12002,
                "nome": "Rio Branco",
                "mesorregiao": {
                    "id": 1201,
                    "nome": "Vale do Acre",
                    "UF": {
                        "id": 12,
                        "sigla": "AC",
                        "nome": "Acre",
                        "regiao": {"id": 1, "sigla": "N", "nome": "Norte"},
                    },
                },
            },
        },
        {
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
                        "regiao": {"id": 3, "sigla": "SE", "nome": "Sudeste"},
                    },
                },
            },
        },
        {
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
                        "regiao": {"id": 3, "sigla": "SE", "nome": "Sudeste"},
                    },
                },
            },
        },
        # Município sem correspondência na PPM (órfão) — testa left join
        {
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
                        "regiao": {"id": 1, "sigla": "N", "nome": "Norte"},
                    },
                },
            },
        },
    ]
    return dataset._localidades_to_dataframe(locs_raw)


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
