"""Coleta e persistência dos datasets brutos do IBGE.

Este módulo implementa a Etapa de *coleta* do pipeline do projeto rec-agro-br.
Ele centraliza toda a interação com as duas APIs externas do IBGE:

- **API de Localidades v1**, que fornece a divisão territorial brasileira
  hierarquizada (município → microrregião → mesorregião → UF → região).
- **API de Dados Agregados v3 (SIDRA)**, da qual extraímos a tabela 3939
  (Pesquisa da Pecuária Municipal — Efetivo dos rebanhos, por tipo de
  rebanho), consumida através do wrapper `sidrapy`.

Filosofia
---------
O módulo opera no modo *write-once, read-many*: baixa cada dataset uma vez,
persiste em disco no formato apropriado (JSON para dados semiestruturados
brutos, Parquet para tabulares), e as funções `load_*` posteriores leem
apenas do disco, sem tocar a rede. Um flag ``force=True`` permite refazer
o download quando a fonte for atualizada ou o cache local corromper.

Notas de reprodutibilidade
--------------------------
A PPM é anual. O ano de referência baixado depende do parâmetro ``ano``
(default: ``config.PPM_ANO``, ou o último disponível se este for ``None``).
Os notebooks que consomem este módulo devem registrar explicitamente qual
ano foi utilizado.

Exemplos
--------
Uso programático em um notebook::

    from rec_agro_br import dataset
    df_loc = dataset.download_localidades()      # 5570 linhas
    df_ppm = dataset.download_ppm_efetivo_rebanhos(ano=2023)

Uso como linha de comando::

    python -m rec_agro_br.dataset all
    python -m rec_agro_br.dataset localidades --force
    python -m rec_agro_br.dataset ppm --ano 2022
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sidrapy
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rec_agro_br import config

logger = logging.getLogger(__name__)


# =============================================================================
# Nomes canônicos de arquivos no disco
# =============================================================================
# Manter os nomes centralizados aqui evita erros de digitação em outros módulos
# e facilita renomeações futuras. Os padrões seguem uma convenção simples:
# `{fonte}_{descricao_curta}[.ext]` e, para dados anuais, incluem o ano no nome.

LOCALIDADES_RAW_JSON: str = "municipios_ibge_localidades.json"
LOCALIDADES_INTERIM_PARQUET: str = "municipios_localidades.parquet"
PPM_RAW_PARQUET_TEMPLATE: str = "ppm_3939_efetivo_rebanhos_{ano}.parquet"

# Ano sentinela quando o usuário deixa PPM_ANO em branco. Sidrapy aceita
# "last 1" como período para pegar o último disponível.
PPM_ULTIMO_DISPONIVEL: str = "last 1"


def get_localidades_raw_path() -> Path:
    """Caminho canônico do JSON bruto da API Localidades."""
    return config.RAW_DATA_DIR / LOCALIDADES_RAW_JSON


def get_localidades_interim_path() -> Path:
    """Caminho canônico do Parquet achatado das localidades."""
    return config.INTERIM_DATA_DIR / LOCALIDADES_INTERIM_PARQUET


def get_ppm_raw_path(ano: int | str) -> Path:
    """Caminho canônico do Parquet bruto da PPM para um dado ano.

    Parameters
    ----------
    ano : int ou str
        Ano de referência (ex.: 2023) ou a string sentinela
        ``PPM_ULTIMO_DISPONIVEL`` quando não se sabe o ano de antemão.
    """
    ano_str = str(ano).replace(" ", "_")
    return config.RAW_DATA_DIR / PPM_RAW_PARQUET_TEMPLATE.format(ano=ano_str)


# =============================================================================
# Sessão HTTP com retry exponencial
# =============================================================================
def _build_session() -> requests.Session:
    """Cria uma requests.Session com retry para falhas transientes.

    A API do IBGE ocasionalmente retorna 502/503 sob carga. Uma sessão com
    retry exponencial mitiga isso automaticamente sem exigir tratamento
    manual em cada função de download.

    Returns
    -------
    requests.Session
        Sessão configurada com HTTPAdapter e política de retry.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# =============================================================================
# Localidades (API v1 do IBGE)
# =============================================================================
def _flatten_localidade(loc: dict[str, Any]) -> dict[str, Any]:
    """Achata um único registro de município da API v1 em um dicionário plano.

    A API retorna cada município como um objeto profundamente aninhado:
    ``município → microrregião → mesorregião → UF → região``. Para uso em um
    DataFrame, precisamos das colunas em um único nível. Esta função extrai
    de forma resiliente cada campo, tolerando níveis ausentes (retorna ``None``).

    Parameters
    ----------
    loc : dict
        Dicionário JSON retornado pela API para um município.

    Returns
    -------
    dict
        Dicionário plano com as chaves canônicas do dataset.
    """
    micro = loc.get("microrregiao") or {}
    meso = micro.get("mesorregiao") or {}
    uf = meso.get("UF") or {}
    regiao = uf.get("regiao") or {}
    return {
        "id_municipio": loc.get("id"),
        "nome_municipio": loc.get("nome"),
        "id_microrregiao": micro.get("id"),
        "nome_microrregiao": micro.get("nome"),
        "id_mesorregiao": meso.get("id"),
        "nome_mesorregiao": meso.get("nome"),
        "id_uf": uf.get("id"),
        "sigla_uf": uf.get("sigla"),
        "nome_uf": uf.get("nome"),
        "id_regiao": regiao.get("id"),
        "sigla_regiao": regiao.get("sigla"),
        "nome_regiao": regiao.get("nome"),
    }


def _localidades_to_dataframe(
    localidades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Converte a lista bruta de municípios da API em DataFrame achatado.

    Parameters
    ----------
    localidades : list of dict
        Lista de dicionários exatamente como retornada pela API do IBGE.

    Returns
    -------
    pandas.DataFrame
        DataFrame com 12 colunas planas por município. Colunas de ID vêm
        como inteiros nullable (Int64) para robustez a dados faltantes.
    """
    if not localidades:
        raise ValueError(
            "A lista de localidades está vazia. A API não retornou dados."
        )
    records = [_flatten_localidade(loc) for loc in localidades]
    df = pd.DataFrame.from_records(records)

    # Tipagem explícita: IDs como Int64 (nullable), textos como string
    int_cols = [
        "id_municipio",
        "id_microrregiao",
        "id_mesorregiao",
        "id_uf",
        "id_regiao",
    ]
    for col in int_cols:
        df[col] = df[col].astype("Int64")

    str_cols = [c for c in df.columns if c not in int_cols]
    for col in str_cols:
        df[col] = df[col].astype("string")

    return df


def download_localidades(
    force: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Baixa a lista completa de municípios brasileiros da API do IBGE.

    O JSON bruto é salvo em ``data/raw/`` e o DataFrame achatado em
    ``data/interim/`` como Parquet. Se ambos os arquivos existirem e
    ``force=False``, a função lê do disco sem tocar a rede.

    Parameters
    ----------
    force : bool
        Se ``True``, refaz o download mesmo que o cache local exista.
    session : requests.Session, optional
        Sessão HTTP para usar. Útil para injeção em testes. Se ``None``,
        cria uma sessão com retry padrão.

    Returns
    -------
    pandas.DataFrame
        DataFrame com um município por linha e 12 colunas planas.

    Raises
    ------
    requests.HTTPError
        Se a API retornar código de erro persistente após retries.
    ValueError
        Se a resposta da API vier vazia ou malformada.
    """
    config.ensure_directories()
    raw_path = get_localidades_raw_path()
    interim_path = get_localidades_interim_path()

    if not force and raw_path.exists() and interim_path.exists():
        logger.info(
            "[CACHE] Localidades já baixadas. Lendo de %s. "
            "Use force=True para refazer.",
            interim_path,
        )
        return pd.read_parquet(interim_path)

    url = f"{config.IBGE_LOCALIDADES_BASE}/municipios"
    logger.info("[HTTP] GET %s", url)

    sess = session or _build_session()
    resp = sess.get(url, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    localidades: list[dict[str, Any]] = resp.json()

    logger.info("[OK] Recebidos %d municípios da API do IBGE", len(localidades))

    # Persiste JSON bruto (imutável) e Parquet achatado (para consumo)
    raw_path.write_text(
        json.dumps(localidades, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[IO] JSON bruto salvo em %s", raw_path)

    df = _localidades_to_dataframe(localidades)
    df.to_parquet(interim_path, index=False)
    logger.info(
        "[IO] Parquet achatado salvo em %s (shape=%s)",
        interim_path,
        df.shape,
    )

    return df


def load_localidades() -> pd.DataFrame:
    """Carrega o DataFrame de localidades já baixado previamente.

    Returns
    -------
    pandas.DataFrame
        DataFrame carregado do Parquet em ``data/interim/``.

    Raises
    ------
    FileNotFoundError
        Se o Parquet ainda não foi gerado (rodar `download_localidades`
        antes, ou o CLI `python -m rec_agro_br.dataset localidades`).
    """
    interim_path = get_localidades_interim_path()
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Parquet de localidades não encontrado em {interim_path}. "
            "Rode primeiro: python -m rec_agro_br.dataset localidades"
        )
    return pd.read_parquet(interim_path)


# =============================================================================
# PPM — Tabela 3939 do SIDRA (Efetivo dos rebanhos)
# =============================================================================
def _resolver_periodo_ppm(ano: int | None) -> tuple[str, int | str]:
    """Resolve o parâmetro `period` da chamada sidrapy e a chave de cache.

    Se ``ano`` for ``None``, usa a sentinela ``PPM_ULTIMO_DISPONIVEL``
    ("last 1"), o que faz o SIDRA devolver o ano mais recente disponível.

    Parameters
    ----------
    ano : int or None
        Ano de referência. Se ``None``, usa o último disponível.

    Returns
    -------
    tuple of (str, int or str)
        - período no formato aceito por sidrapy
        - chave para caminho de arquivo (ano numérico ou string sentinela)
    """
    if ano is None:
        return PPM_ULTIMO_DISPONIVEL, PPM_ULTIMO_DISPONIVEL
    return str(ano), ano


def download_ppm_efetivo_rebanhos(
    ano: int | None = None,
    force: bool = False,
    sidra_client: Any | None = None,
) -> pd.DataFrame:
    """Baixa a tabela 3939 (PPM/SIDRA) para todos os municípios brasileiros.

    A tabela retorna, em formato long, o efetivo de cada tipo de rebanho
    (bovinos, bubalinos, suínos, matrizes de suínos, equinos, ovinos,
    caprinos, galináceos, galinhas e codornas) por município para um ano
    de referência.

    Parameters
    ----------
    ano : int, optional
        Ano de referência. Se ``None``, usa ``config.PPM_ANO``; se este
        também for ``None``, pede o último disponível na API.
    force : bool
        Se ``True``, refaz o download mesmo com cache existente.
    sidra_client : object, optional
        Cliente com método ``get_table(**kwargs)``. Injeção para testes.
        Se ``None``, usa o módulo ``sidrapy`` real.

    Returns
    -------
    pandas.DataFrame
        DataFrame em formato long conforme retornado pelo SIDRA, com o
        cabeçalho descritivo removido.

    Raises
    ------
    RuntimeError
        Se a chamada ao SIDRA falhar após retries internos do sidrapy.
    """
    config.ensure_directories()

    ano_efetivo = ano if ano is not None else config.PPM_ANO
    periodo, cache_key = _resolver_periodo_ppm(ano_efetivo)
    raw_path = get_ppm_raw_path(cache_key)

    if not force and raw_path.exists():
        logger.info(
            "[CACHE] PPM (%s) já baixada. Lendo de %s. "
            "Use force=True para refazer.",
            periodo,
            raw_path,
        )
        return pd.read_parquet(raw_path)

    client = sidra_client if sidra_client is not None else sidrapy
    logger.info(
        "[SIDRA] Baixando tabela %s (PPM Efetivo Rebanhos), "
        "nível=%s, período=%s. Isso pode demorar 30-90s.",
        config.PPM_TABLE_CODE,
        config.SIDRA_TERRITORIAL_LEVEL_MUNICIPIO,
        periodo,
    )

    df = client.get_table(
        table_code=config.PPM_TABLE_CODE,
        territorial_level=config.SIDRA_TERRITORIAL_LEVEL_MUNICIPIO,
        ibge_territorial_code="all",
        variable=config.PPM_VARIABLE_CODE,
        classifications={config.SIDRA_CLASSIFICATION_TIPO_REBANHO: "all"},
        period=periodo,
        header="n",  # descarta linha descritiva de cabeçalho
        format="pandas",
    )

    if df is None or df.empty:
        raise RuntimeError(
            "SIDRA retornou DataFrame vazio para PPM tabela "
            f"{config.PPM_TABLE_CODE}. Verifique a disponibilidade da API."
        )

    logger.info("[OK] PPM recebida, shape=%s", df.shape)

    df.to_parquet(raw_path, index=False)
    logger.info("[IO] Parquet salvo em %s", raw_path)

    return df


def load_ppm_efetivo_rebanhos(ano: int | str) -> pd.DataFrame:
    """Carrega o Parquet bruto da PPM para um ano previamente baixado.

    Parameters
    ----------
    ano : int or str
        Ano de referência ou a sentinela ``PPM_ULTIMO_DISPONIVEL``.

    Returns
    -------
    pandas.DataFrame
        DataFrame long carregado do disco.

    Raises
    ------
    FileNotFoundError
        Se o Parquet ainda não existe.
    """
    raw_path = get_ppm_raw_path(ano)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Parquet da PPM não encontrado em {raw_path}. "
            "Rode primeiro: python -m rec_agro_br.dataset ppm"
        )
    return pd.read_parquet(raw_path)


# =============================================================================
# CLI
# =============================================================================
def _configure_logging() -> None:
    """Configura o logging para o CLI respeitando `config.LOG_LEVEL`."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_localidades(args: argparse.Namespace) -> int:
    df = download_localidades(force=args.force)
    print(f"\n[OK] Localidades: {len(df)} municípios")
    print(f"     Colunas: {list(df.columns)}")
    print(df.head(5).to_string(index=False))
    return 0


def _cmd_ppm(args: argparse.Namespace) -> int:
    df = download_ppm_efetivo_rebanhos(ano=args.ano, force=args.force)
    print(f"\n[OK] PPM: shape={df.shape}")
    print(f"     Colunas: {list(df.columns)}")
    print(df.head(5).to_string(index=False))
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    _cmd_localidades(args)
    _cmd_ppm(args)
    print("\n[OK] Todos os downloads concluídos.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rec_agro_br.dataset",
        description=(
            "CLI de coleta de dados brutos do IBGE para o projeto rec-agro-br. "
            "Baixa a divisão territorial brasileira (API Localidades) e a "
            "Pesquisa da Pecuária Municipal (SIDRA tabela 3939)."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{localidades,ppm,all}",
    )

    p_loc = subparsers.add_parser(
        "localidades",
        help="Baixa a lista de municípios da API de Localidades v1.",
    )
    p_loc.add_argument(
        "--force",
        action="store_true",
        help="Refaz o download mesmo com cache existente.",
    )
    p_loc.set_defaults(func=_cmd_localidades)

    p_ppm = subparsers.add_parser(
        "ppm",
        help="Baixa a tabela 3939 (Efetivo dos rebanhos por município).",
    )
    p_ppm.add_argument(
        "--ano",
        type=int,
        default=None,
        help=(
            "Ano de referência (ex.: 2023). "
            "Se omitido, usa PPM_ANO do .env ou o último disponível."
        ),
    )
    p_ppm.add_argument(
        "--force",
        action="store_true",
        help="Refaz o download mesmo com cache existente.",
    )
    p_ppm.set_defaults(func=_cmd_ppm)

    p_all = subparsers.add_parser(
        "all",
        help="Baixa tudo: localidades + PPM.",
    )
    p_all.add_argument(
        "--ano",
        type=int,
        default=None,
        help=(
            "Ano de referência para a PPM. Se omitido, usa PPM_ANO do .env "
            "ou o último disponível."
        ),
    )
    p_all.add_argument(
        "--force",
        action="store_true",
        help="Refaz o download mesmo com cache existente.",
    )
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (requests.HTTPError, requests.ConnectionError) as e:
        logger.error("[ERRO] Falha de rede ao contatar API do IBGE: %s", e)
        return 2
    except FileNotFoundError as e:
        logger.error("[ERRO] Arquivo não encontrado: %s", e)
        return 3
    except Exception as e:
        logger.exception("[ERRO] Falha inesperada: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
