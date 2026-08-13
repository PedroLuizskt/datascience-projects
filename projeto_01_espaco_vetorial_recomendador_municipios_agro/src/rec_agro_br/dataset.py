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

Limitação da API SIDRA e estratégia de lotes
---------------------------------------------
A API do SIDRA impõe um limite hard de **50.000 valores por request**. Como
a tabela 3939 com todos os municípios × todos os tipos de rebanho retorna
cerca de 55.700 valores (5570 × 10), pedir tudo em uma única chamada estoura
o teto. A estratégia adotada é fatiar o download por Unidade da Federação:
27 requests independentes, cada uma trazendo os municípios de uma UF,
resultando em no máximo ~6.500 valores por chamada (para SP, a UF com mais
municípios). Além de respeitar o limite, essa estratégia produz progressão
natural (uma UF por vez), permite retry granular em caso de falha isolada
e cria uma dependência limpa e explícita entre os dois passos do pipeline:
o download da PPM consome o resultado do download das localidades.

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
    df_loc = dataset.download_localidades()      # 5571 linhas
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
from tqdm import tqdm
from urllib3.util.retry import Retry

from rec_agro_br import config

logger = logging.getLogger(__name__)


# =============================================================================
# Nomes canônicos de arquivos no disco
# =============================================================================
LOCALIDADES_RAW_JSON: str = "municipios_ibge_localidades.json"
LOCALIDADES_INTERIM_PARQUET: str = "municipios_localidades.parquet"
PPM_RAW_PARQUET_TEMPLATE: str = "ppm_3939_efetivo_rebanhos_{ano}.parquet"

# Ano sentinela quando o usuário deixa PPM_ANO em branco. Sidrapy aceita
# "last 1" como período para pegar o último disponível.
PPM_ULTIMO_DISPONIVEL: str = "last 1"

# Limite hard da API SIDRA por request. Documentado apenas parcialmente na
# página oficial da API; descoberto empiricamente durante os testes de
# integração da Fase 1.B.
SIDRA_MAX_VALORES_POR_REQUEST: int = 50_000


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
    """Cria uma requests.Session com retry para falhas transientes."""
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
    """Converte a lista bruta de municípios da API em DataFrame achatado."""
    if not localidades:
        raise ValueError(
            "A lista de localidades está vazia. A API não retornou dados."
        )
    records = [_flatten_localidade(loc) for loc in localidades]
    df = pd.DataFrame.from_records(records)

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
    """Baixa a lista completa de municípios brasileiros da API do IBGE."""
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
    """Carrega o DataFrame de localidades já baixado previamente."""
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
    """Resolve o parâmetro `period` da chamada sidrapy e a chave de cache."""
    if ano is None:
        return PPM_ULTIMO_DISPONIVEL, PPM_ULTIMO_DISPONIVEL
    return str(ano), ano


# Tamanho default de cada lote de municípios enviados por request ao SIDRA.
# Escolha empírica que balanceia dois limites: (i) o limite de 50k valores da
# API SIDRA (500 × 10 rebanhos = 5000 valores, folgado); (ii) o limite de URL
# de HTTP servers (500 × 7 chars + 499 vírgulas ≈ 4.000 chars, dentro do teto
# universal de ~8 KB). Descoberto empiricamente que o chunking por UF adotado
# na primeira versão da Fase 1.B corrompia silenciosamente MG (853 municípios,
# URL ~7.7 KB) e SP (645 municípios, URL ~5.8 KB): o SIDRA aceitava a request
# mas devolvia dados de OUTROS municípios, não os solicitados. O chunking por
# tamanho fixo elimina essa heterogeneidade problemática.
SIDRA_DEFAULT_CHUNK_SIZE: int = 500


def _agrupar_municipios_por_uf(
    df_localidades: pd.DataFrame,
) -> dict[str, list[str]]:
    """Agrupa códigos IBGE de municípios por sigla da UF.

    Mantida na base de código como utilitário auxiliar (não é mais usada
    pelo pipeline principal de download da PPM, que agora fatia por
    tamanho fixo em vez de por UF; ver :func:`_dividir_em_chunks`).

    A saída é ordenada alfabeticamente por sigla (AC, AL, AM, ..., TO)
    para gerar logs previsíveis. Municípios sem UF associada (não deveria
    ocorrer em dados reais) são silenciosamente descartados.

    Parameters
    ----------
    df_localidades : pandas.DataFrame
        DataFrame com as colunas ``sigla_uf`` e ``id_municipio``, tipicamente
        vindo de ``load_localidades()``.

    Returns
    -------
    dict[str, list[str]]
        Dicionário {sigla_uf: [codigo_municipio, ...]}. Códigos como strings
        para uso direto na chamada sidrapy.
    """
    df = df_localidades[["sigla_uf", "id_municipio"]].dropna()
    resultado: dict[str, list[str]] = {}
    for sigla in sorted(df["sigla_uf"].unique()):
        mask = df["sigla_uf"] == sigla
        codigos = df.loc[mask, "id_municipio"].astype("int64").astype(str).tolist()
        resultado[str(sigla)] = codigos
    return resultado


def _dividir_em_chunks(
    codigos: list[str],
    chunk_size: int = SIDRA_DEFAULT_CHUNK_SIZE,
) -> list[list[str]]:
    """Divide uma lista de códigos em sub-listas de tamanho fixo.

    O último chunk pode ser menor que ``chunk_size``. Preserva a ordem
    original da lista.

    Parameters
    ----------
    codigos : list of str
        Lista completa de códigos IBGE (ex.: todos os municípios do Brasil).
    chunk_size : int
        Máximo de códigos por chunk.

    Returns
    -------
    list of list of str
        Ex.: para 5571 códigos e chunk_size=500, retorna 12 chunks
        (11 de 500 + 1 de 71).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size deve ser positivo, recebido {chunk_size}")
    return [codigos[i : i + chunk_size] for i in range(0, len(codigos), chunk_size)]


def _validar_cobertura_ppm(
    df: pd.DataFrame,
    municipios_solicitados: set[str],
    limite_aviso: float = 0.02,
) -> None:
    """Verifica que a resposta consolidada cobre os municípios solicitados.

    Este check foi introduzido após descoberta de que o SIDRA pode aceitar
    requests grandes mas retornar dados de municípios *diferentes* dos
    solicitados (falha silenciosa por URL truncada). A cobertura real do
    dado versus a esperada é a verificação de sanidade que pega esse tipo
    de bug antes de contaminar as fases seguintes do pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Consolidado de todos os chunks concatenados. Precisa ter coluna
        ``D1C`` (código IBGE do município).
    municipios_solicitados : set of str
        Conjunto dos códigos que deveriam estar presentes.
    limite_aviso : float
        Fração de perda acima da qual um aviso é logado. Default 2%,
        que corresponde a algumas dezenas de municípios em 5571 e ainda
        pode ser lacuna legítima do IBGE para municípios pequenos.

    Raises
    ------
    RuntimeError
        Se a cobertura for catastroficamente baixa (< 50% dos solicitados),
        indicando falha sistêmica que exige investigação antes de prosseguir.
    """
    if "D1C" not in df.columns:
        logger.warning("[AVISO] Coluna D1C ausente, cobertura não validada")
        return

    municipios_recebidos = set(df["D1C"].astype(str).unique())
    faltantes = municipios_solicitados - municipios_recebidos
    fracao_faltante = len(faltantes) / max(len(municipios_solicitados), 1)

    if fracao_faltante > 0.5:
        raise RuntimeError(
            f"Cobertura PPM catastroficamente baixa: só "
            f"{len(municipios_recebidos)}/{len(municipios_solicitados)} "
            f"municípios retornaram dados ({100*fracao_faltante:.1f}% faltantes). "
            "Provável falha sistêmica na API SIDRA. Verifique o log de chunks "
            "e considere reduzir SIDRA_DEFAULT_CHUNK_SIZE."
        )

    if fracao_faltante > limite_aviso:
        amostra = sorted(faltantes)[:10]
        logger.warning(
            "[AVISO] Cobertura PPM: %d/%d municípios (%.1f%% faltantes). "
            "Amostra de códigos faltantes: %s",
            len(municipios_recebidos),
            len(municipios_solicitados),
            100 * fracao_faltante,
            amostra,
        )
    else:
        logger.info(
            "[OK] Cobertura PPM: %d/%d municípios (%.1f%% faltantes)",
            len(municipios_recebidos),
            len(municipios_solicitados),
            100 * fracao_faltante,
        )


def download_ppm_efetivo_rebanhos(
    ano: int | None = None,
    force: bool = False,
    sidra_client: Any | None = None,
    show_progress: bool | None = None,
    chunk_size: int = SIDRA_DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Baixa a tabela 3939 (PPM/SIDRA) para todos os municípios brasileiros.

    O download é feito em lotes de :data:`SIDRA_DEFAULT_CHUNK_SIZE` municípios
    (default 500), para contornar dois limites simultâneos da API SIDRA:
    o de 50.000 valores por request (descoberto empiricamente na Fase 1.B)
    e o de tamanho de URL (descoberto empiricamente após bug em MG e SP na
    versão inicial que fatiava por UF). O chunking por tamanho fixo garante
    URLs uniformes e pequenas (~4 KB), abaixo do teto universal HTTP.

    Após concatenar todos os chunks, é executada uma validação de cobertura
    que compara municípios solicitados vs recebidos. Se a perda for
    catastrófica (>50%), uma exceção interrompe o pipeline; se for
    moderada (>2%), um aviso é logado. Isso protege contra falhas
    silenciosas da API que retorna dados sem erro mas com cobertura errada.

    Depende de que ``download_localidades()`` já tenha sido executado, pois
    consulta o Parquet das localidades para saber quais códigos IBGE mandar.
    Se as localidades ainda não estiverem em disco, esta função as baixa
    automaticamente antes de prosseguir.

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
    show_progress : bool, optional
        Se ``True``, mostra barra de progresso via tqdm. Se ``None``
        (default), desabilita quando um ``sidra_client`` foi injetado
        (indicativo de teste) e habilita caso contrário.
    chunk_size : int
        Máximo de municípios por request SIDRA. Default 500 (validado
        empiricamente). Reduza se houver problemas de URL grande ou aumente
        se souber que sua rede aguenta URLs maiores.

    Returns
    -------
    pandas.DataFrame
        DataFrame em formato long conforme retornado pelo SIDRA, com o
        cabeçalho descritivo removido (``header='n'``) e todos os chunks
        concatenados.

    Raises
    ------
    RuntimeError
        Se nenhum chunk retornar dados, ou se a cobertura pós-download
        for catastroficamente baixa (<50% dos municípios solicitados).
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

    # Garante que localidades está em disco (necessária para chunking)
    if not get_localidades_interim_path().exists():
        logger.info(
            "[INFO] Localidades ausentes em disco. "
            "Baixando primeiro como pré-requisito da PPM."
        )
        download_localidades()

    df_loc = load_localidades()
    todos_codigos = (
        df_loc["id_municipio"]
        .dropna()
        .astype("int64")
        .astype(str)
        .tolist()
    )
    chunks = _dividir_em_chunks(todos_codigos, chunk_size=chunk_size)

    logger.info(
        "[INFO] Fatiando download PPM em %d lotes de até %d municípios "
        "(total %d municípios). Limites SIDRA: %d valores/request; "
        "URL estimada por chunk: ~%d chars.",
        len(chunks),
        chunk_size,
        len(todos_codigos),
        SIDRA_MAX_VALORES_POR_REQUEST,
        chunk_size * 8,
    )

    client = sidra_client if sidra_client is not None else sidrapy

    if show_progress is None:
        show_progress = sidra_client is None

    partes: list[pd.DataFrame] = []
    chunks_com_falha: list[int] = []

    iterator = tqdm(
        enumerate(chunks, start=1),
        total=len(chunks),
        desc="Baixando PPM",
        unit="chunk",
        disable=not show_progress,
    )

    for i, chunk in iterator:
        codigos_csv = ",".join(chunk)
        try:
            df_chunk = client.get_table(
                table_code=config.PPM_TABLE_CODE,
                territorial_level=config.SIDRA_TERRITORIAL_LEVEL_MUNICIPIO,
                ibge_territorial_code=codigos_csv,
                variable=config.PPM_VARIABLE_CODE,
                classifications={config.SIDRA_CLASSIFICATION_TIPO_REBANHO: "all"},
                period=periodo,
                header="n",
                format="pandas",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[AVISO] Falha ao baixar chunk %d/%d (%d municípios): %s",
                i,
                len(chunks),
                len(chunk),
                e,
            )
            chunks_com_falha.append(i)
            continue

        if df_chunk is None or df_chunk.empty:
            logger.warning(
                "[AVISO] SIDRA retornou vazio para chunk %d/%d", i, len(chunks)
            )
            chunks_com_falha.append(i)
            continue

        partes.append(df_chunk)
        logger.debug(
            "[OK] Chunk %d/%d: %d linhas recebidas (%d municípios solicitados)",
            i,
            len(chunks),
            len(df_chunk),
            len(chunk),
        )

    if not partes:
        raise RuntimeError(
            f"Nenhum dos {len(chunks)} chunks retornou dados. "
            "Verifique conectividade e disponibilidade da API SIDRA."
        )

    df = pd.concat(partes, ignore_index=True)
    logger.info(
        "[INFO] PPM consolidada: %d linhas de %d chunks bem-sucedidos "
        "(%d falhas: %s)",
        len(df),
        len(chunks) - len(chunks_com_falha),
        len(chunks_com_falha),
        chunks_com_falha or "nenhuma",
    )

    # Validação crítica de cobertura para pegar falhas silenciosas do SIDRA
    _validar_cobertura_ppm(df, set(todos_codigos))

    df.to_parquet(raw_path, index=False)
    logger.info("[IO] Parquet salvo em %s", raw_path)

    return df


def load_ppm_efetivo_rebanhos(ano: int | str) -> pd.DataFrame:
    """Carrega o Parquet bruto da PPM para um ano previamente baixado."""
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
    print(df.head(10).to_string(index=False))
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
