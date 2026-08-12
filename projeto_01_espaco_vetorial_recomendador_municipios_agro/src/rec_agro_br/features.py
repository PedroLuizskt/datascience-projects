"""Feature engineering do dataset de recomendação.

Este módulo implementa a Etapa de *transformação* do pipeline do projeto
rec-agro-br. Consome os artefatos brutos gerados por :mod:`dataset` e
produz o dataset final que será vetorizado na Fase 1.D.

Pipeline
--------
::

    data/raw/ppm_3939_efetivo_rebanhos_last_1.parquet
    data/interim/municipios_localidades.parquet
                          │
                          ▼
             ┌────────────────────────┐
             │    clean_ppm           │  renomear colunas do sidrapy,
             │                        │  tipar valores, tratar '-' e '..'
             └────────────┬───────────┘
                          ▼
             ┌────────────────────────┐
             │    pivot_ppm_wide      │  long → wide (5570 x 8 rebanhos)
             └────────────┬───────────┘
                          ▼
             ┌────────────────────────┐
             │  merge_com_localidades │  join com contexto territorial
             └────────────┬───────────┘
                          ▼
             ┌────────────────────────┐
             │ derive_perfis_agropec  │  categorização por quantis
             │ derive_especializacao  │  atividade dominante
             │ derive_diversidade     │  contagem e lista de atividades
             └────────────┬───────────┘
                          ▼
             ┌────────────────────────┐
             │    build_tags          │  campo textual concatenado
             └────────────┬───────────┘
                          ▼
    data/processed/municipios_features.parquet

Analogia com o projeto DSA original
-----------------------------------
O projeto Cap08 concatenava overview + genres + keywords + cast + crew em
um único campo ``tags`` que era vetorizado pelo :class:`CountVectorizer`.
Este módulo replica exatamente essa estrutura para o domínio agropecuário:

- **overview** (contexto): nome_regiao + sigla_uf + nome_mesorregiao
- **genres** (perfis quantitativos): perfil_bovinocultura, perfil_suinocultura, ...
- **keywords** (rebanhos presentes): lista de atividades presentes
- **cast** (especialização dominante): especializacao_em_bovinocultura, ...
- **crew** (nível de diversidade): sinalizadores de diversidade produtiva

O campo final ``tags`` é uma string separada por espaços, pronta para ser
tokenizada pelo :class:`CountVectorizer` do scikit-learn.

Referências
-----------
IBGE. *Notas técnicas SIDRA — Convenções em tabelas*: o hífen ("-")
significa "dado numérico igual a zero não resultante de arredondamento";
"..." significa "não se aplica dado numérico".

Exemplos
--------
Uso programático::

    from rec_agro_br import features
    df = features.build_features_dataset()  # roda pipeline end-to-end
    print(df[["nome_municipio", "sigla_uf", "tags"]].head())

Uso como linha de comando::

    python -m rec_agro_br.features
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rec_agro_br import config, dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Constantes de mapeamento
# =============================================================================
# Renomeação das colunas cripticas do sidrapy (header='n') para nomes legíveis.
# Após inspeção do dado real, confirmou-se que:
#   NC=nível código, NN=nível nome, MC=medida código, MN=medida nome,
#   V=valor, D1=município (dim 1), D2=ano (dim 2), D3=variável (dim 3),
#   D4=tipo de rebanho (dim 4).
COLUNAS_SIDRA_RENAME: dict[str, str] = {
    "D1C": "id_municipio_str",
    "D1N": "nome_municipio_uf",
    "D2C": "ano_str",
    "D4C": "id_tipo_rebanho",
    "D4N": "tipo_rebanho_raw",
    "V": "valor_str",
}

# Convenções do IBGE para valores em tabelas SIDRA (Notas Técnicas):
#   "-"    = dado numérico igual a zero, não resultante de arredondamento
#   "..."  = não se aplica dado numérico
#   ".."   = não disponível
#   "X"    = dado omitido para não identificar informante
VALORES_ESPECIAIS_IBGE_ZERO: set[str] = {"-"}
VALORES_ESPECIAIS_IBGE_NAN: set[str] = {"...", "..", "X"}

# Mapeamento tipo_rebanho_raw → nome canônico da atividade (para uso em tags).
# Nomes canônicos seguem a convenção -cultura (bovinocultura, suinocultura).
MAPA_TIPOS_REBANHO: dict[str, str] = {
    "Bovino": "bovinocultura",
    "Bubalino": "bubalinocultura",
    "Equino": "equinocultura",
    "Suíno - total": "suinocultura",
    "Caprino": "caprinocultura",
    "Ovino": "ovinocultura",
    "Galináceos - total": "avicultura",
    "Codornas": "coturnicultura",
}

# Tipos que não entram no vetor final (subcategorias redundantes de outras).
# "Suíno - matrizes de suínos" é subconjunto de "Suíno - total".
# "Galináceos - galinhas" é subconjunto de "Galináceos - total".
# Nota: mantemos as variações com typo do próprio IBGE ("desuínos") na lista.
TIPOS_REBANHO_IGNORAR: set[str] = {
    "Suíno - matrizes de suínos",
    "Suíno - matrizes desuínos",
    "Galináceos - galinhas",
}

# Atividades consideradas "principais" para o cálculo de especialização.
ATIVIDADES_PRINCIPAIS: list[str] = [
    "bovinocultura",
    "suinocultura",
    "avicultura",
    "ovinocultura",
    "caprinocultura",
]

# Colunas de contexto territorial usadas no campo de tags.
COLUNAS_CONTEXTO_TERRITORIAL: list[str] = [
    "nome_regiao",
    "sigla_uf",
    "nome_mesorregiao",
]

# Nome canônico do arquivo processado final.
FEATURES_PROCESSED_PARQUET: str = "municipios_features.parquet"


def get_features_processed_path() -> Path:
    """Caminho canônico do Parquet processado final."""
    return config.PROCESSED_DATA_DIR / FEATURES_PROCESSED_PARQUET


# =============================================================================
# Utilitários de normalização
# =============================================================================
def _normalizar_texto(s: object) -> str:
    """Normaliza uma string: strip + colapso de whitespace interno.

    Robusto a valores NaN/None (retorna string vazia).
    """
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def _to_snake(s: object) -> str:
    """Converte texto para snake_case, ignorando NaN.

    "Rio Grande do Sul" -> "rio_grande_do_sul"
    "Sul/Sudoeste de Minas" -> "sul_sudoeste_de_minas"
    """
    if pd.isna(s):
        return ""
    txt = str(s).lower().strip()
    # Remove caracteres não alfanuméricos, mantendo espaços e barras como separadores
    txt = re.sub(r"[^a-z0-9áéíóúâêîôûãõçñü\s/_-]", " ", txt)
    # Substitui separadores por underscore
    txt = re.sub(r"[\s/-]+", "_", txt)
    return txt.strip("_")


# =============================================================================
# Estágio 1: limpeza da PPM
# =============================================================================
def clean_ppm(df_ppm_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpa e tipa o DataFrame bruto vindo do SIDRA.

    Executa quatro operações: renomeia colunas do formato cripto do sidrapy
    para nomes legíveis, normaliza espaços em strings, converte tipos
    (município → int, valor → float com tratamento de convenções IBGE),
    e mapeia os nomes de tipos de rebanho para nomes canônicos de atividades
    ao mesmo tempo em que descarta as subcategorias redundantes.

    Parameters
    ----------
    df_ppm_raw : pandas.DataFrame
        DataFrame como retornado por :func:`dataset.load_ppm_efetivo_rebanhos`.

    Returns
    -------
    pandas.DataFrame
        DataFrame com colunas: ``id_municipio`` (int64),
        ``ano`` (int64), ``atividade`` (str), ``valor`` (Float64).

    Raises
    ------
    ValueError
        Se colunas esperadas do sidrapy não estiverem presentes no DataFrame.
    """
    esperadas = set(COLUNAS_SIDRA_RENAME.keys())
    ausentes = esperadas - set(df_ppm_raw.columns)
    if ausentes:
        raise ValueError(
            f"Colunas esperadas ausentes no DataFrame PPM: {ausentes}. "
            f"Colunas encontradas: {list(df_ppm_raw.columns)}"
        )

    df = df_ppm_raw.rename(columns=COLUNAS_SIDRA_RENAME)
    df = df[list(COLUNAS_SIDRA_RENAME.values())].copy()

    # Normaliza strings: strip e colapso de whitespace
    for col in ["nome_municipio_uf", "tipo_rebanho_raw", "valor_str"]:
        df[col] = df[col].map(_normalizar_texto)

    # Ignora tipos redundantes (matrizes de suínos, galinhas separadas)
    df = df[~df["tipo_rebanho_raw"].isin(TIPOS_REBANHO_IGNORAR)].copy()

    # Mapeia tipo bruto → nome canônico da atividade
    df["atividade"] = df["tipo_rebanho_raw"].map(MAPA_TIPOS_REBANHO)

    # Detecta tipos não mapeados (útil para o log)
    nao_mapeados = df.loc[df["atividade"].isna(), "tipo_rebanho_raw"].unique()
    if len(nao_mapeados) > 0:
        logger.warning(
            "[AVISO] Tipos de rebanho não mapeados descartados: %s",
            list(nao_mapeados),
        )
    df = df[df["atividade"].notna()].copy()

    # Tipa município como int64
    df["id_municipio"] = df["id_municipio_str"].astype("int64")

    # Tipa ano como int64
    df["ano"] = df["ano_str"].astype("int64")

    # Trata valores: zero explícito (-), NaN especial (..., .., X), depois numérico
    valor_series = df["valor_str"]
    valor_series = valor_series.where(~valor_series.isin(VALORES_ESPECIAIS_IBGE_ZERO), "0")
    valor_series = valor_series.where(
        ~valor_series.isin(VALORES_ESPECIAIS_IBGE_NAN), np.nan
    )
    df["valor"] = pd.to_numeric(valor_series, errors="coerce").astype("Float64")

    return df[["id_municipio", "ano", "atividade", "valor"]].reset_index(drop=True)


# =============================================================================
# Estágio 2: pivot long → wide
# =============================================================================
def pivot_ppm_wide(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Transforma o DataFrame de formato long para wide.

    Cada município passa a ocupar uma única linha, com uma coluna por
    atividade (bovinocultura, suinocultura, ..., coturnicultura). Valores
    ausentes (município que não aparece para dado rebanho) são preenchidos
    com zero, consistente com a convenção IBGE para hífen.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Saída de :func:`clean_ppm`, formato long com atividade nas linhas.

    Returns
    -------
    pandas.DataFrame
        DataFrame wide, uma linha por município, colunas de atividades
        preenchidas com zero para ausentes.
    """
    wide = df_clean.pivot_table(
        index="id_municipio",
        columns="atividade",
        values="valor",
        aggfunc="sum",
        observed=True,
    )
    wide = wide.fillna(0).astype("Float64")
    wide.columns.name = None
    return wide.reset_index()


# =============================================================================
# Estágio 3: merge com contexto territorial
# =============================================================================
def merge_com_localidades(
    df_wide: pd.DataFrame,
    df_localidades: pd.DataFrame,
) -> pd.DataFrame:
    """Faz left join do wide de atividades com o dataset de localidades.

    O left join **a partir das localidades** garante que todos os 5571
    municípios brasileiros estejam presentes no output final, mesmo aqueles
    que a PPM não reporta (municípios urbanos sem atividade pecuária ou
    novos municípios ainda sem série histórica). Nesses casos, as colunas
    de atividades ficam preenchidas com zero.

    Parameters
    ----------
    df_wide : pandas.DataFrame
        Saída de :func:`pivot_ppm_wide`.
    df_localidades : pandas.DataFrame
        Saída de :func:`dataset.load_localidades`, com 5571 municípios.

    Returns
    -------
    pandas.DataFrame
        DataFrame com 5571 linhas, colunas de localidades + colunas de
        atividades pecuárias, sem valores ausentes nas atividades.
    """
    atividades = [c for c in df_wide.columns if c != "id_municipio"]

    df = df_localidades.merge(df_wide, on="id_municipio", how="left")

    # Municípios sem correspondência na PPM: atividades = 0
    for atv in atividades:
        df[atv] = df[atv].fillna(0).astype("Float64")

    logger.debug(
        "[MERGE] localidades=%d, wide=%d, resultado=%d, "
        "municípios sem PPM=%d",
        len(df_localidades),
        len(df_wide),
        len(df),
        len(df_localidades) - len(df_wide),
    )
    return df


# =============================================================================
# Estágio 4: features derivadas (perfis, especialização, diversidade)
# =============================================================================
def derive_perfis_agropecuarios(
    df: pd.DataFrame,
    atividades: list[str] | None = None,
    quantis: tuple[float, float] = (0.33, 0.66),
) -> pd.DataFrame:
    """Adiciona colunas ``perfil_<atividade>`` com categorização ordinal.

    Para cada atividade, calcula dois quantis nacionais considerando apenas
    os municípios com valor > 0 (para não distorcer a distribuição com os
    zeros dos municípios sem produção). Cada município é classificado em
    uma de quatro categorias:

    - ``sem_<atividade>``: valor == 0
    - ``baixa_<atividade>``: valor <= quantil inferior
    - ``media_<atividade>``: valor entre os quantis
    - ``alta_<atividade>``: valor > quantil superior

    Essas categorias fazem o papel dos ``genres`` do projeto DSA original:
    são tokens categóricos que descrevem o município e serão contados pelo
    ``CountVectorizer``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame com colunas de atividades numéricas.
    atividades : list of str, optional
        Nomes das colunas de atividade a processar. Se None, usa todas
        do :data:`MAPA_TIPOS_REBANHO`.
    quantis : tuple of two floats
        Percentis inferior e superior. Default: (0.33, 0.66) — tercis.

    Returns
    -------
    pandas.DataFrame
        Cópia do input com colunas ``perfil_<atividade>`` adicionadas.
    """
    if atividades is None:
        atividades = list(MAPA_TIPOS_REBANHO.values())

    df = df.copy()

    for atv in atividades:
        if atv not in df.columns:
            logger.debug("[SKIP] Atividade %s ausente no DataFrame", atv)
            continue

        valores = df[atv].astype("float64")
        positivos = valores[valores > 0]

        if len(positivos) < 3:
            # Não há dados suficientes para quantis: só sem/com
            df[f"perfil_{atv}"] = np.where(valores > 0, f"presente_{atv}", f"sem_{atv}")
            continue

        q_baixo, q_alto = positivos.quantile(list(quantis))
        conditions = [
            valores == 0,
            valores <= q_baixo,
            valores <= q_alto,
        ]
        choices = [f"sem_{atv}", f"baixa_{atv}", f"media_{atv}"]
        df[f"perfil_{atv}"] = np.select(conditions, choices, default=f"alta_{atv}")

    return df


def derive_especializacao(
    df: pd.DataFrame,
    atividades_principais: list[str] | None = None,
) -> pd.DataFrame:
    """Adiciona coluna ``especializacao`` com a atividade dominante.

    Para cada município, calcula em qual atividade principal ele ocupa o
    maior percentil nacional (rank pct) e marca isso como sua especialização.
    Municípios com zero em todas as atividades principais recebem o valor
    especial ``sem_producao_pecuaria``.

    Análogo ao ``crew`` (função dominante) do projeto DSA original.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame com colunas de atividades numéricas.
    atividades_principais : list of str, optional
        Atividades a considerar. Default: :data:`ATIVIDADES_PRINCIPAIS`.

    Returns
    -------
    pandas.DataFrame
        Cópia do input com coluna ``especializacao`` adicionada.
    """
    if atividades_principais is None:
        atividades_principais = ATIVIDADES_PRINCIPAIS

    df = df.copy()
    presentes = [a for a in atividades_principais if a in df.columns]

    if not presentes:
        logger.warning("[AVISO] Nenhuma atividade principal presente no DataFrame")
        df["especializacao"] = "sem_producao_pecuaria"
        return df

    # Percentil nacional por atividade — substituindo zeros por NaN para que
    # municípios sem produção em uma atividade não sejam considerados
    # "especialistas" nela. NaN é ignorado por idxmax naturalmente.
    percentis = pd.DataFrame(index=df.index)
    for atv in presentes:
        valores = df[atv].astype("float64").replace(0, np.nan)
        percentis[atv] = valores.rank(pct=True, method="max")

    # Município com pelo menos alguma produção nas atividades principais
    tem_producao = df[presentes].astype("float64").sum(axis=1) > 0

    # Computa idxmax apenas nas linhas com produção para evitar o
    # FutureWarning do pandas sobre idxmax em linhas 100% NaN.
    atividade_dominante = pd.Series(index=df.index, dtype="object")
    if tem_producao.any():
        atividade_dominante.loc[tem_producao] = percentis.loc[tem_producao].idxmax(
            axis=1
        )

    df["especializacao"] = np.where(
        tem_producao,
        "especializado_em_" + atividade_dominante.astype(str),
        "sem_producao_pecuaria",
    )
    return df


def derive_diversidade(
    df: pd.DataFrame,
    atividades: list[str] | None = None,
) -> pd.DataFrame:
    """Adiciona colunas ``atividades_presentes`` (str) e ``n_atividades`` (int).

    ``atividades_presentes`` é uma string com os nomes das atividades cujo
    valor é > 0, separados por espaço. Faz o papel do ``cast`` (elenco de
    atividades presentes) do projeto DSA original.

    ``n_atividades`` é o total de atividades com valor > 0. Útil para EDA
    mas não entra no vetor final.

    Parameters
    ----------
    df : pandas.DataFrame
    atividades : list of str, optional
        Default: :data:`MAPA_TIPOS_REBANHO` valores.

    Returns
    -------
    pandas.DataFrame
        Cópia do input com as duas colunas adicionadas.
    """
    if atividades is None:
        atividades = list(MAPA_TIPOS_REBANHO.values())

    df = df.copy()
    presentes = [a for a in atividades if a in df.columns]

    if not presentes:
        df["atividades_presentes"] = ""
        df["n_atividades"] = 0
        return df

    # Máscara booleana (linhas × atividades) de "valor > 0"
    mask = df[presentes].astype("float64").gt(0)

    df["n_atividades"] = mask.sum(axis=1).astype("Int64")
    df["atividades_presentes"] = mask.apply(
        lambda row: " ".join(a for a, presente in row.items() if presente),
        axis=1,
    )
    return df


# =============================================================================
# Estágio 5: montagem das "tags" (o campo que será vetorizado)
# =============================================================================
def build_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a coluna ``tags`` concatenando features textuais e categóricas.

    A concatenação segue a mesma lógica do projeto DSA original, que juntava
    overview + genres + keywords + cast + crew em um único campo. Aqui:

    - **overview** vira: nome_regiao + sigla_uf + nome_mesorregiao (contexto)
    - **genres** vira: colunas perfil_<atividade> (categorização quantitativa)
    - **crew** vira: coluna especializacao (atividade dominante)
    - **cast/keywords** vira: coluna atividades_presentes (lista das atividades)

    Todos os tokens são normalizados para snake_case para que o
    :class:`CountVectorizer` os trate como tokens únicos (evita que
    "Rio Grande do Sul" seja quebrado em quatro features).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame já enriquecido pelas funções derive_*.

    Returns
    -------
    pandas.DataFrame
        Cópia do input com coluna ``tags`` (str) adicionada.
    """
    df = df.copy()

    partes: list[pd.Series] = []

    # overview: contexto territorial (snake_case para virar 1 token cada)
    for col in COLUNAS_CONTEXTO_TERRITORIAL:
        if col in df.columns:
            partes.append(df[col].map(_to_snake))

    # genres: perfis quantitativos
    perfil_cols = sorted(c for c in df.columns if c.startswith("perfil_"))
    for col in perfil_cols:
        partes.append(df[col].astype(str))

    # crew: especialização
    if "especializacao" in df.columns:
        partes.append(df["especializacao"].astype(str))

    # cast/keywords: atividades presentes (já é string separada por espaços)
    if "atividades_presentes" in df.columns:
        partes.append(df["atividades_presentes"].astype(str))

    if not partes:
        raise ValueError(
            "Nenhuma feature textual disponível para montar tags. "
            "Verifique se as funções derive_* foram aplicadas."
        )

    # Concatena parte a parte com espaço, colapsa múltiplos espaços em um só
    tags = partes[0].astype(str)
    for parte in partes[1:]:
        tags = tags.str.cat(parte.astype(str), sep=" ")
    tags = tags.str.replace(r"\s+", " ", regex=True).str.strip()

    df["tags"] = tags.astype("string")
    return df


# =============================================================================
# Pipeline end-to-end
# =============================================================================
def build_features_dataset(
    ano: int | str | None = None,
    df_ppm_raw: pd.DataFrame | None = None,
    df_localidades: pd.DataFrame | None = None,
    quantis: tuple[float, float] = (0.33, 0.66),
) -> pd.DataFrame:
    """Executa o pipeline completo de feature engineering.

    Se ``df_ppm_raw`` e ``df_localidades`` forem passados, usa-os
    diretamente (útil para testes). Caso contrário, carrega dos parquets
    em disco. Assume que os downloads foram feitos previamente.

    Parameters
    ----------
    ano : int or str, optional
        Ano de referência da PPM. Se None, usa a sentinela
        :data:`dataset.PPM_ULTIMO_DISPONIVEL`.
    df_ppm_raw : pandas.DataFrame, optional
        DataFrame bruto da PPM. Se None, carrega do disco.
    df_localidades : pandas.DataFrame, optional
        DataFrame de localidades. Se None, carrega do disco.
    quantis : tuple of two floats
        Passado para :func:`derive_perfis_agropecuarios`.

    Returns
    -------
    pandas.DataFrame
        Dataset final com localidades + atividades + perfis + especialização
        + diversidade + tags.
    """
    if df_ppm_raw is None:
        ano_efetivo = ano if ano is not None else dataset.PPM_ULTIMO_DISPONIVEL
        df_ppm_raw = dataset.load_ppm_efetivo_rebanhos(ano=ano_efetivo)
    if df_localidades is None:
        df_localidades = dataset.load_localidades()

    logger.info("[STAGE 1/5] Limpando PPM: shape inicial=%s", df_ppm_raw.shape)
    df_clean = clean_ppm(df_ppm_raw)
    logger.info("[STAGE 1/5] PPM limpa: shape=%s, atividades=%s",
                df_clean.shape, sorted(df_clean["atividade"].unique()))

    logger.info("[STAGE 2/5] Pivot long → wide")
    df_wide = pivot_ppm_wide(df_clean)
    logger.info("[STAGE 2/5] Wide shape=%s, colunas=%s",
                df_wide.shape, list(df_wide.columns))

    logger.info("[STAGE 3/5] Merge com localidades")
    df_merged = merge_com_localidades(df_wide, df_localidades)
    logger.info("[STAGE 3/5] Merged shape=%s", df_merged.shape)

    logger.info("[STAGE 4/5] Derivando perfis, especialização e diversidade")
    df_perfis = derive_perfis_agropecuarios(df_merged, quantis=quantis)
    df_esp = derive_especializacao(df_perfis)
    df_div = derive_diversidade(df_esp)

    logger.info("[STAGE 5/5] Construindo tags")
    df_final = build_tags(df_div)
    logger.info(
        "[OK] Dataset final: shape=%s, colunas=%d",
        df_final.shape,
        len(df_final.columns),
    )
    return df_final


def save_features_dataset(df: pd.DataFrame) -> Path:
    """Persiste o DataFrame processado em ``data/processed/``."""
    config.ensure_directories()
    path = get_features_processed_path()
    df.to_parquet(path, index=False)
    logger.info("[IO] Dataset processado salvo em %s", path)
    return path


def load_features_dataset() -> pd.DataFrame:
    """Carrega o DataFrame processado do disco.

    Raises
    ------
    FileNotFoundError
        Se o Parquet ainda não foi gerado (rodar :func:`build_features_dataset`
        e :func:`save_features_dataset` antes, ou o CLI
        ``python -m rec_agro_br.features``).
    """
    path = get_features_processed_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado em {path}. "
            "Rode primeiro: python -m rec_agro_br.features"
        )
    return pd.read_parquet(path)


# =============================================================================
# CLI
# =============================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_dataset_summary(df: pd.DataFrame) -> None:
    """Imprime um resumo compacto do dataset processado."""
    print(f"\n[OK] Dataset processado: {df.shape[0]} municípios × {df.shape[1]} colunas")
    print(f"     Salvo em: {get_features_processed_path()}\n")

    print("=" * 78)
    print("Colunas de contexto territorial:")
    for col in COLUNAS_CONTEXTO_TERRITORIAL:
        if col in df.columns:
            print(f"  - {col}: {df[col].nunique()} valores únicos")

    print("\nColunas de atividades (numéricas, efetivo em cabeças):")
    atividades = [c for c in MAPA_TIPOS_REBANHO.values() if c in df.columns]
    for atv in atividades:
        n_com = (df[atv] > 0).sum()
        total = int(df[atv].sum())
        print(f"  - {atv}: {n_com} municípios, total {total:,} cabeças")

    print("\nDistribuição de especialização:")
    if "especializacao" in df.columns:
        print(df["especializacao"].value_counts().to_string())

    print("\nDistribuição de número de atividades por município:")
    if "n_atividades" in df.columns:
        print(df["n_atividades"].value_counts().sort_index().to_string())

    print("\n" + "=" * 78)
    print("Amostra do campo 'tags' (primeiras 5 linhas):\n")
    for _, row in df.head(5).iterrows():
        print(f"  [{row['sigla_uf']}] {row['nome_municipio']}:")
        print(f"    {row['tags']}\n")


def _cmd_features(args: argparse.Namespace) -> int:
    df = build_features_dataset(ano=args.ano)
    save_features_dataset(df)
    _print_dataset_summary(df)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rec_agro_br.features",
        description=(
            "Executa o pipeline de feature engineering do projeto rec-agro-br: "
            "limpa a PPM, pivota para wide, faz merge com localidades, deriva "
            "perfis quantitativos e categóricos, e monta o campo 'tags' que "
            "será vetorizado pelo CountVectorizer na Fase 1.D."
        ),
    )
    parser.add_argument(
        "--ano",
        type=int,
        default=None,
        help=(
            "Ano de referência da PPM. Se omitido, usa o último disponível "
            "no cache (arquivo 'ppm_3939_efetivo_rebanhos_last_1.parquet')."
        ),
    )
    parser.set_defaults(func=_cmd_features)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        logger.error(
            "[ERRO] Pré-requisito ausente: %s. "
            "Rode primeiro 'python -m rec_agro_br.dataset all'.",
            e,
        )
        return 3
    except Exception as e:
        logger.exception("[ERRO] Falha inesperada: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
