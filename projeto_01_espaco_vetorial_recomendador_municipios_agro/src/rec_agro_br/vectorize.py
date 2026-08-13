"""Vetorização de texto: transformação das tags em espaço vetorial.

Este módulo implementa a Etapa de *representação vetorial* do pipeline do
projeto rec-agro-br. Consome o dataset processado gerado por :mod:`features`
e produz duas coisas: (i) um vetorizador (``CountVectorizer``) ajustado ao
vocabulário do corpus de tags agropecuárias, e (ii) a matriz esparsa
:math:`X \\in \\mathbb{R}^{n \\times v}` onde :math:`n` é o número de
municípios brasileiros (5571) e :math:`v` o tamanho do vocabulário
efetivamente construído (~215 tokens após pré-processamento).

Cada linha de :math:`X` é a representação vetorial de um município no
espaço de tags agropecuárias. É esta matriz que o módulo :mod:`similarity`
consome para calcular distâncias e o :mod:`recommender` para produzir
recomendações content-based.

Núcleo pedagógico
-----------------
Este módulo é a implementação direta do conceito ensinado no projeto Cap08
da pós-graduação em Ciência de Dados da DSA: transformar itens de qualquer
natureza em vetores em um espaço :math:`\\mathbb{R}^v` onde :math:`v` é o
tamanho do vocabulário de features textuais. No projeto original, os itens
eram filmes; aqui são municípios. O ferramental do scikit-learn é o mesmo
(``CountVectorizer``), assim como o esquema de bag-of-words: cada célula
:math:`X_{i,j}` conta quantas vezes o token :math:`j` aparece no documento
(tags) do município :math:`i`.

Adaptação linguística: stemming português
------------------------------------------
O projeto DSA original usa o :class:`PorterStemmer` do NLTK, projetado
para inglês. Como nosso corpus é em português, adotamos o
:class:`RSLPStemmer` (Removedor de Sufixos da Língua Portuguesa), também
do NLTK. O RSLP é um dos poucos algoritmos de stemming projetados
especificamente para português e cobre corretamente os padrões
morfológicos da língua.

Uma sutileza importante: nossos tokens misturam palavras simples
(``bovinocultura``, ``nordeste``) com tokens compostos por underscore
(``sul_sudoeste_de_minas``, ``especializado_em_bovinocultura``). O tokenizer
customizado aqui implementado stemmiza apenas os tokens simples e preserva
intactos os compostos, evitando que o stemmer os fragmente ou introduza
ruído em identificadores geográficos que são semanticamente unitários.

Exemplos
--------
Uso programático::

    from rec_agro_br import vectorize, features
    df = features.load_features_dataset()
    vec, X = vectorize.fit_and_transform(df["tags"], use_stemming=True)
    print(f"Vocabulário: {len(vec.vocabulary_)}, Matriz: {X.shape}")

Uso como linha de comando::

    python -m rec_agro_br.vectorize
    python -m rec_agro_br.vectorize --sem-stemming
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.stem import RSLPStemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from rec_agro_br import config, features

logger = logging.getLogger(__name__)


# =============================================================================
# Nomes canônicos de arquivos persistidos
# =============================================================================
VECTORIZER_JOBLIB: str = "count_vectorizer.joblib"
MATRIX_NPZ: str = "tags_matrix.npz"

# Path canônico deste módulo, usado para resolver referências de função via
# importlib. Ver notas em `build_vectorizer` e no bloco de correção ao final
# das definições dos tokenizers para o mecanismo completo.
_CANONICAL_MODULE_PATH: str = "rec_agro_br.vectorize"


def get_vectorizer_path() -> Path:
    """Caminho canônico do vectorizer serializado."""
    return config.PROCESSED_DATA_DIR / VECTORIZER_JOBLIB


def get_matrix_path() -> Path:
    """Caminho canônico da matriz esparsa serializada."""
    return config.PROCESSED_DATA_DIR / MATRIX_NPZ


# =============================================================================
# Stemmer português (RSLP) — download idempotente e cache singleton
# =============================================================================
_stemmer_singleton: RSLPStemmer | None = None


def _ensure_rslp_downloaded() -> None:
    """Garante que os dados do RSLP estejam disponíveis no NLTK.

    O RSLPStemmer requer arquivos de regras que o NLTK baixa sob demanda.
    Esta função é idempotente: se já baixado, não faz nada; se não, baixa
    silenciosamente. Chamada uma única vez na primeira instanciação do
    stemmer.
    """
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        logger.info("[NLTK] Baixando dados do RSLPStemmer (uma vez só)...")
        nltk.download("rslp", quiet=True)


def _get_stemmer() -> RSLPStemmer:
    """Retorna a instância singleton do RSLPStemmer.

    O stemmer é caro para instanciar (carrega arquivos de regras), então
    reutilizamos uma única instância ao longo do processo.
    """
    global _stemmer_singleton
    if _stemmer_singleton is None:
        _ensure_rslp_downloaded()
        _stemmer_singleton = RSLPStemmer()
    return _stemmer_singleton


# =============================================================================
# Tokenizers
# =============================================================================
def tokenize_simples(texto: str) -> list[str]:
    """Tokenizador sem stemming: apenas lowercase + split por espaço.

    Útil quando queremos preservar o vocabulário original sem redução
    morfológica, ou para debug do que está entrando na vetorização.
    """
    if texto is None or not isinstance(texto, str):
        return []
    return texto.lower().split()


def tokenize_com_stemming(texto: str) -> list[str]:
    """Tokenizador com stemming português seletivo (RSLP).

    Regra: stemmiza apenas tokens sem underscore. Tokens compostos como
    ``sul_sudoeste_de_minas`` ou ``especializado_em_bovinocultura`` são
    preservados intactos, pois representam entidades semanticamente
    unitárias que não devem ser fragmentadas pelo stemmer.

    Exemplos:
        ``"bovinocultura"`` → ``"bovinocultur"`` (RSLP remove sufixo)
        ``"sul_sudoeste_de_minas"`` → ``"sul_sudoeste_de_minas"`` (intacto)
        ``"especializado_em_bovinocultura"`` → intacto
        ``"avicultura"`` → ``"avicultur"``
    """
    if texto is None or not isinstance(texto, str):
        return []
    stemmer = _get_stemmer()
    tokens = texto.lower().split()
    resultado: list[str] = []
    for tok in tokens:
        if "_" in tok:
            resultado.append(tok)
        else:
            try:
                resultado.append(stemmer.stem(tok))
            except (IndexError, ValueError):
                # RSLP pode falhar em tokens muito curtos ou anômalos
                resultado.append(tok)
    return resultado


# =============================================================================
# Robustez de pickle: forçar `__module__` canônico das funções tokenizer
# =============================================================================
# Complementa a resolução via ``importlib.import_module`` feita em
# :func:`build_vectorizer`. As duas correções trabalham em conjunto:
#
# 1. Este bloco força ``tokenize_*.__module__ = 'rec_agro_br.vectorize'``,
#    o que garante que quando as funções forem serializadas pelo pickle,
#    a referência simbólica gravada seja o path canônico do pacote
#    (não ``__main__``).
#
# 2. :func:`build_vectorizer` importa via ``importlib.import_module`` para
#    garantir que a instância da função passada ao CountVectorizer seja
#    exatamente a mesma que está em ``sys.modules['rec_agro_br.vectorize']``.
#    Sem isso, ao rodar como ``python -m rec_agro_br.vectorize``, o pickle
#    detectaria que a função de ``__main__`` é objeto Python distinto da
#    versão canônica e falharia com ``PicklingError: it's not the same object``.
tokenize_simples.__module__ = _CANONICAL_MODULE_PATH
tokenize_com_stemming.__module__ = _CANONICAL_MODULE_PATH


# =============================================================================
# Construção do vectorizer
# =============================================================================
def build_vectorizer(
    use_stemming: bool = True,
    max_features: int | None = None,
    min_df: int | float = 1,
    max_df: int | float = 1.0,
) -> CountVectorizer:
    """Constrói um ``CountVectorizer`` do scikit-learn parametrizado.

    Parameters
    ----------
    use_stemming : bool
        Se ``True`` (default), aplica RSLPStemmer aos tokens simples.
        Se ``False``, usa apenas lowercase + split.
    max_features : int, optional
        Limite superior de features. Se ``None``, mantém todo o vocabulário.
        Nosso corpus tem ~215 tokens únicos, então ``None`` é seguro e
        preserva toda a informação.
    min_df : int or float
        Frequência mínima de documento (número absoluto ou proporção).
        Default 1 = mantém tokens que aparecem em pelo menos 1 município.
    max_df : int or float
        Frequência máxima de documento. Default 1.0 = sem limite superior.

    Returns
    -------
    CountVectorizer
        Vectorizer não ajustado, pronto para ``fit_transform``.

    Notas de implementação (robustez de pickle)
    -------------------------------------------
    As funções tokenizer são resolvidas via ``importlib.import_module`` do
    caminho canônico ``rec_agro_br.vectorize``, ao invés de usar as
    referências locais deste módulo. Quando este arquivo é executado como
    script (``python -m rec_agro_br.vectorize``), o Python cria duas
    instâncias distintas do módulo em ``sys.modules``: uma em ``__main__``
    e outra em ``rec_agro_br.vectorize``, cada uma com suas próprias
    funções tokenizer (mesmo código, objetos Python distintos). Sem o
    ``importlib.import_module`` explícito, o CountVectorizer receberia a
    função da instância ``__main__``, e o pickle subsequente falharia com
    ``PicklingError: it's not the same object`` ao tentar validar contra
    a versão em ``rec_agro_br.vectorize``.
    """
    import importlib

    canonical_module = importlib.import_module(_CANONICAL_MODULE_PATH)
    tokenizer = (
        canonical_module.tokenize_com_stemming
        if use_stemming
        else canonical_module.tokenize_simples
    )
    return CountVectorizer(
        tokenizer=tokenizer,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,  # o tokenizer já faz lower
        token_pattern=None,  # desabilitado quando tokenizer é custom
    )


def fit_and_transform(
    tags: pd.Series,
    use_stemming: bool = True,
    max_features: int | None = None,
    min_df: int | float = 1,
    max_df: int | float = 1.0,
) -> tuple[CountVectorizer, sparse.csr_matrix]:
    """Constrói e ajusta o vectorizer sobre o corpus de tags.

    Parameters
    ----------
    tags : pandas.Series
        Coluna ``tags`` do dataset processado (uma tag string por município).
    use_stemming, max_features, min_df, max_df
        Passados diretamente para :func:`build_vectorizer`.

    Returns
    -------
    tuple of (CountVectorizer, scipy.sparse.csr_matrix)
        Vectorizer ajustado e matriz de features com shape
        ``(n_municipios, tamanho_vocabulario)``.
    """
    corpus = tags.fillna("").astype(str).tolist()
    if not corpus:
        raise ValueError("Série de tags está vazia.")

    vec = build_vectorizer(
        use_stemming=use_stemming,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    logger.info(
        "[FIT] Ajustando CountVectorizer sobre %d documentos (stemming=%s)...",
        len(corpus),
        use_stemming,
    )
    matrix = vec.fit_transform(corpus)
    logger.info(
        "[OK] Matriz gerada: shape=%s, vocabulário=%d, densidade=%.4f%%",
        matrix.shape,
        len(vec.vocabulary_),
        100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )
    return vec, matrix


def transform(
    vectorizer: CountVectorizer,
    tags: pd.Series,
) -> sparse.csr_matrix:
    """Aplica um vectorizer já ajustado a um novo conjunto de tags.

    Útil quando queremos vetorizar um único município novo sem refit do
    vocabulário — por exemplo, para uma consulta ao recomendador.
    """
    corpus = tags.fillna("").astype(str).tolist()
    return vectorizer.transform(corpus)


# =============================================================================
# Persistência (joblib para vectorizer, npz para matriz esparsa)
# =============================================================================
def save_vectorizer(vec: CountVectorizer, path: Path | None = None) -> Path:
    """Persiste o vectorizer com ``joblib``."""
    config.ensure_directories()
    target = path or get_vectorizer_path()
    joblib.dump(vec, target)
    logger.info("[IO] Vectorizer salvo em %s", target)
    return target


def load_vectorizer(path: Path | None = None) -> CountVectorizer:
    """Carrega vectorizer previamente salvo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo ainda não foi gerado.
    """
    target = path or get_vectorizer_path()
    if not target.exists():
        raise FileNotFoundError(
            f"Vectorizer não encontrado em {target}. "
            "Rode primeiro: python -m rec_agro_br.vectorize"
        )
    return joblib.load(target)


def save_matrix(matrix: sparse.csr_matrix, path: Path | None = None) -> Path:
    """Persiste a matriz esparsa em formato ``.npz`` do scipy."""
    config.ensure_directories()
    target = path or get_matrix_path()
    sparse.save_npz(target, matrix)
    logger.info("[IO] Matriz esparsa salva em %s", target)
    return target


def load_matrix(path: Path | None = None) -> sparse.csr_matrix:
    """Carrega a matriz esparsa previamente salva.

    Raises
    ------
    FileNotFoundError
        Se o arquivo ainda não foi gerado.
    """
    target = path or get_matrix_path()
    if not target.exists():
        raise FileNotFoundError(
            f"Matriz não encontrado em {target}. "
            "Rode primeiro: python -m rec_agro_br.vectorize"
        )
    return sparse.load_npz(target)


# =============================================================================
# Pipeline de alto nível
# =============================================================================
def build_and_persist(
    use_stemming: bool = True,
    df_features: pd.DataFrame | None = None,
) -> tuple[CountVectorizer, sparse.csr_matrix]:
    """Executa o pipeline completo: carrega, vetoriza e persiste.

    Parameters
    ----------
    use_stemming : bool
        Se True, usa RSLPStemmer nos tokens simples.
    df_features : pandas.DataFrame, optional
        Se passado, usa diretamente. Se None, carrega do disco.

    Returns
    -------
    tuple of (CountVectorizer, scipy.sparse.csr_matrix)
        Objetos ajustados e persistidos em ``data/processed/``.
    """
    if df_features is None:
        df_features = features.load_features_dataset()

    if "tags" not in df_features.columns:
        raise ValueError(
            "Coluna 'tags' ausente no DataFrame. Rode primeiro o pipeline "
            "de features: python -m rec_agro_br.features"
        )

    vec, matrix = fit_and_transform(df_features["tags"], use_stemming=use_stemming)
    save_vectorizer(vec)
    save_matrix(matrix)
    return vec, matrix


# =============================================================================
# CLI
# =============================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_vocabulary_summary(vec: CountVectorizer, matrix: sparse.csr_matrix) -> None:
    """Imprime resumo do vocabulário construído."""
    print(f"\n[OK] CountVectorizer ajustado")
    print(f"     Documentos: {matrix.shape[0]} municípios")
    print(f"     Vocabulário: {len(vec.vocabulary_)} tokens únicos")
    print(f"     Densidade da matriz: "
          f"{100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2f}%")

    print(f"\n     Salvos em:")
    print(f"       - {get_vectorizer_path()}")
    print(f"       - {get_matrix_path()}")

    # Top 20 tokens mais frequentes
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    idx_top = np.argsort(freq)[::-1][:20]
    inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
    print(f"\n     Top 20 tokens mais frequentes:")
    for i, idx in enumerate(idx_top, 1):
        print(f"       {i:>2}. {inv_vocab[idx]:<40s} freq={int(freq[idx])}")


def _cmd_vectorize(args: argparse.Namespace) -> int:
    vec, matrix = build_and_persist(use_stemming=not args.sem_stemming)
    _print_vocabulary_summary(vec, matrix)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rec_agro_br.vectorize",
        description=(
            "Vetoriza o corpus de tags agropecuárias usando CountVectorizer "
            "do scikit-learn com stemming português (RSLP). Persiste o "
            "vectorizer e a matriz esparsa em data/processed/."
        ),
    )
    parser.add_argument(
        "--sem-stemming",
        action="store_true",
        help=(
            "Desabilita o stemming RSLP. Útil para comparação didática "
            "entre corpus stemmizado e não-stemmizado."
        ),
    )
    parser.set_defaults(func=_cmd_vectorize)
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
            "Rode primeiro 'python -m rec_agro_br.features'.",
            e,
        )
        return 3
    except Exception as e:
        logger.exception("[ERRO] Falha inesperada: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
