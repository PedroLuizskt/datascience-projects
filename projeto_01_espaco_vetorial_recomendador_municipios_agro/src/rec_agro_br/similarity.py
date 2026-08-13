"""Métricas de similaridade e distância entre vetores.

Este módulo implementa as métricas de comparação vetorial que fundamentam
o sistema de recomendação content-based. Oferece três abordagens ao
problema de medir "quão parecidos" são dois municípios representados
como vetores em :math:`\\mathbb{R}^v`:

- **Similaridade cosseno**: mede o ângulo entre vetores; robusta a
  diferenças de magnitude, ideal para bag-of-words.
- **Distância euclidiana**: mede o comprimento do segmento entre pontos;
  sensível a diferenças de magnitude, útil quando escala importa.
- **Distância Manhattan** (L1): soma das diferenças absolutas por
  dimensão; útil para features esparsas onde queremos "contar" divergências.

Todas as três são exercitadas no módulo Cap08 da pós-graduação em Ciência
de Dados da DSA. O projeto original enfatiza a cosseno para o recomendador
e a euclidiana no exercício complementar; aqui replicamos ambas e ainda
adicionamos Manhattan por completude conceitual, permitindo que o notebook
da Fase 1.E possa comparar as três empiricamente.

Formalismos matemáticos
-----------------------
Para dois vetores :math:`\\mathbf{u}, \\mathbf{v} \\in \\mathbb{R}^v`:

- Cosseno: :math:`\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}
  {\\lVert \\mathbf{u} \\rVert_2 \\cdot \\lVert \\mathbf{v} \\rVert_2}`.
  Domínio: :math:`[-1, 1]` (em bag-of-words não-negativa, :math:`[0, 1]`).

- Euclidiana: :math:`d_E(\\mathbf{u}, \\mathbf{v}) =
  \\sqrt{\\sum_{i=1}^{v} (u_i - v_i)^2}`. Domínio: :math:`[0, \\infty)`.

- Manhattan: :math:`d_M(\\mathbf{u}, \\mathbf{v}) =
  \\sum_{i=1}^{v} |u_i - v_i|`. Domínio: :math:`[0, \\infty)`.

Implementação
-------------
Para operações em lote (matriz × matriz), delegamos ao scikit-learn, que
tem implementações vetorizadas eficientes em Cython. Para casos
individuais (par único de vetores), fornecemos implementações manuais
didáticas usando apenas NumPy, com o propósito explícito de exercitar
os conceitos matemáticos que o módulo da pós-graduação ensina.

Exemplos
--------
Uso programático::

    from rec_agro_br import similarity, vectorize
    vec, X = vectorize.load_vectorizer(), vectorize.load_matrix()
    # Todos-contra-todos: matriz 5571 x 5571
    S = similarity.cosine_similarity_matrix(X)
    # Consulta específica: top-5 mais similares ao município no índice 42
    top = similarity.top_k_similares(S[42], k=5, excluir_indice=42)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import (
    cosine_similarity as _sk_cosine,
    euclidean_distances as _sk_euclidean,
    manhattan_distances as _sk_manhattan,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Métricas em lote (matriz × matriz) — implementações scikit-learn
# =============================================================================
def cosine_similarity_matrix(
    X: sparse.csr_matrix | np.ndarray,
    Y: sparse.csr_matrix | np.ndarray | None = None,
) -> np.ndarray:
    """Calcula a matriz de similaridade cosseno entre linhas.

    Se ``Y`` for ``None``, calcula ``X × X`` (todos-contra-todos). Caso
    contrário, calcula ``X × Y`` (útil para consultas: X é a query, Y é
    o corpus completo).

    Delegado a :func:`sklearn.metrics.pairwise.cosine_similarity`, que
    normaliza os vetores internamente e usa BLAS por baixo. Para uma
    matriz esparsa 5571 × 215, roda em milissegundos.

    Parameters
    ----------
    X : sparse or ndarray, shape (n, v)
    Y : sparse or ndarray, shape (m, v), optional

    Returns
    -------
    ndarray, shape (n, m) or (n, n)
        Cada célula :math:`S_{i,j}` contém a similaridade cosseno entre
        a linha :math:`i` de ``X`` e a linha :math:`j` de ``Y`` (ou ``X``).
    """
    return _sk_cosine(X, Y)


def euclidean_distance_matrix(
    X: sparse.csr_matrix | np.ndarray,
    Y: sparse.csr_matrix | np.ndarray | None = None,
) -> np.ndarray:
    """Matriz de distâncias euclidianas entre linhas.

    Delegado a :func:`sklearn.metrics.pairwise.euclidean_distances`.
    """
    return _sk_euclidean(X, Y)


def manhattan_distance_matrix(
    X: sparse.csr_matrix | np.ndarray,
    Y: sparse.csr_matrix | np.ndarray | None = None,
) -> np.ndarray:
    """Matriz de distâncias Manhattan (L1) entre linhas.

    Delegado a :func:`sklearn.metrics.pairwise.manhattan_distances`.
    """
    return _sk_manhattan(X, Y)


# =============================================================================
# Métricas manuais entre dois vetores — implementações didáticas
# =============================================================================
def _to_dense_1d(v: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    """Converte vetor esparso ou 2D em array 1D denso."""
    if sparse.issparse(v):
        v = v.toarray()
    v = np.asarray(v, dtype=np.float64).ravel()
    return v


def cosine_similarity_pair(
    u: np.ndarray | sparse.csr_matrix,
    v: np.ndarray | sparse.csr_matrix,
) -> float:
    """Similaridade cosseno entre dois vetores (implementação didática).

    Implementa a fórmula clássica passo a passo, para exercitar o
    conceito ensinado no módulo Cap08:

    .. math::
        \\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}
        {\\lVert \\mathbf{u} \\rVert_2 \\cdot \\lVert \\mathbf{v} \\rVert_2}

    Para uso em produção sobre muitas comparações, prefira
    :func:`cosine_similarity_matrix`, que é vetorizada.

    Parameters
    ----------
    u, v : array-like, shape (n,) ou (1, n)
        Vetores a comparar. Aceita array 1D, array 2D com uma linha,
        ou matriz esparsa com uma linha.

    Returns
    -------
    float
        Similaridade em :math:`[-1, 1]`. Para vetores não-negativos
        (nosso caso, bag-of-words), :math:`[0, 1]`.

    Raises
    ------
    ValueError
        Se algum dos vetores tiver norma zero (não há direção definida).
    """
    u_arr = _to_dense_1d(u)
    v_arr = _to_dense_1d(v)
    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Shapes incompatíveis: {u_arr.shape} vs {v_arr.shape}"
        )

    produto_interno = float(np.dot(u_arr, v_arr))
    norma_u = float(np.linalg.norm(u_arr))
    norma_v = float(np.linalg.norm(v_arr))

    if norma_u == 0.0 or norma_v == 0.0:
        raise ValueError(
            "Similaridade cosseno indefinida para vetores nulos "
            "(norma zero). Verifique se o município tem alguma tag."
        )

    return produto_interno / (norma_u * norma_v)


def euclidean_distance_pair(
    u: np.ndarray | sparse.csr_matrix,
    v: np.ndarray | sparse.csr_matrix,
) -> float:
    """Distância euclidiana entre dois vetores (implementação didática).

    Implementa a fórmula:

    .. math::
        d_E(\\mathbf{u}, \\mathbf{v}) = \\sqrt{\\sum_{i=1}^{n} (u_i - v_i)^2}

    Este é exatamente o cálculo exercitado no `Exercicio5-Solucao.ipynb`
    do projeto DSA original (embora aquele exercício não faça parte da
    entrega deste adaptador, o conceito é preservado aqui para completude).
    """
    u_arr = _to_dense_1d(u)
    v_arr = _to_dense_1d(v)
    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Shapes incompatíveis: {u_arr.shape} vs {v_arr.shape}"
        )
    diff = u_arr - v_arr
    return float(np.sqrt(np.sum(diff * diff)))


def manhattan_distance_pair(
    u: np.ndarray | sparse.csr_matrix,
    v: np.ndarray | sparse.csr_matrix,
) -> float:
    """Distância Manhattan (L1) entre dois vetores.

    Implementa:

    .. math::
        d_M(\\mathbf{u}, \\mathbf{v}) = \\sum_{i=1}^{n} |u_i - v_i|
    """
    u_arr = _to_dense_1d(u)
    v_arr = _to_dense_1d(v)
    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Shapes incompatíveis: {u_arr.shape} vs {v_arr.shape}"
        )
    return float(np.sum(np.abs(u_arr - v_arr)))


# =============================================================================
# Recuperação de top-k (usado pelo recomendador)
# =============================================================================
def top_k_similares(
    scores: np.ndarray,
    k: int = 5,
    excluir_indice: int | None = None,
    excluir_indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Retorna os índices e scores dos ``k`` maiores valores em ``scores``.

    Usado para similaridade (maior é melhor). Para distâncias (menor é
    melhor), use :func:`top_k_mais_proximos`.

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Vetor 1D de scores de similaridade.
    k : int
        Quantos itens retornar.
    excluir_indice : int, optional
        Um índice a excluir (tipicamente o próprio município consultado,
        que teria score 1.0 e apareceria em primeiro).
    excluir_indices : list of int, optional
        Múltiplos índices a excluir. Complementa ``excluir_indice``.

    Returns
    -------
    list of tuples (indice, score)
        Ordenados por score decrescente. Comprimento ``k``.
    """
    return _top_k_ordenado(
        scores=scores,
        k=k,
        descending=True,
        excluir_indice=excluir_indice,
        excluir_indices=excluir_indices,
    )


def top_k_mais_proximos(
    distances: np.ndarray,
    k: int = 5,
    excluir_indice: int | None = None,
    excluir_indices: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Retorna os índices e distâncias dos ``k`` menores valores.

    Usado para distâncias (euclidiana, manhattan) onde menor é melhor.
    """
    return _top_k_ordenado(
        distances,
        k=k,
        descending=False,
        excluir_indice=excluir_indice,
        excluir_indices=excluir_indices,
    )


def _top_k_ordenado(
    scores: np.ndarray,
    k: int,
    descending: bool,
    excluir_indice: int | None,
    excluir_indices: list[int] | None,
) -> list[tuple[int, float]]:
    """Implementação compartilhada de top-k, parametrizada pela direção."""
    scores = np.asarray(scores)
    if k <= 0:
        raise ValueError(f"k deve ser positivo, recebido {k}")
    if scores.ndim != 1:
        raise ValueError(
            f"scores deve ser 1D, recebido shape {scores.shape}. "
            "Para consultas com múltiplas queries, itere linha a linha."
        )

    a_excluir: set[int] = set()
    if excluir_indice is not None:
        a_excluir.add(int(excluir_indice))
    if excluir_indices:
        a_excluir.update(int(i) for i in excluir_indices)

    if a_excluir:
        # Substitui os excluídos por sentinela apropriada
        sentinela = -np.inf if descending else np.inf
        scores = scores.copy()
        for i in a_excluir:
            if 0 <= i < len(scores):
                scores[i] = sentinela

    k_efetivo = min(k, len(scores) - len(a_excluir))
    if k_efetivo <= 0:
        return []

    # argpartition: O(n) para pegar os k maiores/menores (não ordenados)
    if descending:
        idx_top = np.argpartition(scores, -k_efetivo)[-k_efetivo:]
        idx_top = idx_top[np.argsort(scores[idx_top])[::-1]]
    else:
        idx_top = np.argpartition(scores, k_efetivo - 1)[:k_efetivo]
        idx_top = idx_top[np.argsort(scores[idx_top])]

    return [(int(i), float(scores[i])) for i in idx_top]
