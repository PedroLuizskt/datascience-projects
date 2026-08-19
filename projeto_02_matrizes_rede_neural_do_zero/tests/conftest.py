"""Fixtures compartilhadas para a suite pytest do projeto rna_matrizes."""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# Datasets sintéticos
# =============================================================================
@pytest.fixture
def dataset_dsa_original() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dataset toy 100% idêntico ao usado no projeto DSA original (8 amostras).

    Útil para testes de fidelidade: garantir que nossa implementação, dados
    os mesmos hiperparâmetros e o mesmo dataset, chega em resultados
    matematicamente compatíveis com os que aparecem no notebook DSA.

    Cenário narrativo original: features [valor_transacao, hora_dia], rótulo
    binário 0=legítima / 1=fraude.

    Returns
    -------
    X_treino, y_treino, X_teste, y_teste : ndarray
        Arrays com dtype float64 para X e int64 para y.
    """
    X_treino = np.array(
        [[1, 2.5], [2, 3], [3, 5], [1, 4], [5, 6], [6, 7]],
        dtype=np.float64,
    )
    y_treino = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
    X_teste = np.array([[1.5, 2], [4, 5.5]], dtype=np.float64)
    y_teste = np.array([0, 1], dtype=np.int64)
    return X_treino, y_treino, X_teste, y_teste


@pytest.fixture
def dataset_linearmente_separavel() -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintético 2D perfeitamente separável por uma linha reta.

    Duas nuvens gaussianas com médias bem afastadas — problema trivial que
    o modelo tem que resolver com acurácia próxima de 100% dado tempo
    suficiente de treino.
    """
    rng = np.random.default_rng(seed=42)
    n_por_classe = 50

    classe_0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n_por_classe, 2))
    classe_1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n_por_classe, 2))

    X = np.vstack([classe_0, classe_1]).astype(np.float64)
    y = np.concatenate([
        np.zeros(n_por_classe, dtype=np.int64),
        np.ones(n_por_classe, dtype=np.int64),
    ])
    return X, y


@pytest.fixture
def dataset_nao_separavel() -> tuple[np.ndarray, np.ndarray]:
    """Dataset com sobreposição significativa das classes.

    Nuvens gaussianas próximas — o modelo nunca vai chegar a 100% de
    acurácia. Útil para testar comportamento sob dificuldade real.
    """
    rng = np.random.default_rng(seed=123)
    n_por_classe = 100

    classe_0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_por_classe, 2))
    classe_1 = rng.normal(loc=[1.0, 1.0], scale=1.0, size=(n_por_classe, 2))

    X = np.vstack([classe_0, classe_1]).astype(np.float64)
    y = np.concatenate([
        np.zeros(n_por_classe, dtype=np.int64),
        np.ones(n_por_classe, dtype=np.int64),
    ])
    return X, y
