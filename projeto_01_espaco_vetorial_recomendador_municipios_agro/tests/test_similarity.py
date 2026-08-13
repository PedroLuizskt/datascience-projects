"""Testes do módulo `rec_agro_br.similarity`.

Cobre as três métricas (cosseno, euclidiana, manhattan) tanto em suas
implementações vetorizadas (scikit-learn) quanto nas implementações
didáticas manuais. Também testa a recuperação de top-k com e sem
exclusão de índices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from rec_agro_br import similarity


# =============================================================================
# Cosseno em par (manual)
# =============================================================================
class TestCosineSimilarityPair:
    def test_vetores_identicos_similaridade_1(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        assert similarity.cosine_similarity_pair(u, u) == pytest.approx(1.0)

    def test_vetores_ortogonais_similaridade_0(self) -> None:
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert similarity.cosine_similarity_pair(u, v) == pytest.approx(0.0)

    def test_vetores_opostos_similaridade_menos_1(self) -> None:
        u = np.array([1.0, 1.0])
        v = np.array([-1.0, -1.0])
        assert similarity.cosine_similarity_pair(u, v) == pytest.approx(-1.0)

    def test_valor_esperado_conhecido(self) -> None:
        """Cosseno entre [1,2,3] e [4,5,6] = 32/(√14 * √77)."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        esperado = 32 / (math.sqrt(14) * math.sqrt(77))
        assert similarity.cosine_similarity_pair(u, v) == pytest.approx(esperado)

    def test_norma_zero_levanta_erro(self) -> None:
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nulos"):
            similarity.cosine_similarity_pair(u, v)

    def test_shapes_incompativeis(self) -> None:
        u = np.array([1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="incompat"):
            similarity.cosine_similarity_pair(u, v)

    def test_aceita_sparse(self) -> None:
        u_dense = np.array([1.0, 2.0, 3.0])
        u_sparse = sparse.csr_matrix(u_dense)
        v_dense = np.array([4.0, 5.0, 6.0])
        s_dense = similarity.cosine_similarity_pair(u_dense, v_dense)
        s_sparse = similarity.cosine_similarity_pair(u_sparse, v_dense)
        assert s_dense == pytest.approx(s_sparse)


# =============================================================================
# Euclidiana em par (manual)
# =============================================================================
class TestEuclideanDistancePair:
    def test_vetores_identicos_distancia_0(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        assert similarity.euclidean_distance_pair(u, u) == pytest.approx(0.0)

    def test_valor_esperado_pitagoras(self) -> None:
        """Distância entre [0,0] e [3,4] = 5 (triângulo 3-4-5)."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])
        assert similarity.euclidean_distance_pair(u, v) == pytest.approx(5.0)

    def test_simetrica(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        d1 = similarity.euclidean_distance_pair(u, v)
        d2 = similarity.euclidean_distance_pair(v, u)
        assert d1 == pytest.approx(d2)


# =============================================================================
# Manhattan em par (manual)
# =============================================================================
class TestManhattanDistancePair:
    def test_vetores_identicos_distancia_0(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        assert similarity.manhattan_distance_pair(u, u) == pytest.approx(0.0)

    def test_valor_esperado(self) -> None:
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        assert similarity.manhattan_distance_pair(u, v) == pytest.approx(6.0)

    def test_diferente_de_euclidiana_em_geral(self) -> None:
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])
        assert similarity.manhattan_distance_pair(u, v) == pytest.approx(7.0)
        assert similarity.euclidean_distance_pair(u, v) == pytest.approx(5.0)


# =============================================================================
# Matrizes de similaridade/distância (scikit-learn)
# =============================================================================
class TestMatrizes:
    @pytest.fixture
    def X_pequeno(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

    def test_cosine_matriz_quadrada_diagonal_1(
        self, X_pequeno: np.ndarray
    ) -> None:
        S = similarity.cosine_similarity_matrix(X_pequeno)
        assert S.shape == (4, 4)
        for i in range(4):
            assert S[i, i] == pytest.approx(1.0)

    def test_cosine_simetrica(self, X_pequeno: np.ndarray) -> None:
        S = similarity.cosine_similarity_matrix(X_pequeno)
        np.testing.assert_allclose(S, S.T)

    def test_cosine_X_versus_Y(self, X_pequeno: np.ndarray) -> None:
        Y = X_pequeno[:2]
        S = similarity.cosine_similarity_matrix(X_pequeno, Y)
        assert S.shape == (4, 2)

    def test_euclidean_matriz(self, X_pequeno: np.ndarray) -> None:
        D = similarity.euclidean_distance_matrix(X_pequeno)
        assert D.shape == (4, 4)
        for i in range(4):
            assert D[i, i] == pytest.approx(0.0)

    def test_manhattan_matriz(self, X_pequeno: np.ndarray) -> None:
        D = similarity.manhattan_distance_matrix(X_pequeno)
        assert D.shape == (4, 4)
        for i in range(4):
            assert D[i, i] == pytest.approx(0.0)

    def test_aceita_sparse(self, X_pequeno: np.ndarray) -> None:
        X_sparse = sparse.csr_matrix(X_pequeno)
        S_sparse = similarity.cosine_similarity_matrix(X_sparse)
        S_dense = similarity.cosine_similarity_matrix(X_pequeno)
        np.testing.assert_allclose(S_sparse, S_dense)


# =============================================================================
# top_k_similares (maior é melhor)
# =============================================================================
class TestTopKSimilares:
    def test_retorna_k_itens_ordenados_desc(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
        top = similarity.top_k_similares(scores, k=3)
        assert len(top) == 3
        # Ordem: 0.9 (idx=1), 0.7 (idx=3), 0.5 (idx=2)
        assert top[0] == (1, pytest.approx(0.9))
        assert top[1] == (3, pytest.approx(0.7))
        assert top[2] == (2, pytest.approx(0.5))

    def test_excluir_indice_ignora_esse(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
        top = similarity.top_k_similares(scores, k=2, excluir_indice=1)
        indices = [idx for idx, _ in top]
        assert 1 not in indices

    def test_excluir_multiplos_indices(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
        top = similarity.top_k_similares(scores, k=3, excluir_indices=[1, 3])
        indices = [idx for idx, _ in top]
        assert 1 not in indices
        assert 3 not in indices

    def test_k_maior_que_n_disponivel(self) -> None:
        scores = np.array([0.1, 0.2, 0.3])
        top = similarity.top_k_similares(scores, k=100)
        assert len(top) == 3

    def test_k_zero_ou_negativo_erro(self) -> None:
        scores = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="positivo"):
            similarity.top_k_similares(scores, k=0)

    def test_scores_2d_erro(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            similarity.top_k_similares(np.array([[0.1, 0.2]]), k=1)


# =============================================================================
# top_k_mais_proximos (menor é melhor)
# =============================================================================
class TestTopKMaisProximos:
    def test_retorna_k_itens_ordenados_asc(self) -> None:
        distancias = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        top = similarity.top_k_mais_proximos(distancias, k=3)
        assert len(top) == 3
        # Ordem: 1.0 (idx=1), 2.0 (idx=3), 3.0 (idx=2)
        assert top[0] == (1, pytest.approx(1.0))
        assert top[1] == (3, pytest.approx(2.0))
        assert top[2] == (2, pytest.approx(3.0))

    def test_excluir_indice(self) -> None:
        distancias = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        top = similarity.top_k_mais_proximos(distancias, k=2, excluir_indice=1)
        indices = [idx for idx, _ in top]
        assert 1 not in indices
        # Sem o menor (idx=1), os dois menores são idx=3 (2.0) e idx=2 (3.0)
        assert top[0][0] == 3
        assert top[1][0] == 2


# =============================================================================
# Coerência entre implementações manuais e matriciais
# =============================================================================
class TestCoerencia:
    """Verifica que as implementações manuais concordam com as matriciais."""

    def test_cosine_manual_vs_sklearn(self) -> None:
        u = np.array([1.0, 2.0, 3.0, 0.0, 1.0])
        v = np.array([2.0, 0.0, 1.0, 4.0, 1.0])

        s_manual = similarity.cosine_similarity_pair(u, v)
        S_matriz = similarity.cosine_similarity_matrix(
            u.reshape(1, -1), v.reshape(1, -1)
        )
        assert s_manual == pytest.approx(S_matriz[0, 0])

    def test_euclidean_manual_vs_sklearn(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 6.0, 8.0])
        d_manual = similarity.euclidean_distance_pair(u, v)
        D_matriz = similarity.euclidean_distance_matrix(
            u.reshape(1, -1), v.reshape(1, -1)
        )
        assert d_manual == pytest.approx(D_matriz[0, 0])

    def test_manhattan_manual_vs_sklearn(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 6.0, 8.0])
        d_manual = similarity.manhattan_distance_pair(u, v)
        D_matriz = similarity.manhattan_distance_matrix(
            u.reshape(1, -1), v.reshape(1, -1)
        )
        assert d_manual == pytest.approx(D_matriz[0, 0])
