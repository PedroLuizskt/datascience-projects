"""Testes do módulo `rna_matrizes.core` — classe RedeNeuralBinaria.

Cobre validação de hiperparâmetros no construtor, sigmoide (correção +
estabilidade numérica), função de custo BCE (correção + robustez a
saturação), forward pass, backward pass, convergência, early stopping,
predict e predict_proba. Fecha com um teste de coerência que compara
nosso resultado com `sklearn.linear_model.LogisticRegression` em um
dataset comum, garantindo que a matemática está correta.
"""

from __future__ import annotations

import numpy as np
import pytest

from rna_matrizes import RedeNeuralBinaria


# =============================================================================
# Validação de hiperparâmetros
# =============================================================================
class TestConstrucao:
    def test_defaults_ok(self) -> None:
        modelo = RedeNeuralBinaria()
        assert modelo.taxa_aprendizado > 0
        assert modelo.num_iteracoes > 0
        assert modelo.pesos_ is None
        assert modelo.bias_ is None
        assert modelo.historico_perda_ == []

    def test_hiperparametros_customizados(self) -> None:
        modelo = RedeNeuralBinaria(
            taxa_aprendizado=0.5, num_iteracoes=200, tolerancia=1e-4, verbose=True
        )
        assert modelo.taxa_aprendizado == 0.5
        assert modelo.num_iteracoes == 200
        assert modelo.tolerancia == 1e-4
        assert modelo.verbose is True

    @pytest.mark.parametrize("valor_invalido", [0, -0.01, -1.0])
    def test_taxa_aprendizado_nao_positiva_levanta_erro(self, valor_invalido) -> None:
        with pytest.raises(ValueError, match="taxa_aprendizado"):
            RedeNeuralBinaria(taxa_aprendizado=valor_invalido)

    @pytest.mark.parametrize("valor_invalido", [0, -1, -100])
    def test_num_iteracoes_nao_positivo_levanta_erro(self, valor_invalido) -> None:
        with pytest.raises(ValueError, match="num_iteracoes"):
            RedeNeuralBinaria(num_iteracoes=valor_invalido)

    def test_tolerancia_negativa_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="tolerancia"):
            RedeNeuralBinaria(tolerancia=-0.001)

    def test_repr_legivel(self) -> None:
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.05, num_iteracoes=500)
        repr_str = repr(modelo)
        assert "RedeNeuralBinaria" in repr_str
        assert "0.05" in repr_str
        assert "500" in repr_str

    def test_get_params(self) -> None:
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.02, num_iteracoes=100)
        params = modelo.get_params()
        assert set(params.keys()) == {
            "taxa_aprendizado", "num_iteracoes", "tolerancia", "verbose"
        }
        assert params["taxa_aprendizado"] == 0.02


# =============================================================================
# Função de ativação sigmoide
# =============================================================================
class TestSigmoide:
    def test_sigmoide_de_zero_igual_meio(self) -> None:
        assert RedeNeuralBinaria.func_activation_sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoide_de_valor_positivo_grande(self) -> None:
        assert RedeNeuralBinaria.func_activation_sigmoid(np.array([100.0]))[0] == pytest.approx(1.0)

    def test_sigmoide_de_valor_negativo_grande(self) -> None:
        assert RedeNeuralBinaria.func_activation_sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0)

    def test_sigmoide_valores_intermediarios(self) -> None:
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        esperado = np.array([0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708])
        np.testing.assert_allclose(
            RedeNeuralBinaria.func_activation_sigmoid(z), esperado, rtol=1e-6
        )

    def test_sigmoide_estabilidade_numerica_valores_extremos(self) -> None:
        """Regressão: sigmoide direta gera overflow com z muito grande em magnitude."""
        z = np.array([-1000.0, 1000.0, -500.0, 500.0])
        resultado = RedeNeuralBinaria.func_activation_sigmoid(z)
        # Não deve ter NaN nem inf
        assert not np.any(np.isnan(resultado))
        assert not np.any(np.isinf(resultado))
        # Todos os valores devem estar em [0, 1]
        assert np.all((resultado >= 0.0) & (resultado <= 1.0))

    def test_sigmoide_simetria(self) -> None:
        """σ(-z) == 1 - σ(z) — propriedade fundamental da sigmoide."""
        z = np.array([-3.0, -0.5, 1.5, 4.0])
        pos = RedeNeuralBinaria.func_activation_sigmoid(z)
        neg = RedeNeuralBinaria.func_activation_sigmoid(-z)
        np.testing.assert_allclose(neg, 1.0 - pos, rtol=1e-10)


# =============================================================================
# Função de custo BCE
# =============================================================================
class TestFuncaoCustoBCE:
    def test_previsao_perfeita_custo_zero(self) -> None:
        y = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        # Clipping impede zero exato, mas resultado deve ser praticamente zero
        assert RedeNeuralBinaria.funcao_custo_bce(y, y_prob) < 1e-10

    def test_previsao_totalmente_errada_custo_alto(self) -> None:
        y = np.array([0, 1])
        y_prob = np.array([1.0, 0.0])
        # Com clipping em 1e-15, esperamos custo ≈ -log(1e-15) ≈ 34.5
        custo = RedeNeuralBinaria.funcao_custo_bce(y, y_prob)
        assert custo > 30.0

    def test_previsao_meio_meio_custo_conhecido(self) -> None:
        """BCE de p=0.5 constante == log(2) ≈ 0.693, independente do y."""
        y = np.array([0, 1, 0, 1, 1])
        y_prob = np.full(5, 0.5)
        assert RedeNeuralBinaria.funcao_custo_bce(y, y_prob) == pytest.approx(np.log(2), rel=1e-6)

    def test_shapes_incompativeis_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="incompat"):
            RedeNeuralBinaria.funcao_custo_bce(np.array([0, 1, 0]), np.array([0.5, 0.5]))

    def test_bce_nao_negativa(self) -> None:
        rng = np.random.default_rng(seed=0)
        y = rng.integers(0, 2, size=100)
        y_prob = rng.uniform(0.01, 0.99, size=100)
        assert RedeNeuralBinaria.funcao_custo_bce(y, y_prob) >= 0


# =============================================================================
# Fit — treino
# =============================================================================
class TestFit:
    def test_fit_popula_atributos_aprendidos(self, dataset_dsa_original) -> None:
        X_treino, y_treino, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=100, tolerancia=0)
        modelo.fit(X_treino, y_treino)
        assert modelo.pesos_ is not None
        assert modelo.bias_ is not None
        assert modelo.pesos_.shape == (X_treino.shape[1],)
        assert isinstance(modelo.bias_, float)

    def test_fit_retorna_self_para_encadeamento(self, dataset_dsa_original) -> None:
        X, y, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=10)
        resultado = modelo.fit(X, y)
        assert resultado is modelo

    def test_historico_perda_comprimento_correto(self, dataset_dsa_original) -> None:
        X, y, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=50, tolerancia=0)
        modelo.fit(X, y)
        assert len(modelo.historico_perda_) == 50
        assert modelo.n_iteracoes_executadas_ == 50

    def test_historico_perda_decrescente(self, dataset_linearmente_separavel) -> None:
        """A BCE deve diminuir monotonamente (com pequenas oscilações permitidas)."""
        X, y = dataset_linearmente_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=200, tolerancia=0)
        modelo.fit(X, y)
        # Primeira perda >> última perda
        assert modelo.historico_perda_[0] > modelo.historico_perda_[-1]
        # Última perda menor que 10% da primeira
        assert modelo.historico_perda_[-1] < 0.1 * modelo.historico_perda_[0]

    def test_convergencia_dispara_early_stopping(self, dataset_linearmente_separavel) -> None:
        X, y = dataset_linearmente_separavel
        # Tolerância alta força convergência rápida
        modelo = RedeNeuralBinaria(
            taxa_aprendizado=0.1, num_iteracoes=10000, tolerancia=1e-3
        )
        modelo.fit(X, y)
        assert modelo.convergiu_ is True
        assert modelo.n_iteracoes_executadas_ < 10000

    def test_tolerancia_zero_forca_todas_iteracoes(self, dataset_dsa_original) -> None:
        X, y, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=30, tolerancia=0.0)
        modelo.fit(X, y)
        assert modelo.n_iteracoes_executadas_ == 30
        assert modelo.convergiu_ is False

    def test_x_1d_levanta_erro(self) -> None:
        modelo = RedeNeuralBinaria(num_iteracoes=10)
        X_ruim = np.array([1.0, 2.0, 3.0])  # 1D
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="X deve ser 2D"):
            modelo.fit(X_ruim, y)

    def test_y_2d_levanta_erro(self) -> None:
        modelo = RedeNeuralBinaria(num_iteracoes=10)
        X = np.array([[1.0], [2.0], [3.0]])
        y_ruim = np.array([[0], [1], [0]])  # 2D
        with pytest.raises(ValueError, match="y deve ser 1D"):
            modelo.fit(X, y_ruim)

    def test_shapes_x_y_incompativeis(self) -> None:
        modelo = RedeNeuralBinaria(num_iteracoes=10)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1])  # tamanho errado
        with pytest.raises(ValueError, match="mesmo número"):
            modelo.fit(X, y)

    def test_determinismo(self, dataset_linearmente_separavel) -> None:
        """Duas instâncias iguais treinadas com mesmo dado dão resultado idêntico.

        Como a inicialização é com zeros (não aleatória), o determinismo é
        automático — este teste protege contra regressões que introduzam
        aleatoriedade sem controle.
        """
        X, y = dataset_linearmente_separavel
        m1 = RedeNeuralBinaria(taxa_aprendizado=0.05, num_iteracoes=200, tolerancia=0)
        m2 = RedeNeuralBinaria(taxa_aprendizado=0.05, num_iteracoes=200, tolerancia=0)
        m1.fit(X, y)
        m2.fit(X, y)
        np.testing.assert_array_equal(m1.pesos_, m2.pesos_)
        assert m1.bias_ == m2.bias_


# =============================================================================
# Predict e predict_proba
# =============================================================================
class TestPredict:
    def test_predict_sem_fit_levanta_erro(self) -> None:
        modelo = RedeNeuralBinaria()
        with pytest.raises(RuntimeError, match="não treinado"):
            modelo.predict(np.array([[1.0, 2.0]]))

    def test_predict_proba_sem_fit_levanta_erro(self) -> None:
        modelo = RedeNeuralBinaria()
        with pytest.raises(RuntimeError, match="não treinado"):
            modelo.predict_proba(np.array([[1.0, 2.0]]))

    def test_predict_shape_e_tipo(self, dataset_linearmente_separavel) -> None:
        X, y = dataset_linearmente_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=200)
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        assert y_pred.shape == (X.shape[0],)
        assert y_pred.dtype == np.int64
        assert set(np.unique(y_pred).tolist()).issubset({0, 1})

    def test_predict_proba_valores_em_zero_um(self, dataset_linearmente_separavel) -> None:
        X, y = dataset_linearmente_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=200)
        modelo.fit(X, y)
        probs = modelo.predict_proba(X)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_predict_coerente_com_predict_proba_e_limiar(self, dataset_linearmente_separavel) -> None:
        X, y = dataset_linearmente_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=200)
        modelo.fit(X, y)
        probs = modelo.predict_proba(X)
        preds = modelo.predict(X, limiar=0.5)
        np.testing.assert_array_equal(preds, (probs > 0.5).astype(np.int64))

    def test_limiar_customizado_afeta_previsao(self, dataset_nao_separavel) -> None:
        X, y = dataset_nao_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=200)
        modelo.fit(X, y)
        # Limiar baixo → mais previsões positivas
        preds_baixo = modelo.predict(X, limiar=0.2)
        preds_padrao = modelo.predict(X, limiar=0.5)
        preds_alto = modelo.predict(X, limiar=0.8)
        assert preds_baixo.sum() >= preds_padrao.sum() >= preds_alto.sum()

    def test_x_teste_1d_levanta_erro(self, dataset_dsa_original) -> None:
        X, y, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=10).fit(X, y)
        with pytest.raises(ValueError, match="X deve ser 2D"):
            modelo.predict(np.array([1.0, 2.0]))


# =============================================================================
# Casos ponta a ponta e coerência com sklearn
# =============================================================================
class TestPipelineCompleto:
    def test_dataset_dsa_original_acuracia_100(self, dataset_dsa_original) -> None:
        """No mini-dataset do DSA, o modelo deve acertar as 2 amostras de teste."""
        X_treino, y_treino, X_teste, y_teste = dataset_dsa_original
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.01, num_iteracoes=1000, tolerancia=0)
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)
        np.testing.assert_array_equal(y_pred, y_teste)

    def test_dataset_separavel_acuracia_alta(self, dataset_linearmente_separavel) -> None:
        X, y = dataset_linearmente_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=500, tolerancia=0)
        modelo.fit(X, y)
        y_pred = modelo.predict(X)
        acuracia = float(np.mean(y_pred == y))
        assert acuracia > 0.98

    def test_coerencia_com_sklearn_logistic_regression(
        self, dataset_linearmente_separavel
    ) -> None:
        """Regressão logística nossa vs. sklearn — devem convergir para pesos similares.

        Sklearn usa por default regularização L2 (C=1.0); nossa
        implementação não tem regularização. Para tornar comparável,
        passamos C muito alto ao sklearn (regularização desprezível) e
        treinamos ambos até convergência forte.
        """
        from sklearn.linear_model import LogisticRegression

        X, y = dataset_linearmente_separavel

        nosso = RedeNeuralBinaria(
            taxa_aprendizado=0.5, num_iteracoes=10000, tolerancia=1e-10
        )
        nosso.fit(X, y)

        sk = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000, tol=1e-10)
        sk.fit(X, y)

        # Não esperamos coeficientes idênticos (algoritmos diferentes,
        # convergência para regiões próximas mas não iguais em dataset
        # perfeitamente separável), mas as previsões devem coincidir.
        y_pred_nosso = nosso.predict(X)
        y_pred_sk = sk.predict(X)
        # Concordância > 98% (praticamente sempre 100% em dataset separável)
        concordancia = float(np.mean(y_pred_nosso == y_pred_sk))
        assert concordancia > 0.98

    def test_bce_final_menor_que_bce_inicial(self, dataset_nao_separavel) -> None:
        """Mesmo em dataset não separável, o treino deve reduzir a BCE."""
        X, y = dataset_nao_separavel
        modelo = RedeNeuralBinaria(taxa_aprendizado=0.1, num_iteracoes=500, tolerancia=0)
        modelo.fit(X, y)
        assert modelo.historico_perda_[-1] < modelo.historico_perda_[0]

    def test_verbose_true_loga_sem_erro(self, dataset_dsa_original, caplog) -> None:
        """Regressão: verbose=True não deve quebrar (só emitir logs)."""
        import logging as _logging

        caplog.set_level(_logging.INFO)
        X, y, _, _ = dataset_dsa_original
        modelo = RedeNeuralBinaria(num_iteracoes=200, tolerancia=0, verbose=True)
        modelo.fit(X, y)
        # Deve ter emitido pelo menos as mensagens de início e fim
        mensagens = [r.message for r in caplog.records if "[FIT]" in r.message]
        assert len(mensagens) > 0
