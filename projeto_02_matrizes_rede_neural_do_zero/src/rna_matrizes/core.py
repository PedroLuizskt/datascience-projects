"""Núcleo matemático: rede neural binária de camada única via matrizes.

Este módulo implementa a versão amadurecida do algoritmo apresentado no
Projeto 2 da pós-graduação em Ciência de Dados da Data Science Academy
(disciplina *Matemática e Estatística Aplicada Para Data Science, Machine
Learning e IA*, Cap09). O algoritmo do DSA original é uma *regressão
logística com gradiente descendente*, apresentada didaticamente como
"rede neural artificial de camada única" para exercitar as operações com
matrizes fundamentais: produto matriz-vetor, transposta, gradiente
vetorizado.

Fidelidade e amadurecimento
---------------------------
A matemática do algoritmo é preservada 100% em relação ao original DSA
(``AlgoritmoNeuralNetworkDSA``). As mudanças são estritamente de qualidade
de engenharia, sem alterar o comportamento algorítmico quando os mesmos
hiperparâmetros e dados são passados:

1. **Sem** ``print()`` no meio do :meth:`fit`. O treino agora pode ser
   usado programaticamente sem poluir stdout. Um parâmetro ``verbose``
   habilita mensagens INFO via logging padrão quando desejado.
2. **Função de custo BCE explícita**. O DSA original só computava o
   gradiente da perda, sem nunca calcular a perda em si. Este módulo
   expõe :meth:`funcao_custo_bce` e grava seu valor a cada iteração em
   :attr:`historico_perda_`.
3. **Detecção de convergência via tolerância**. Se |ΔBCE| < ``tolerancia``
   entre iterações consecutivas, o treino para. Isso corta iterações
   desnecessárias mantendo compatibilidade com o comportamento de "número
   fixo de iterações" quando ``tolerancia=0.0``.
4. **Novo método** :meth:`predict_proba` para retornar probabilidades
   contínuas — essencial para métricas que dependem de score (ROC-AUC,
   PR-AUC), assunto da Fase 2.B.

Formalização matemática
-----------------------
Para :math:`n` amostras e :math:`d` atributos, com matriz de entrada
:math:`X \\in \\mathbb{R}^{n \\times d}`, vetor de rótulos
:math:`\\mathbf{y} \\in \\{0, 1\\}^n`, pesos :math:`\\mathbf{w} \\in \\mathbb{R}^d`
e viés :math:`b \\in \\mathbb{R}`, o forward pass é:

.. math::
    \\mathbf{z} = X \\mathbf{w} + b
    \\qquad
    \\hat{\\mathbf{y}} = \\sigma(\\mathbf{z}) = \\frac{1}{1 + e^{-\\mathbf{z}}}

A função de custo é a *Binary Cross-Entropy* (BCE), calculada por amostra
e agregada como média:

.. math::
    \\mathcal{L}(\\mathbf{w}, b) =
    -\\frac{1}{n} \\sum_{i=1}^{n} \\left[
        y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i)
    \\right]

Os gradientes de :math:`\\mathcal{L}` em relação a :math:`\\mathbf{w}` e
:math:`b` são:

.. math::
    \\nabla_{\\mathbf{w}} \\mathcal{L} = \\frac{1}{n} X^\\top (\\hat{\\mathbf{y}} - \\mathbf{y})
    \\qquad
    \\nabla_b \\mathcal{L} = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i)

O update do gradiente descendente é:

.. math::
    \\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla_{\\mathbf{w}} \\mathcal{L}
    \\qquad
    b \\leftarrow b - \\eta \\nabla_b \\mathcal{L}

onde :math:`\\eta` é a taxa de aprendizado. Todas essas operações estão
implementadas com NumPy em :meth:`fit`.

Exemplo
-------

    >>> import numpy as np
    >>> from rna_matrizes import RedeNeuralBinaria
    >>> X = np.array([[1, 2.5], [2, 3], [3, 5], [1, 4], [5, 6], [6, 7]])
    >>> y = np.array([0, 0, 1, 0, 1, 1])
    >>> modelo = RedeNeuralBinaria(taxa_aprendizado=0.01, num_iteracoes=1000)
    >>> modelo.fit(X, y)
    >>> modelo.predict(np.array([[1.5, 2], [4, 5.5]]))
    array([0, 1])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rna_matrizes import config

logger = config.get_logger(__name__)


class RedeNeuralBinaria:
    """Classificador binário via gradiente descendente com ativação sigmoide.

    Adaptação amadurecida da classe ``AlgoritmoNeuralNetworkDSA`` do projeto
    original da Data Science Academy. Consulte o docstring do módulo para
    a formalização matemática completa.

    Parameters
    ----------
    taxa_aprendizado : float, default=0.01
        Passo do gradiente descendente (:math:`\\eta` na notação matemática).
        Valores muito altos causam divergência; muito baixos, convergência
        lenta. O default 0.01 é o mesmo usado no projeto DSA original.
    num_iteracoes : int, default=1000
        Número **máximo** de iterações do loop de treino. Pode terminar
        antes por convergência (ver ``tolerancia``).
    tolerancia : float, default=1e-6
        Se a variação absoluta da BCE entre duas iterações consecutivas
        for menor que este valor, o treino é interrompido (early stopping).
        Passe ``0.0`` para desabilitar e forçar exatamente ``num_iteracoes``.
    verbose : bool, default=False
        Se ``True``, emite mensagens INFO via logging a cada 100 iterações
        e no início/fim do treino. Se ``False``, silencioso.

    Attributes
    ----------
    pesos_ : ndarray of shape (n_features,)
        Vetor de pesos aprendidos. Disponível após ``fit``.
    bias_ : float
        Viés escalar aprendido. Disponível após ``fit``.
    historico_perda_ : list of float
        Valor da BCE em cada iteração do treino. Comprimento igual ao
        número de iterações efetivamente executadas (pode ser menor que
        ``num_iteracoes`` se houve convergência).
    n_iteracoes_executadas_ : int
        Número de iterações efetivamente rodadas. Igual a ``num_iteracoes``
        se não houve early stopping.
    convergiu_ : bool
        Se ``True``, o treino parou por convergência (|ΔBCE| < tolerancia).
        Se ``False``, atingiu o limite de ``num_iteracoes``.

    Notes
    -----
    A classe segue a convenção ``sklearn``-style de sufixo ``_`` para
    atributos aprendidos após ``fit``, embora não herde de
    ``sklearn.base.BaseEstimator`` (para manter o núcleo em NumPy puro,
    fiel ao propósito didático do projeto DSA).
    """

    def __init__(
        self,
        taxa_aprendizado: float = config.DEFAULT_TAXA_APRENDIZADO,
        num_iteracoes: int = config.DEFAULT_NUM_ITERACOES,
        tolerancia: float = config.DEFAULT_TOLERANCIA,
        verbose: bool = False,
    ) -> None:
        if taxa_aprendizado <= 0:
            raise ValueError(
                f"taxa_aprendizado deve ser positiva, recebido {taxa_aprendizado}"
            )
        if num_iteracoes <= 0:
            raise ValueError(
                f"num_iteracoes deve ser positivo, recebido {num_iteracoes}"
            )
        if tolerancia < 0:
            raise ValueError(
                f"tolerancia deve ser >= 0, recebido {tolerancia}"
            )

        self.taxa_aprendizado = taxa_aprendizado
        self.num_iteracoes = num_iteracoes
        self.tolerancia = tolerancia
        self.verbose = verbose

        # Atributos aprendidos, populados por fit()
        self.pesos_: np.ndarray | None = None
        self.bias_: float | None = None
        self.historico_perda_: list[float] = []
        self.n_iteracoes_executadas_: int = 0
        self.convergiu_: bool = False

    # -------------------------------------------------------------------------
    # Função de ativação
    # -------------------------------------------------------------------------
    @staticmethod
    def func_activation_sigmoid(z: np.ndarray) -> np.ndarray:
        """Aplica a função sigmoide de forma numericamente estável.

        Fórmula clássica: :math:`\\sigma(z) = 1 / (1 + e^{-z})`.

        Para valores de ``z`` muito grandes em magnitude, o cálculo direto
        pode gerar overflow em ``np.exp``. Esta implementação usa a
        identidade :math:`\\sigma(z) = e^z / (1 + e^z)` para valores
        negativos, evitando o overflow sem alterar o resultado matemático
        para tolerâncias de ``float64``.

        Parameters
        ----------
        z : ndarray
            Ativações pré-sigmoide, de qualquer shape.

        Returns
        -------
        ndarray
            Sigmoide aplicada elemento a elemento, valores em (0, 1).
        """
        # Usamos máscaras explícitas ao invés de np.where para evitar avaliação
        # dos dois ramos em valores onde geraria overflow. np.where computa
        # ambos e depois seleciona; máscaras computam apenas o necessário.
        z = np.asarray(z, dtype=np.float64)
        resultado = np.empty_like(z)
        positivos = z >= 0
        negativos = ~positivos

        # Para z >= 0: forma padrão 1/(1+e^-z)
        resultado[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
        # Para z < 0: forma alternativa e^z/(1+e^z), equivalente mas sem overflow
        exp_z_neg = np.exp(z[negativos])
        resultado[negativos] = exp_z_neg / (1.0 + exp_z_neg)

        return resultado

    # -------------------------------------------------------------------------
    # Função de custo
    # -------------------------------------------------------------------------
    @staticmethod
    def funcao_custo_bce(
        y_verdadeiro: np.ndarray,
        y_probabilidade: np.ndarray,
        epsilon: float = 1e-15,
    ) -> float:
        """Calcula a Binary Cross-Entropy (BCE) média entre alvos e previsões.

        .. math::
            \\mathcal{L} = -\\frac{1}{n} \\sum_{i=1}^{n} \\left[
                y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i)
            \\right]

        Um clip em ``epsilon`` é aplicado às probabilidades para evitar
        ``log(0)`` quando o modelo produz previsão saturada. Valor
        pequeno o suficiente para não alterar o resultado quando as
        probabilidades estão bem no interior de (0, 1).

        Parameters
        ----------
        y_verdadeiro : ndarray of shape (n,)
            Rótulos binários (0 ou 1).
        y_probabilidade : ndarray of shape (n,)
            Probabilidades previstas da classe positiva, valores em (0, 1).
        epsilon : float, default=1e-15
            Clipping para estabilidade numérica.

        Returns
        -------
        float
            BCE média sobre as ``n`` amostras.

        Raises
        ------
        ValueError
            Se os arrays não tiverem shapes compatíveis.
        """
        y_verdadeiro = np.asarray(y_verdadeiro, dtype=np.float64)
        y_probabilidade = np.asarray(y_probabilidade, dtype=np.float64)
        if y_verdadeiro.shape != y_probabilidade.shape:
            raise ValueError(
                f"Shapes incompatíveis: y_verdadeiro={y_verdadeiro.shape}, "
                f"y_probabilidade={y_probabilidade.shape}"
            )

        p = np.clip(y_probabilidade, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(y_verdadeiro * np.log(p) + (1.0 - y_verdadeiro) * np.log(1.0 - p))
        )

    # -------------------------------------------------------------------------
    # Treino: forward + backward + update
    # -------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RedeNeuralBinaria":
        """Ajusta os pesos e o viés minimizando a BCE via gradiente descendente.

        Para cada iteração, executa três operações matriciais fundamentais:

        1. **Forward pass**: :math:`\\hat{\\mathbf{y}} = \\sigma(X\\mathbf{w} + b)`
        2. **Backward pass**: :math:`\\nabla_{\\mathbf{w}} = \\frac{1}{n} X^\\top (\\hat{\\mathbf{y}} - \\mathbf{y})`
        3. **Update**: :math:`\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla_{\\mathbf{w}}`

        Ao final de cada iteração, a BCE é calculada e armazenada em
        :attr:`historico_perda_`. Se a variação absoluta da BCE entre
        duas iterações consecutivas ficar abaixo de :attr:`tolerancia`,
        o treino é interrompido (early stopping).

        Parameters
        ----------
        X : ndarray of shape (n_amostras, n_features)
            Matriz de entrada.
        y : ndarray of shape (n_amostras,)
            Rótulos binários (0 ou 1).

        Returns
        -------
        self : RedeNeuralBinaria
            A própria instância, permitindo encadeamento
            (``modelo.fit(X, y).predict(X_novo)``).

        Raises
        ------
        ValueError
            Se ``X`` não for 2D, ``y`` não for 1D, ou seus tamanhos não
            baterem.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X deve ser 2D, recebido ndim={X.ndim}")
        if y.ndim != 1:
            raise ValueError(f"y deve ser 1D, recebido ndim={y.ndim}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X e y devem ter o mesmo número de amostras: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        n_amostras, n_atributos = X.shape

        # Inicialização com zeros (fiel ao DSA original)
        self.pesos_ = np.zeros(n_atributos, dtype=np.float64)
        self.bias_ = 0.0
        self.historico_perda_ = []
        self.convergiu_ = False

        if self.verbose:
            logger.info(
                "[FIT] Iniciando treino: %d amostras, %d features, "
                "taxa_aprendizado=%.4f, num_iteracoes=%d, tolerancia=%.2e",
                n_amostras, n_atributos,
                self.taxa_aprendizado, self.num_iteracoes, self.tolerancia,
            )

        perda_anterior = np.inf

        for iteracao in range(self.num_iteracoes):
            # Forward pass: z = X @ w + b, depois sigmoide
            z = np.dot(X, self.pesos_) + self.bias_
            y_prob = self.func_activation_sigmoid(z)

            # Custo BCE — calculado e gravado antes do update dos parâmetros
            perda_atual = self.funcao_custo_bce(y, y_prob)
            self.historico_perda_.append(perda_atual)

            # Backward pass: gradientes da BCE em relação a pesos e bias
            erro = y_prob - y
            gradiente_pesos = (1.0 / n_amostras) * np.dot(X.T, erro)
            gradiente_bias = (1.0 / n_amostras) * np.sum(erro)

            # Update dos parâmetros
            self.pesos_ -= self.taxa_aprendizado * gradiente_pesos
            self.bias_ -= self.taxa_aprendizado * gradiente_bias

            # Log periódico se verbose
            if self.verbose and (iteracao % 100 == 0 or iteracao == self.num_iteracoes - 1):
                logger.info(
                    "[FIT] Iteração %4d/%d — BCE=%.6f",
                    iteracao, self.num_iteracoes, perda_atual,
                )

            # Detecção de convergência
            if self.tolerancia > 0 and abs(perda_anterior - perda_atual) < self.tolerancia:
                self.convergiu_ = True
                if self.verbose:
                    logger.info(
                        "[FIT] Convergiu na iteração %d (|ΔBCE|=%.2e < %.2e)",
                        iteracao, abs(perda_anterior - perda_atual), self.tolerancia,
                    )
                break
            perda_anterior = perda_atual

        self.n_iteracoes_executadas_ = len(self.historico_perda_)

        if self.verbose:
            logger.info(
                "[FIT] Treino concluído: %d iterações efetivas, BCE final=%.6f, convergiu=%s",
                self.n_iteracoes_executadas_,
                self.historico_perda_[-1] if self.historico_perda_ else float("nan"),
                self.convergiu_,
            )

        return self

    # -------------------------------------------------------------------------
    # Previsão
    # -------------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """Levanta ``NotFittedError`` (via ``RuntimeError``) se ainda não houve fit."""
        if self.pesos_ is None or self.bias_ is None:
            raise RuntimeError(
                "Modelo não treinado. Chame fit(X, y) antes de predict()."
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna as probabilidades previstas da classe positiva.

        Útil para cálculo de métricas que dependem de score
        (ROC-AUC, PR-AUC) e para calibração posterior. As probabilidades
        são o resultado da sigmoide aplicada ao forward pass.

        Parameters
        ----------
        X : ndarray of shape (n_amostras, n_features)

        Returns
        -------
        ndarray of shape (n_amostras,)
            Probabilidades em (0, 1).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X deve ser 2D, recebido ndim={X.ndim}")
        z = np.dot(X, self.pesos_) + self.bias_
        return self.func_activation_sigmoid(z)

    def predict(
        self,
        X: np.ndarray,
        limiar: float = config.DEFAULT_LIMIAR_DECISAO,
    ) -> np.ndarray:
        """Retorna as classes previstas (0 ou 1) aplicando um limiar de decisão.

        Parameters
        ----------
        X : ndarray of shape (n_amostras, n_features)
        limiar : float, default=0.5
            Corte para converter probabilidade em classe. Valores acima do
            limiar viram 1, abaixo (ou iguais) viram 0. Ajustar este
            parâmetro é uma técnica clássica para lidar com custo assimétrico
            de falsos positivos vs falsos negativos, essencial em detecção
            de fraude (assunto da Fase 2.B).

        Returns
        -------
        ndarray of shape (n_amostras,)
            Classes previstas, com dtype ``int64``.
        """
        probs = self.predict_proba(X)
        return (probs > limiar).astype(np.int64)

    # -------------------------------------------------------------------------
    # Introspecção
    # -------------------------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        """Retorna os hiperparâmetros do modelo como dicionário.

        Compatível com a API de estimators do scikit-learn (embora esta
        classe não herde de ``BaseEstimator``).
        """
        return {
            "taxa_aprendizado": self.taxa_aprendizado,
            "num_iteracoes": self.num_iteracoes,
            "tolerancia": self.tolerancia,
            "verbose": self.verbose,
        }

    def __repr__(self) -> str:
        return (
            f"RedeNeuralBinaria(taxa_aprendizado={self.taxa_aprendizado}, "
            f"num_iteracoes={self.num_iteracoes}, "
            f"tolerancia={self.tolerancia})"
        )
