"""rna_matrizes — Rede Neural Artificial via Operações com Matrizes.

Pacote de adaptação didática do Projeto 2 da pós-graduação em Ciência de
Dados da Data Science Academy, exercitando operações com matrizes na
implementação de um classificador binário via gradiente descendente.

Uso típico::

    from rna_matrizes import RedeNeuralBinaria

    modelo = RedeNeuralBinaria(taxa_aprendizado=0.01, num_iteracoes=1000)
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)

    # Histórico de perda por iteração está em modelo.historico_perda_
"""

from rna_matrizes.core import RedeNeuralBinaria

__version__ = "0.1.0"
__all__ = ["RedeNeuralBinaria"]
