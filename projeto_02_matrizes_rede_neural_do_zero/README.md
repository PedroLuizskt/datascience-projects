# Projeto 02 — Rede Neural Artificial via Operações com Matrizes

Adaptação didática do Projeto 2 da pós-graduação em Ciência de Dados da [Data Science Academy](https://www.datascienceacademy.com.br) — disciplina *Matemática e Estatística Aplicada Para Data Science, Machine Learning e IA*. Este repositório reengenheira o algoritmo apresentado no Cap09 (rede neural artificial de camada única via operações matriciais em NumPy) e o aplica a um problema real de detecção de fraude em transações de cartão de crédito, com todo o rigor esperado de um projeto de portfólio profissional.

## Contexto

O projeto DSA original implementa, do zero em NumPy, uma classe `AlgoritmoNeuralNetworkDSA` que exercita as **operações com matrizes** fundamentais em algoritmos de aprendizado supervisionado: produto matriz-vetor, transposta, gradiente vetorizado, adição escalar-vetor por broadcasting. Matematicamente, o algoritmo é regressão logística com gradiente descendente e uma única ativação sigmoide — a forma mais simples possível de "rede neural", ideal para consolidar os conceitos matriciais antes de partir para multi-camadas verdadeiras.

Esta adaptação preserva 100% da matemática do original e o transforma em pacote Python testado, com:

- Núcleo matemático fiel ao DSA, sem `print()` no meio do treino, com função de custo BCE (Binary Cross-Entropy) exposta e histórico de perda gravado por iteração
- Detecção de convergência via tolerância na variação da perda
- Ingestão do dataset [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) do Kaggle (284.807 transações reais, altamente desbalanceadas)
- Métricas apropriadas para problema desbalanceado (precision, recall, F1, ROC-AUC, PR-AUC) calculadas manualmente e validadas contra `scikit-learn`
- Estratégias para lidar com o desbalanceamento severo (class weights, undersampling)
- Visualizações (curva de perda, matriz de confusão, curvas ROC e PR)
- Comparação empírica com `sklearn.linear_model.LogisticRegression` para validar equivalência
- Notebook demonstrativo end-to-end
- Apostila didática em `docs/apostila/`

## Roadmap

| Fase | Escopo | Status |
|------|--------|--------|
| 2.A | Estrutura, config, núcleo matemático fiel ao DSA (sem prints, com BCE explícita, histórico de perda, early stopping) | Concluída |
| 2.B | Ingestão do dataset Kaggle creditcardfraud, tratamento de desbalanceamento, métricas para classificação binária | A implementar |
| 2.C | API de alto nível, notebook demonstrativo end-to-end, comparação com sklearn, visualizações | A implementar |
| 2.D | Apostila didática (4 sessões) | A implementar |

## Estrutura

Projeto organizado seguindo a estrutura [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/), com pacote Python instalável em `src/` e testes em `tests/`.

```
projeto_02_matrizes_rede_neural_do_zero/
├── src/rna_matrizes/          # Pacote Python instalável
│   ├── config.py              # Configurações centralizadas
│   └── core.py                # Classe RedeNeuralBinaria (núcleo matemático)
├── tests/                     # Suite pytest
├── data/                      # Data lake local (ignorado pelo git)
│   ├── raw/                   # Dataset bruto do Kaggle
│   ├── interim/               # Transformações intermediárias
│   └── processed/             # Dataset final pronto para treino
├── docs/apostila/             # Material didático (Fase 2.D)
├── notebooks/                 # Notebooks Jupyter (Fase 2.C)
├── Makefile                   # Automacao Unix
└── tasks.ps1                  # Automacao PowerShell (Windows primary)
```

## Setup

```powershell
# Cria e ativa venv (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instala pacote em modo editable com extras dev + notebook
.\tasks.ps1 setup

# Roda a suite de testes
.\tasks.ps1 test
```

## Alvos disponíveis

```powershell
.\tasks.ps1 setup             # instala pacote + dependências
.\tasks.ps1 test              # suite offline (default)
.\tasks.ps1 test-network      # testes que precisam de rede (baixar dataset)
.\tasks.ps1 test-all          # tudo
.\tasks.ps1 lint              # ruff check
.\tasks.ps1 notebook          # jupyter lab em notebooks/
.\tasks.ps1 clean             # remove caches
```

Em Unix, os mesmos alvos estão disponíveis via `make`.

## Fidelidade ao projeto DSA original

Para preservar rastreabilidade pedagógica, cada função da classe `RedeNeuralBinaria` corresponde diretamente a um método da classe `AlgoritmoNeuralNetworkDSA` original:

| Original DSA                          | Adaptação                       | Modificação                                                                   |
|---------------------------------------|--------------------------------|-------------------------------------------------------------------------------|
| `AlgoritmoNeuralNetworkDSA.__init__`  | `RedeNeuralBinaria.__init__`   | + `tolerancia`, + `verbose`                                                   |
| `func_activation_sigmoid`             | `func_activation_sigmoid`      | Idêntica; adicionada versão numericamente estável para valores muito grandes |
| `fit`                                 | `fit`                          | Sem `print()`; grava `historico_perda_`; early stopping por tolerância        |
| `predict`                             | `predict`                      | Sem `print()`; retorna `np.ndarray` ao invés de `list`                        |
| —                                     | `predict_proba`                | Novo. Devolve probabilidades da classe positiva (útil para ROC-AUC)           |
| —                                     | `funcao_custo_bce`             | Nova. Implementa Binary Cross-Entropy explicitamente                          |

## Referência acadêmica

Data Science Academy. *Formação Cientista de Dados 4.0 — Matemática e Estatística Aplicada Para Data Science, Machine Learning e IA*. Cap09 — Projeto 2: Construindo Algoritmo de Rede Neural Artificial Através de Operações com Matrizes. Rio de Janeiro, 2024.

Kaggle. *Credit Card Fraud Detection*. Machine Learning Group ULB. [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Acesso em 2025.
