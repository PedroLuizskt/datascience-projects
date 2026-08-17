# projeto_01 — Recomendador de Municípios Brasileiros por Perfil Agropecuário

Adaptação didática do projeto Cap08 da pós-graduação em Ciência de Dados da **Data Science Academy (DSA)**. O projeto original, intitulado *"Vetores e Espaço Vetorial em Sistemas de Recomendação"*, ensina a construção de um sistema de recomendação content-based a partir da representação de itens como vetores em um espaço de alta dimensão e do cálculo de similaridade por cosseno. Esta adaptação transporta o mesmo aparato conceitual e algorítmico para o domínio agropecuário brasileiro, respondendo a uma pergunta prática: *"quais municípios brasileiros têm perfil agropecuário similar a um município de referência?"*

## Referência ao projeto original

O projeto Cap08 da DSA usa como material de trabalho o dataset TMDB 5000 Movies, com cerca de 4.800 títulos cinematográficos representados por metadados textuais (overview, gêneros, palavras-chave, elenco e equipe). Aplica sobre esses metadados o fluxo canônico de vetorização — parsing de estruturas aninhadas, concatenação de campos, tokenização, stemming, `CountVectorizer` do scikit-learn — e calcula a matriz de similaridade cosseno, a partir da qual um sistema de recomendação content-based devolve os títulos mais próximos de um filme consultado.

O que esta adaptação preserva integralmente do original: o fluxo pedagógico do módulo (representação vetorial, vetorização, similaridade cosseno, recomendador content-based) e a arquitetura de features "tags" mistas que agregam múltiplas dimensões textuais e categóricas em um único campo de entrada para o vetorizador. O que esta adaptação substitui: o dataset (municípios brasileiros no lugar de filmes), a fonte de dados (API SIDRA do IBGE no lugar de CSV pré-empacotado do Kaggle), e o stemmer (RSLPStemmer para português no lugar do PorterStemmer para inglês).

Nenhum artefato do projeto original da DSA é redistribuído neste repositório. O código-fonte, dados e documentação aqui presentes são autorais.

## Pergunta que o projeto responde

Dado um município brasileiro qualquer, o sistema retorna os `k` municípios mais similares em termos de perfil agropecuário. A definição operacional de "perfil agropecuário" é o conjunto de características refletidas nos dados da Pesquisa da Pecuária Municipal (PPM) do IBGE, complementadas por informações da divisão territorial (região, unidade da federação, mesorregião, microrregião).

Exemplo de uso pretendido: um consultor agrícola que trabalha em Cambuquira (MG) quer identificar municípios com perfil similar em outras regiões do país, seja para benchmarking de produtividade, seja para planejar expansão comercial. Ele consulta o sistema com o nome do município e recebe os cinco mais similares, ordenados por distância cosseno crescente.

## Fonte de dados

**Pesquisa da Pecuária Municipal (PPM), IBGE, tabela SIDRA 3939.** Fornece o efetivo dos rebanhos por município para bovinos, bubalinos, suínos, matrizes de suínos, equinos, ovinos, caprinos, galináceos, galinhas e codornas. Periodicidade anual, cobertura nacional para todos os 5.570 municípios brasileiros. Ano de referência configurável via variável de ambiente `PPM_ANO` (default: 2023). Acesso via biblioteca `sidrapy` sobre a API pública do IBGE. Licença: dados abertos do IBGE, uso permitido com citação da fonte.

**Divisão Territorial Brasileira, IBGE, API Localidades.** Fornece a hierarquia territorial (região, UF, mesorregião, microrregião, município) que compõe as features textuais contextuais de cada município. Acesso via requests HTTP diretos ao endpoint `https://servicodados.ibge.gov.br/api/v1/localidades/municipios`.

Detalhes técnicos da coleta serão documentados no módulo `src/rec_agro_br/dataset.py` (a ser implementado na Fase 1.B).

## Estrutura do projeto

```
projeto_01_espaco_vetorial_recomendador_municipios_agro/
├── data/
│   ├── raw/            # respostas brutas das APIs IBGE
│   ├── interim/        # transformações intermediárias
│   ├── processed/      # dataset final para vetorização
│   └── external/       # dados externos (reservado)
├── docs/apostila/      # apostila didática por sessão
├── notebooks/          # notebooks executáveis end-to-end
├── references/         # documentação técnica e papers
├── reports/figures/    # figuras geradas por notebooks e relatórios
├── src/rec_agro_br/    # pacote Python instalável
├── tests/              # suíte pytest
├── .env.example
├── .gitignore
├── Makefile            # receitas Unix/WSL/Git Bash
├── tasks.ps1           # receitas PowerShell/Windows
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requisitos

Python 3.12 (recomendado) ou 3.13. As dependências de tempo de execução são gerenciadas via `pyproject.toml`; um `requirements.txt` espelho é fornecido para conveniência.

## Como reproduzir

Todos os comandos abaixo assumem PowerShell no Windows. Para WSL, Linux ou macOS, use os alvos equivalentes do `Makefile`.

### Setup do ambiente

```powershell
cd projeto_01_espaco_vetorial_recomendador_municipios_agro

# Cria o venv e instala tudo
.\tasks.ps1 setup

# Ativa o ambiente
.\.venv\Scripts\Activate.ps1

# Copia o arquivo de configuração de exemplo
Copy-Item .env.example .env
```

### Execução dos testes

```powershell
.\tasks.ps1 test
```

A suíte pytest cobre progressivamente as fases de desenvolvimento. Na Fase 1.A, apenas o smoke test do módulo `config` está presente, validando que o ambiente está sadio, os paths são detectados corretamente e as constantes estão bem definidas.

### Execução do pipeline

O pipeline completo será implementado nas fases subsequentes. Cada fase adicionará um alvo específico ao `tasks.ps1`. Documentação atualizada conforme cada fase for concluída.

### Abrir os notebooks

```powershell
.\tasks.ps1 notebook
```

Sobe o Jupyter Lab apontado para a pasta `notebooks/`.

## Fases de desenvolvimento

O projeto está sendo construído em fases progressivas para permitir validação incremental:

| Fase | Escopo | Status |
|------|--------|--------|
| 1.A | Estrutura, config, testes de sanidade | Concluída |
| 1.B | Módulo `dataset.py` — download da API SIDRA e da API Localidades | Concluída |
| 1.C | Módulo `features.py` — feature engineering e montagem das tags | Concluída |
| 1.D | Módulos `vectorize.py` e `similarity.py` | Concluída |
| 1.E | Módulo `recommender.py` + notebook demonstrativo end-to-end | Concluída |
| 1.F | Apostila didática completa (`docs/apostila/`) | Concluída |
| 1.G | Extensão de mestrado — validação espacial (Moran's I) | Opcional |

### Novidades da Fase 1.F

Cinco sessões didáticas em `docs/apostila/` cobrem o núcleo conceitual do projeto (Sessões 01 a 05), mais uma sessão bônus (Sessão 06) que apresenta o roadmap para a extensão espacial de nível mestrado (Fase 1.G). Todas escritas no estilo "como se ensinasse" do repositório [estudos-observabilidade](https://github.com/PedroLuizskt/estudos-observabilidade): prosa didática entre trechos de código, diagramas Mermaid nos pontos-chave, exemplos empíricos com dados reais do dataset (Cambuquira/MG, Uberlândia/MG, Guaxupé/MG), fórmulas matemáticas em LaTeX, referências acadêmicas ao final. Sem emojis, tom acadêmico mas acessível.

A **Sessão 01** firma o conceito de espaço vetorial $\mathbb{R}^v$ e explica por que representar itens como vetores viabiliza comparação matemática. A **Sessão 02** percorre o pipeline de feature engineering em cinco estágios (`clean_ppm` → `pivot_ppm_wide` → `merge_com_localidades` → `derive_*` → `build_tags`) e mostra como as tags são o análogo agropecuário dos campos textuais do projeto DSA original. A **Sessão 03** explica o modelo bag-of-words, o `CountVectorizer` do scikit-learn, o `RSLPStemmer` do NLTK para português, e a decisão de projeto de aplicar stemming seletivamente apenas aos tokens simples. A **Sessão 04** cobre as três métricas de comparação vetorial — cosseno, euclidiana e Manhattan — com suas implementações manuais didáticas e vetorizadas via sklearn. A **Sessão 05** amarra tudo mostrando como a classe `MunicipioRecommender` encapsula o pipeline em uma API amigável e como a função `explain` decompõe qualquer recomendação em fatores interpretáveis, usando como exemplo o par Uberlândia/MG × Uberaba/MG que compartilha 19 tokens de 20 possíveis.

A **Sessão 06** é o roadmap opcional para nível mestrado: apresenta o índice de Moran (autocorrelação espacial global) como método para validar cientificamente se a similaridade agropecuária tem estrutura espacial coerente. Descreve as bibliotecas necessárias (geopandas, libpysal, esda), esboça o código, elenca hipóteses testáveis (municípios com "especializado_em_avicultura" formam clusters no Sul de Minas e Norte-Noroeste de SP?) e cuidados metodológicos (escolha da matriz de pesos, MAUP, correção para múltiplos testes). Execução na Fase 1.G quando as dependências geoespaciais forem viáveis no ambiente Windows.

Para navegar a apostila, comece pelo [índice](docs/apostila/00_indice.md).

### Novidades da Fase 1.E

Duas entregas fecham o pipeline funcional do projeto: o módulo `recommender.py` que empacota tudo em uma API amigável, e um notebook Jupyter demonstrativo que percorre o pipeline de ponta a ponta.

O `recommender.py` expõe a classe `MunicipioRecommender` com quatro métodos de consulta e um de explicação. O `recommend_by_name(nome, uf, k, excluir_mesmo_uf)` é o mais usado — aceita nome do município e opcionalmente UF para desambiguar homônimos, retornando os `k` municípios mais similares por similaridade cosseno. O `recommend_by_code(codigo, k)` faz a mesma coisa a partir do código IBGE de 7 dígitos, útil para integração com outros sistemas. O `recommend_by_tags(tags, k)` permite queries hipotéticas: você monta manualmente uma string como `"sudeste alta_bovinocultura alta_avicultura"` e o sistema devolve os municípios que mais se aproximam desse perfil. O `search(parcial)` faz busca fuzzy por nome parcial, útil quando o usuário não sabe grafia exata. E o `explain(query, recomendado)` decompõe uma recomendação em tokens compartilhados e distintos, mostrando as três métricas de distância (cosseno, euclidiana, manhattan) lado a lado — resposta à pergunta *"por que este município foi recomendado?"*.

Uso via CLI:

```powershell
# Consulta simples por nome
.\tasks.ps1 recommend Cambuquira MG

# Ou direto com argumentos completos
python -m rec_agro_br.recommender Cambuquira --uf MG --k 5

# Busca análogos em outras UFs (excluindo MG)
python -m rec_agro_br.recommender Cambuquira --uf MG --excluir-mesmo-uf

# Com explicação da top-1
python -m rec_agro_br.recommender Cambuquira --uf MG --explicar

# Por código IBGE
python -m rec_agro_br.recommender --code 3111606

# Por tags customizadas
python -m rec_agro_br.recommender --tags "sudeste alta_bovinocultura alta_avicultura"

# Busca fuzzy por nome parcial
python -m rec_agro_br.recommender --search "Cambu"
```

O notebook `notebooks/01_pipeline_end_to_end.ipynb` demonstra o projeto inteiro em 17 células de código e 18 de markdown, cobrindo: visão geral do dataset, distribuição de especializações agropecuárias, exemplos de tags construídas, funcionamento do `CountVectorizer` com stemming português, top 20 tokens mais frequentes, cálculo manual passo a passo da similaridade cosseno (para exercitar o conceito matemático do módulo Cap08), consulta ao recomendador com múltiplos casos (Cambuquira/MG, Uberlândia/MG, análogos interestaduais), consulta por tags customizadas, explicação de recomendação, e um heatmap de similaridade entre 12 municípios de referência. Para abrir:

```powershell
.\tasks.ps1 notebook
```

### Novidades da Fase 1.D

Dois novos módulos que implementam o coração matemático do sistema de recomendação content-based.

O `vectorize.py` transforma o corpus de tags agropecuárias em uma matriz esparsa de features usando o `CountVectorizer` do scikit-learn. Ele traz uma adaptação linguística importante: usa o `RSLPStemmer` do NLTK (stemmer específico para português brasileiro) em vez do `PorterStemmer` (inglês) usado no projeto DSA original. O tokenizer é seletivo — stemmiza apenas tokens simples (`bovinocultura` → `bovinocultur`) e preserva intactos os tokens compostos por underscore (`sul_sudoeste_de_minas`, `especializado_em_bovinocultura`), evitando que identificadores geográficos ou categóricos semanticamente unitários sejam fragmentados. O vectorizer e a matriz esparsa são serializados em `data/processed/` para consumo pelas fases seguintes.

O `similarity.py` fornece as três métricas fundamentais de comparação vetorial exercitadas no módulo Cap08: similaridade cosseno, distância euclidiana e distância Manhattan. Cada métrica tem duas implementações: uma vetorizada em lote via scikit-learn (para operações matriz × matriz, usada em produção) e outra manual didática em NumPy puro (para exercitar a fórmula matemática passo a passo, com docstrings que explicam a matemática por trás). Um par de funções `top_k_similares` e `top_k_mais_proximos` recupera os `k` vizinhos mais próximos considerando ordenação decrescente (similaridade) ou crescente (distância), com suporte a exclusão de índices para evitar que o município consultado apareça em sua própria lista de recomendações.

Para gerar o vetorizador e a matriz de features:

```powershell
.\tasks.ps1 vectorize
```

Isso lê `data/processed/municipios_features.parquet`, ajusta o `CountVectorizer` (com stemming por default) e persiste `count_vectorizer.joblib` + `tags_matrix.npz` em `data/processed/`. Também imprime um resumo com os 20 tokens mais frequentes do corpus, o que é útil para debug e para a apostila didática.

### Novidades da Fase 1.C

O módulo `features.py` implementa o pipeline completo de feature engineering, transformando os artefatos brutos do IBGE em um dataset final pronto para vetorização. Cinco estágios organizados como funções puras (recebem DataFrame, devolvem DataFrame), permitindo composição, teste isolado e reuso:

- `clean_ppm` renomeia colunas do formato criptografado do `sidrapy` (D1C, D2N, ...) para nomes legíveis, tipa `valor` corretamente (o hífen `-` do IBGE vira zero por convenção), normaliza espaços em nomes e descarta subcategorias redundantes (matrizes de suínos, galinhas separadas).
- `pivot_ppm_wide` transforma o formato long em wide, uma linha por município e uma coluna por atividade.
- `merge_com_localidades` faz left join preservando todos os 5571 municípios brasileiros; municípios que a PPM não reporta recebem zero nas atividades.
- `derive_perfis_agropecuarios`, `derive_especializacao`, `derive_diversidade` derivam features categóricas que fazem o papel dos `genres`, `crew` e `cast` do projeto DSA original: cada município ganha um perfil quantitativo por atividade (sem/baixa/media/alta), uma atividade dominante, e uma lista de atividades presentes.
- `build_tags` concatena tudo em uma única string por município, normalizada para snake_case, pronta para o `CountVectorizer` da Fase 1.D.

Para executar o pipeline:

```powershell
.\tasks.ps1 build-features
```

Isso lê `data/raw/ppm_3939_efetivo_rebanhos_last_1.parquet` e `data/interim/municipios_localidades.parquet` e gera `data/processed/municipios_features.parquet` com aproximadamente 22 colunas por município.

## Aspectos pedagógicos

O projeto exercita, de forma coordenada, cinco competências centrais do módulo de Matemática e Estatística Aplicada da pós-graduação em Ciência de Dados: representação de itens de qualquer natureza como vetores em espaços de alta dimensão; vetorização de features textuais mistas com Bag-of-Words; redução de dimensionalidade linguística por stemming; cálculo de similaridade em espaços vetoriais por métricas de cosseno e distância euclidiana; e construção de um sistema de recomendação content-based que produz saídas úteis a partir dessas primitivas.

Ao mesmo tempo, o projeto oferece contexto real de aplicação: os municípios brasileiros são o tecido econômico e territorial do país, e um sistema que identifique afinidades agropecuárias entre eles tem utilidade genuína em consultoria agrícola, análise setorial e política pública. A pergunta *"esta técnica que aprendi na pós-graduação resolve um problema que importa?"* tem, aqui, uma resposta operacional.

## Referências

Data Science Academy. *Curso de Pós-Graduação em Ciência de Dados*. Disponível em: [datascienceacademy.com.br](https://www.datascienceacademy.com.br).

IBGE. *Pesquisa da Pecuária Municipal*. Disponível em: [ibge.gov.br](https://www.ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/2041-np-producao-da-pecuaria-municipal.html).

IBGE. *API de Dados Agregados (SIDRA), versão 3*. Documentação: [servicodados.ibge.gov.br/api/docs/agregados?versao=3](https://servicodados.ibge.gov.br/api/docs/agregados?versao=3).

Taranti, A. *sidrapy: A library that provides a python interface for the IBGE SIDRA API*. Disponível em: [github.com/AlanTaranti/sidrapy](https://github.com/AlanTaranti/sidrapy).

Salton, G.; McGill, M. J. *Introduction to Modern Information Retrieval*. McGraw-Hill, 1983. (Referência clássica para o modelo de espaço vetorial e a similaridade cosseno em recuperação de informação.)

Bird, S.; Klein, E.; Loper, E. *Natural Language Processing with Python*. O'Reilly, 2009. (Referência para o uso de NLTK, incluindo o RSLPStemmer para português.)

## Licença

Este projeto é distribuído sob a licença MIT — ver arquivo `LICENSE` na raiz do monorepo. Os dados do IBGE são de domínio público, com uso permitido mediante citação da fonte.
