# Sessão 03 — Vetorização de Texto: Bag-of-Words, Stemming e o RSLP

> **Objetivo desta sessão.** Entender como uma string textual como *"sudeste mg sul_sudoeste_de_minas alta_bovinocultura"* vira um vetor esparso em $\mathbb{R}^v$. Cobre o modelo bag-of-words, a implementação via `CountVectorizer` do scikit-learn, o stemming (com o `RSLPStemmer` do NLTK) e a decisão de projeto de aplicar stemming *seletivamente* apenas aos tokens simples, preservando compostos com underscore.

## O que estamos tentando fazer

Na Sessão 02 chegamos ao dataset processado com uma coluna `tags` por município: uma string curta que sintetiza contexto territorial, perfis quantitativos, especialização e atividades presentes. Precisamos agora transformar cada uma dessas strings em um vetor numérico, todos vivendo no mesmo espaço, para que as comparações da Sessão 04 sejam possíveis.

Essa transformação — texto para vetor numérico — é o problema central de *representação* na área de processamento de linguagem natural. Existem esquemas modernos complexos (word2vec, sentence-transformers, BERT), mas o esquema clássico chamado *bag-of-words* é o adotado pelo projeto DSA original e, para nosso corpus estruturado, é mais que suficiente. Explica-se em uma frase: contamos quantas vezes cada palavra do vocabulário aparece em cada documento.

## O modelo bag-of-words

Suponha um corpus com três documentos:

```
d1: "sudeste mg bovinocultura"
d2: "sudeste mg avicultura"
d3: "sul rs suinocultura"
```

O vocabulário do corpus (união de todas as palavras distintas) tem seis termos:

```
vocab = [avicultura, bovinocultura, mg, rs, sudeste, suinocultura, sul]
```

O modelo bag-of-words representa cada documento como um vetor com uma componente por termo do vocabulário; a componente é a quantidade de vezes que aquele termo aparece no documento. Para nosso mini-corpus:

|    | avicultura | bovinocultura | mg | rs | sudeste | suinocultura | sul |
|----|-----------:|--------------:|---:|---:|--------:|-------------:|----:|
| d1 |          0 |             1 |  1 |  0 |       1 |            0 |   0 |
| d2 |          1 |             0 |  1 |  0 |       1 |            0 |   0 |
| d3 |          0 |             0 |  0 |  1 |       0 |            1 |   1 |

Três decisões implícitas neste modelo merecem destaque. Primeira: a ordem das palavras é descartada — daí o nome "bag" (saco), sem estrutura. Segunda: cada palavra é um átomo indivisível — "mg" é uma coisa, "minas gerais" seriam duas coisas distintas. Terceira: palavras diferentes são dimensões distintas independentemente do seu significado — o modelo não sabe que "bovinocultura" e "bovinos" estão relacionados.

A terceira limitação é o que o *stemming* tenta mitigar, e discutiremos em detalhe adiante.

## CountVectorizer: o executor no scikit-learn

O `CountVectorizer` do `sklearn.feature_extraction.text` implementa bag-of-words de forma vetorizada e otimizada. O uso típico:

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["sudeste mg bovinocultura", "sudeste mg avicultura", "sul rs suinocultura"]
vec = CountVectorizer()
X = vec.fit_transform(corpus)
```

Duas chamadas fazem tudo. O `fit_transform` percorre o corpus construindo o vocabulário (armazenado em `vec.vocabulary_`, um dicionário `{termo: índice}`) e simultaneamente produz a matriz esparsa `X` com uma linha por documento. Para os três documentos acima, `X.shape` seria `(3, 7)`.

Por que matriz esparsa? Porque documentos reais tocam apenas uma pequena fração do vocabulário. Um município do nosso dataset tem cerca de 17 tokens nas suas tags, mas o vocabulário tem 215 tokens. Armazenar 215 valores por linha quando 198 deles são zero é desperdício de memória; o formato esparso (`scipy.sparse.csr_matrix`) armazena só os valores não-nulos com seus índices. Para 5571 municípios, a matriz densa ocuparia $5571 \times 215 \times 4$ bytes $\approx 4{,}8$ MB — pequeno neste caso, mas em corpora textuais reais o ganho pode ser de várias ordens de magnitude.

## Nosso módulo `vectorize`

O módulo `rec_agro_br.vectorize` encapsula o `CountVectorizer` com uma configuração específica ao projeto:

```python
def build_vectorizer(use_stemming=True, max_features=None, min_df=1, max_df=1.0):
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
        lowercase=False,       # o tokenizer já faz lower
        token_pattern=None,    # desabilitado quando tokenizer é custom
    )
```

Três detalhes técnicos merecem explicação. O parâmetro `max_features=None` significa "sem limite de vocabulário", o que é seguro no nosso caso (215 tokens são poucos). O `lowercase=False` desabilita a normalização default do CountVectorizer porque nosso tokenizer já cuida disso. E o `importlib.import_module` do path canônico existe para resolver um bug de identidade de função que aparece quando este módulo é executado como script — o comentário no código-fonte tem os detalhes se você tiver curiosidade.

## Stemming: reduzir palavras à raiz

Um problema clássico do bag-of-words puro é que palavras morfologicamente relacionadas viram dimensões diferentes. "Correr", "corrida", "corredor" seriam três tokens distintos, apesar de compartilharem o radical *corr*. Isso infla o vocabulário e enfraquece a similaridade entre documentos que falam sobre o mesmo tópico com palavras diferentes.

*Stemming* é o processo de reduzir cada palavra ao seu radical, removendo sufixos. Depois do stemming, "correr", "corrida" e "corredor" viram todas o mesmo token (aproximadamente "corr"), e passam a ocupar a mesma dimensão do espaço vetorial.

O projeto DSA original usa o `PorterStemmer` do NLTK, que é um stemmer clássico para *inglês*. Aplica-se aí porque o corpus são metadados de filmes em inglês. Para nosso corpus em português, esse stemmer é inadequado — ele foi projetado para os padrões morfológicos do inglês. Precisamos de um stemmer específico para português.

## RSLPStemmer: o stemmer brasileiro

O `RSLPStemmer` (Removedor de Sufixos da Língua Portuguesa) do NLTK é um dos poucos algoritmos de stemming projetados especificamente para português brasileiro. Cobre corretamente os padrões morfológicos comuns da língua — plurais, sufixos verbais, aumentativos e diminutivos.

Nosso uso é padrão do NLTK:

```python
from nltk.stem import RSLPStemmer
stemmer = RSLPStemmer()
stemmer.stem("bovinocultura")    # → "bovinocultur"
stemmer.stem("avicultura")       # → "avicultur"
stemmer.stem("caprinos")         # → "caprin"
```

O único cuidado adicional é que o `RSLPStemmer` requer arquivos de regras que o NLTK baixa sob demanda. A primeira vez que instanciamos o stemmer, isso é feito automaticamente:

```python
def _ensure_rslp_downloaded():
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        logger.info("[NLTK] Baixando dados do RSLPStemmer (uma vez só)...")
        nltk.download("rslp", quiet=True)
```

## A decisão fina: stemming *seletivo*

Aqui está uma decisão de projeto que ilustra bem o compromisso entre "usar as ferramentas padrão" e "adaptar as ferramentas ao problema". Nosso vocabulário mistura dois tipos de tokens:

Primeiro tipo: **palavras simples** como `bovinocultura`, `avicultura`, `nordeste`, `mg`. Essas se beneficiam do stemming — se o RSLP reduz "bovinocultura" e "bovinocultores" (se ambos aparecessem no corpus) ao mesmo radical, ganhamos coesão.

Segundo tipo: **tokens compostos por underscore** como `sul_sudoeste_de_minas`, `metropolitana_de_são_paulo`, `especializado_em_bovinocultura`. Esses são identificadores semanticamente unitários. Se stemmizássemos, o RSLP poderia fragmentá-los ou introduzir ruído em identificadores geográficos que precisam ser tratados como átomos.

A solução foi um tokenizer híbrido que aplica stemming *apenas* aos tokens simples e preserva os compostos intactos:

```python
def tokenize_com_stemming(texto):
    if texto is None or not isinstance(texto, str):
        return []
    stemmer = _get_stemmer()
    tokens = texto.lower().split()
    resultado = []
    for tok in tokens:
        if "_" in tok:
            resultado.append(tok)          # preservado intacto
        else:
            try:
                resultado.append(stemmer.stem(tok))
            except (IndexError, ValueError):
                resultado.append(tok)      # fallback: sem stemming
    return resultado
```

Simples e didático. A regra `if "_" in tok` é a linha que define nossa estratégia. Um teste automatizado (`tests/test_vectorize.py::TestTokenizeComStemming`) garante que essa regra continue funcionando conforme o projeto evolua.

## O que aconteceu na prática com o nosso corpus

Quando você rodou `python -m rec_agro_br.vectorize` no seu ambiente, o output foi:

```
Documentos:                 5571
Dimensão do vocabulário:    215
Densidade da matriz:        8.60%
```

Interpretemos. Cinco mil e quinhentos e setenta e um documentos (municípios). Duzentos e quinze tokens únicos no vocabulário após stemming. Densidade de 8,6%, ou seja, cerca de 8,6% das células da matriz $5571 \times 215$ são não-nulas — o equivalente a uns 18-19 tokens não-nulos por linha em média, consistente com a estrutura do nosso campo `tags` que tem 17 tokens em média por município.

O tamanho do vocabulário (215) confirma o que a estimativa dava: 5 regiões + 27 UFs + ~137 mesorregiões + 32 perfis (8 atividades × 4 níveis) + 9 especializações + 8 atividades presentes ≈ 218 tokens. O stemming pouco reduz esse número porque a maioria dos tokens ou já é composto (preservado pelo tokenizer híbrido) ou já é raiz curta.

## Um passo além: TF-IDF (opcional)

O `CountVectorizer` conta ocorrências brutas. Um esquema mais sofisticado, o TF-IDF (*Term Frequency – Inverse Document Frequency*), pondera cada contagem pela raridade do termo no corpus: termos que aparecem em muitos documentos ganham peso baixo (são pouco discriminativos), termos raros ganham peso alto. Para nosso caso, `sudeste` aparece em ~2100 documentos (todos os municípios do Sudeste) — sob TF-IDF, esse token ganharia peso baixo, e a similaridade entre dois municípios do Sudeste seria menos "empurrada" pelo compartilhamento dessa palavra comum.

O projeto DSA original usa `CountVectorizer` puro; mantemos essa escolha por fidelidade pedagógica. Uma extensão natural (e boa candidata para a Fase 1.G) seria comparar empiricamente os dois esquemas — se o TF-IDF produz recomendações mais discriminativas que o `CountVectorizer` para nosso corpus. A infraestrutura já suporta isso; basta trocar a classe.

## Serialização: por que joblib e o bug do pickle

O vetorizador ajustado (com o vocabulário construído e o tokenizer configurado) é salvo em disco como `count_vectorizer.joblib` para reutilização pelo recomendador. Usamos `joblib` porque é o padrão de fato do ecossistema scikit-learn para serialização de estimators.

Um detalhe importante que descobrimos na prática: quando o CountVectorizer usa um tokenizer customizado (função Python), o pickle guarda uma referência simbólica à função (nome do módulo + nome da função), não o código da função. Se o vetorizador é criado em um contexto onde o módulo virou `__main__` (ex: `python -m rec_agro_br.vectorize`) e depois carregado em outro contexto (ex: `python -m rec_agro_br.recommender`), o unpickle não encontra a função e falha. A correção usa `importlib.import_module` no `build_vectorizer` para garantir que a função referenciada seja sempre a canônica do pacote, protegida por três testes de regressão em `tests/test_vectorize.py::TestPickleRobustezModuloCanonico`.

Sem entrar em mais detalhes técnicos aqui — a informação está no código com comentários extensos —, o fato de que tropeçamos nesse bug é em si informativo: pickle é uma ferramenta poderosa mas cheia de armadilhas, e testes automatizados são a única linha real de defesa contra regressões nesse tipo de código.

## Recapitulando

Bag-of-words transforma cada documento textual em vetor esparso onde cada dimensão corresponde a um token único do vocabulário. O `CountVectorizer` do scikit-learn implementa isso de forma vetorizada. O stemming reduz variações morfológicas ao radical, aumentando a coesão semântica das dimensões — usamos o `RSLPStemmer` do NLTK adaptado ao português brasileiro. Nosso tokenizer híbrido aplica stemming apenas a tokens simples, preservando compostos com underscore intactos. Para nosso corpus estruturado, isso resulta em vetores em $\mathbb{R}^{215}$ com ~8% de densidade — matéria-prima ideal para o cálculo de similaridade da próxima sessão.

## Próxima sessão

Na Sessão 04 mergulharemos nas métricas de similaridade e distância entre vetores. Veremos por que a cosseno é preferida para bag-of-words, o que diferencia euclidiana e manhattan, e como implementar cada uma tanto em versão didática (NumPy puro) quanto vetorizada (scikit-learn).

## Referências

Manning, C. D.; Raghavan, P.; Schütze, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008. Disponível em [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book). Capítulo 6 cobre bag-of-words, TF-IDF e o modelo de espaço vetorial em profundidade.

Bird, S.; Klein, E.; Loper, E. *Natural Language Processing with Python*. O'Reilly, 2009. Referência para o NLTK, incluindo o capítulo sobre stemming e o uso do RSLPStemmer para português.

Orengo, V. M.; Huyck, C. *A Stemming Algorithm for the Portuguese Language*. Symposium on String Processing and Information Retrieval (SPIRE), 2001. O paper original que descreve o algoritmo RSLP.

scikit-learn Documentation. *Feature Extraction — Text feature extraction*. [scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). Referência canônica do CountVectorizer.
