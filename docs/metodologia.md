# Metodologia de Adaptação dos Projetos DSA

Este documento descreve o framework aplicado à reengenharia de cada projeto da pós-graduação em Ciência de Dados da Data Science Academy para composição do portfólio `datascience-projects`. É um documento vivo, atualizado quando novos aprendizados justificarem refinamentos.

## Princípio orientador

A meta central é transformar um projeto didático de domínio genérico em um projeto autoral que resolva um problema real na área de atuação do autor — dados espaciais, agronômicos, geográficos, demográficos e agropastoris — sem descaracterizar o núcleo pedagógico que a DSA se propôs a ensinar. Um projeto adaptado com sucesso demonstra simultaneamente três coisas ao leitor: primeiro, que o autor absorveu a técnica que a pós ensinou; segundo, que o autor sabe transferir essa técnica para um domínio real; terceiro, que o autor tem visão para estender a técnica além do que foi ensinado quando pertinente.

## Etapas de adaptação

### Etapa 1 — Diagnóstico

O projeto DSA original é lido integralmente e três camadas são explicitadas. A camada **essencial** contém o conceito matemático ou algorítmico que a disciplina ensina, e é preservada sem exceção. A camada **estrutural** contém o fluxo de trabalho de ciência de dados — carregamento, limpeza, feature engineering, modelagem, avaliação — e é preservada com adaptações mínimas necessárias ao novo domínio. A camada **de domínio** contém o dataset original e a narrativa da aplicação, e é integralmente substituída.

O produto desta etapa é uma breve nota escrita no README do projeto, seção "Referência ao projeto original", explicitando o que veio da DSA e o que foi autoral.

### Etapa 2 — Redomínio

Escolhido o vetor de domínio (espacial, agronômico, geográfico, demográfico ou agropastoril), busca-se um dataset público que atenda três critérios simultâneos. Primeiro, escala compatível com o dataset original — se o original tinha milhares de itens, o substituto deve ter ordem de magnitude semelhante para preservar a experiência computacional. Segundo, tipos de features compatíveis com o que a disciplina exercita — se o original exercitava vetorização de texto, o substituto deve ter matéria-prima textual ou categórica rica. Terceiro, fonte referenciável e licença permissiva — o dataset deve poder ser baixado programaticamente e ter licença que permita redistribuição analítica.

As fontes prioritárias, em ordem de conveniência, são: APIs governamentais brasileiras (IBGE SIDRA, TerraBrasilis, MapBiomas, ICMBio, MAPA), datasets científicos públicos (GBIF, CHELSA, WorldClim, Flora do Brasil, NOAA), datasets embarcados em pacotes Python (scikit-learn, seaborn, statsmodels, pydataset, NLTK, Hugging Face) e agregadores tabulares (Kaggle, UCI ML Repository) quando as fontes anteriores não atenderem.

O produto desta etapa é a documentação do dataset escolhido no README do projeto, seção "Fonte de dados", com URL, licença, ano de referência e método de aquisição.

### Etapa 3 — Reengenharia

O projeto é reimplementado sob a estrutura Cookiecutter Data Science v2 como pacote Python instalável com nome curto e mnemônico. A estrutura padrão é:

```
projeto_NN_nome_curto/
├── data/{raw,interim,processed,external}/
├── docs/apostila/
├── notebooks/
├── references/
├── reports/figures/
├── src/nome_do_pacote/
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

O código analítico é factorizado em módulos temáticos dentro de `src/`, os notebooks tornam-se demonstrações executáveis end-to-end que importam do pacote, e uma suíte pytest cobre os módulos críticos. O ambiente virtual é criado localmente à pasta do projeto como `.venv/`, isolado do restante do monorepo.

Uma apostila didática em `docs/apostila/` acompanha o projeto, com um arquivo markdown por bloco conceitual, escrita como se ensinasse. O modelo de escrita segue o padrão adotado no repositório `estudos-observabilidade`, com prosa explicativa entre trechos de código e diagramas Mermaid nos pontos-chave.

O produto desta etapa é o projeto executável de ponta a ponta, com testes verdes, notebooks executados e apostila redigida.

### Etapa 4 — Extensão metodológica (opcional)

Quando o projeto sustentar uma contribuição adicional cientificamente interessante, uma seção de extensão é adicionada. Extensões comuns incluem: comparação empírica entre técnicas (por exemplo, CountVectorizer versus TF-IDF versus sentence embeddings), validação estatística mais rigorosa (bootstrap, permutação, correção para múltiplos testes), análise espacial (Moran's I, correlogramas), interpretabilidade de modelos (SHAP, LIME), ou reflexão epistemológica sobre limites da técnica no domínio escolhido.

Cada projeto sinaliza no seu README se contém extensão de nível mestrado ou apenas a adaptação de nível especialização. Ambos os níveis são válidos e complementares no portfólio.

## Convenções operacionais

**Nomenclatura de pastas**. Cada projeto ocupa uma pasta com nome `projeto_NN_descricao_curta_snake_case`, onde `NN` é o número sequencial de dois dígitos e a descrição resume o problema em três a cinco palavras.

**Nomenclatura de pacotes Python**. O pacote dentro de `src/` recebe um nome curto e mnemônico, tipicamente três a seis caracteres, refletindo o domínio do projeto. Exemplos: `rec_agro_br`, `sil_bio`, `agro_cli`.

**Ambientes virtuais**. Cada projeto usa `.venv/` local, criado com `py -3.12 -m venv .venv`. Se um projeto exigir versão diferente de Python, isso é documentado no README daquele projeto e refletido na chamada `py -X.Y`.

**Commits**. Adotam o padrão Conventional Commits. Cada novo projeto entra no monorepo com uma sequência progressiva de commits que conta a história do desenvolvimento, e não com um único commit gigante. Exemplo de sequência típica: `feat(projeto_01): scaffold structure`, `feat(projeto_01): add data loading module`, `feat(projeto_01): implement vectorization`, `test(projeto_01): add unit tests`, `docs(projeto_01): write apostila`.

**Ausência de emojis**. Nenhum emoji em código, comentários, documentação, logs ou commits. Logs de produção usam marcadores textuais `[OK]`, `[AVISO]` e `[ERRO]`. Esta convenção segue a preferência estética do autor por documentação de tom acadêmico.

**Reprodução**. Cada projeto documenta seu procedimento de reprodução no README próprio. O padrão default é criar o venv, ativá-lo, instalar dependências, rodar testes e executar os notebooks. Desvios do padrão são explicitados.

## Fluxo de trabalho conjunto autor-assistente

Cada projeto DSA a ser adaptado segue um ciclo previsível de interação. O autor envia o zip do projeto DSA original. O assistente lê e produz o diagnóstico da Etapa 1, com propostas de vetor de domínio para a Etapa 2. O autor valida o rumo. O assistente executa a Etapa 3 entregando os arquivos completos por partes lógicas, com instruções PowerShell exatas para reprodução no ambiente Windows do autor. O autor reproduz localmente e envia outputs e prints ou logs em caso de falha. As iterações continuam até o projeto atingir o estado alvo, com testes verdes e notebooks executados. Só então o autor commita e passa para o próximo projeto.
