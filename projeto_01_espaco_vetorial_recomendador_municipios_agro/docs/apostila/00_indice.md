# Apostila — Espaço Vetorial e Sistemas de Recomendação

Apostila didática que acompanha o projeto `rec-agro-br`. Escrita "como se ensinasse", cada arquivo cobre um bloco conceitual do projeto, explicando não só *o quê* o código faz, mas *por quê* e *como* a matemática do módulo da pós-graduação se manifesta no domínio agropecuário brasileiro.

O estilo segue o mesmo padrão adotado no repositório [`estudos-observabilidade`](https://github.com/PedroLuizskt/estudos-observabilidade): prosa didática entre trechos de código, diagramas Mermaid nos pontos-chave, e progressão do conceito para a implementação.

## Índice

| Sessão | Título | Status |
|--------|--------|--------|
| [01](01_conceito_espaco_vetorial.md) | Conceito de Espaço Vetorial | Concluída |
| [02](02_do_dominio_ao_vetor.md) | Do Domínio ao Vetor: Municípios como Pontos no R^n | Concluída |
| [03](03_vetorizacao_de_texto.md) | Vetorização de Texto: Bag-of-Words, Stemming e o RSLP | Concluída |
| [04](04_distancias_e_similaridade.md) | Distâncias e Similaridade: Cosseno, Euclidiana, Manhattan | Concluída |
| [05](05_construindo_o_recomendador.md) | Construindo o Recomendador Content-Based | Concluída |
| [06](06_extensao_validacao_espacial.md) | Extensão de Mestrado: Validação Espacial com Moran's I | Roadmap (execução na Fase 1.G, opcional) |

## Como ler

A apostila foi pensada para ser lida sequencialmente por quem quer entender o projeto do zero. Cada sessão parte da última e assume que você já leu as anteriores. Referências cruzadas entre sessões usam o formato `Ver Sessão XX`.

Se você já domina os conceitos e só quer navegar rapidamente pela implementação, o notebook `notebooks/01_pipeline_end_to_end.ipynb` é um caminho mais direto — percorre o mesmo material em código executável.

## Progressão conceitual

As cinco primeiras sessões formam uma progressão linear que cobre o núcleo do projeto:

```
01 (conceito matemático de espaço vetorial)
        │
        ▼
02 (mapeamento domínio agropecuário → tags textuais)
        │
        ▼
03 (tags textuais → vetores esparsos via CountVectorizer + RSLP)
        │
        ▼
04 (métricas de comparação entre vetores: cosseno, euclidiana, Manhattan)
        │
        ▼
05 (sistema completo: MunicipioRecommender com múltiplos modos de consulta)
```

A Sessão 06 é um bônus opcional que aponta para além do escopo original: como validar cientificamente as similaridades produzidas pelo sistema usando análise espacial (Moran's I). Depende de bibliotecas geoespaciais pesadas (geopandas, libpysal, esda) e é executada apenas na Fase 1.G, marcada como opcional no roteiro do projeto.

## Sobre o estilo

Cada sessão foi escrita para ser lida em 20-30 minutos. Segue uma estrutura consistente: contexto (por que essa sessão importa), conceito matemático central, aplicação ao domínio, trechos de código real do projeto que implementam o conceito, exemplos empíricos com dados reais (Cambuquira/MG, Uberlândia/MG, Guaxupé/MG), recap e referências para aprofundamento. Sem emojis, sem jargão desnecessário, com fórmulas matemáticas escritas em LaTeX.

O objetivo é servir tanto como material de estudo (para você fixar os conceitos da matéria da pós-DSA) quanto como documentação viva do projeto (para qualquer pessoa que abra o repositório entender as decisões arquiteturais).
