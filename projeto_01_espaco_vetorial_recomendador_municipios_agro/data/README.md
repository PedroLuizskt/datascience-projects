# data/

Estrutura de dados do projeto seguindo o padrão Cookiecutter Data Science v2. Nenhum arquivo de dados é versionado no Git — todos são regeneráveis a partir do pipeline. As subpastas aqui existem para fixar a estrutura via `.gitkeep`.

## Subpastas

**raw/** contém os artefatos brutos baixados das fontes originais, no formato em que a fonte os fornece. Para este projeto, são os JSONs retornados pela API SIDRA/IBGE (tabela 3939, PPM) e a lista de municípios da API Localidades. Nunca editar arquivos aqui à mão. Se o arquivo bruto precisar de correção, corrigir o `dataset.py` que faz a leitura, não o arquivo.

**interim/** contém transformações intermediárias, tipicamente em formato Parquet. São o resultado de operações como pivot, limpeza de tipos e normalização de nomes de município, feitas sobre os dados de `raw/`. Podem ser regenerados a qualquer momento executando o passo correspondente do pipeline.

**processed/** contém o dataset final consumido pelos notebooks e pelo recomendador. É o resultado das operações de feature engineering (montagem das "tags" agropecuárias) feitas sobre `interim/`. Um único arquivo Parquet consolidado com os 5570 municípios e suas features vetorizáveis.

**external/** reservado para dados externos que não vêm da API SIDRA, quando forem necessários. Exemplo futuro: shapefile de biomas do IBGE, dados climáticos do CHELSA. Por enquanto vazio.

## Como popular

Após configurar o ambiente (ver README principal do projeto), o pipeline de download será acionado por comando específico (a ser implementado na Fase 1.B). Documentaremos aqui o passo assim que a função de download estiver escrita.
