# Apostila Didática — Projeto 2

Material de estudo escrito no estilo "como se ensinasse" do repositório [estudos-observabilidade](https://github.com/PedroLuizskt/estudos-observabilidade). Cada sessão parte do conceito matemático central, mostra como se manifesta no código real do projeto, e traz exemplos empíricos do dataset de fraude usado como aplicação.

## Índice

| Sessão | Título | Status |
|--------|--------|--------|
| 01 | Do Neurônio à Regressão Logística: uma rede neural mínima | A escrever (Fase 2.D) |
| 02 | Forward Pass e Backward Pass como Operações com Matrizes | A escrever (Fase 2.D) |
| 03 | Detecção de Fraude e o Problema do Desbalanceamento | A escrever (Fase 2.D) |
| 04 | Avaliação Honesta: Além da Acurácia | A escrever (Fase 2.D) |

## Como ler

A apostila foi pensada para ser lida sequencialmente por quem quer entender o projeto do zero. Cada sessão parte da última.

Se você já domina os conceitos e só quer navegar rapidamente pela implementação, o notebook `notebooks/01_pipeline_end_to_end.ipynb` (Fase 2.C) é um caminho mais direto — percorre o mesmo material em código executável.

## Progressão conceitual

As quatro sessões formam uma progressão linear que cobre o núcleo do projeto:

```
01 (o algoritmo: por que "rede neural" de 1 camada = regressão logística)
        │
        ▼
02 (a matemática: forward pass e backward pass como operações matriciais)
        │
        ▼
03 (a aplicação: detecção de fraude e o problema real do desbalanceamento)
        │
        ▼
04 (a avaliação: por que acurácia mente, e o que usar no lugar)
```

Escopo enxuto por escolha — o Projeto 2 é matematicamente mais compacto que o Projeto 1 (uma classe de 91 linhas versus todo um pipeline de features + vetorização + similaridade). A apostila reflete essa diferença de escala.

## Sobre o estilo

Cada sessão foi escrita para ser lida em 15-20 minutos. Sem emojis, sem jargão desnecessário, com fórmulas matemáticas escritas em LaTeX. Objetivo duplo: material de estudo (fixar os conceitos de operações com matrizes) e documentação viva do projeto.
