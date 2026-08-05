# datascience-projects

Portfólio aplicado de Ciência de Dados construído a partir das adaptações e extensões dos projetos práticos do curso de pós-graduação em Ciência de Dados da **Data Science Academy (DSA)**. Cada projeto original do curso é redesenhado sobre um domínio próprio do autor — dados espaciais, agronômicos, geográficos, demográficos e agropastoris — preservando o núcleo pedagógico da disciplina e adicionando extensões metodológicas de nível mestrado quando pertinente.

## Autor

**Pedro Luiz R. Vaz de Melo**
Engenheiro Florestal e Cientista de Dados Geoespacial
Cientista de Dados no EnvironBIT — Consultor GIS na Vimef Engenharia — Co-fundador da Bússola dos Dados
Pós-graduando em Ciência de Dados pela Data Science Academy

GitHub: [github.com/PedroLuizskt](https://github.com/PedroLuizskt)

## Motivação e escopo

Os módulos da pós-graduação em Ciência de Dados da DSA oferecem projetos práticos que cobrem um espectro amplo de fundamentos — álgebra linear aplicada, estatística, aprendizado de máquina supervisionado e não supervisionado, aprendizado profundo, séries temporais e processamento de linguagem natural. Cada projeto original é apresentado com um dataset genérico escolhido para fins didáticos. Este repositório executa um trabalho de reengenharia sobre esses projetos, mantendo o mesmo escopo conceitual e substituindo o domínio de aplicação por problemas reais das áreas de atuação do autor. O objetivo é duplo: reforçar o aprendizado a partir da resolução de um problema autêntico e produzir um portfólio verificável que evidencie a especialização adquirida no curso.

Todas as adaptações preservam a autoria conceitual da DSA sobre os projetos originais e explicitam, em cada pasta, qual foi o projeto-fonte, o que foi preservado e o que foi estendido. Nenhum dado ou artefato proprietário da DSA é redistribuído; os projetos aqui presentes utilizam datasets públicos com fontes referenciadas.

## Estrutura do repositório

```
datascience-projects/
├── docs/
│   └── metodologia.md
├── projeto_01_espaco_vetorial_recomendador_municipios_agro/
├── projeto_02_.../
├── projeto_NN_.../
├── .gitignore
├── LICENSE
└── README.md
```

Cada pasta `projeto_NN_*` é um projeto autocontido com seu próprio ambiente virtual, dependências, pipeline de dados, notebooks, testes, apostila didática e README específico. A estratégia de isolamento por projeto garante que dependências conflitantes entre módulos da pós não se contaminem mutuamente e que cada projeto seja individualmente reprodutível.

## Índice de projetos

| # | Projeto | Módulo DSA | Domínio | Status |
|---|---------|------------|---------|--------|
| 01 | Recomendador de Municípios Brasileiros por Perfil Agropecuário | Matemática e Estatística Aplicada Para Data Science, Machine Learning e IA | Agropastoril e demográfico | Em desenvolvimento |

Novos projetos são adicionados a esta tabela conforme concluídos. Cada linha aponta para a pasta local com o README específico do projeto.

## Metodologia de adaptação

O framework aplicado a cada projeto está documentado em `docs/metodologia.md` e resumido a seguir. Um projeto DSA passa por quatro etapas de reengenharia. Primeiro, diagnóstico: identificação do núcleo conceitual do projeto original, do domínio substituível e das técnicas específicas ensinadas. Segundo, redomínio: escolha do problema de aplicação real e do dataset público adequado à escala e à natureza das features do original. Terceiro, reengenharia: reimplementação sob o padrão Cookiecutter Data Science v2 como pacote Python instalável, com testes automatizados e notebooks executáveis end-to-end. Quarto, extensão metodológica opcional para projetos que sustentem uma contribuição adicional cientificamente interessante — análise espacial, comparação de modelos, validação estatística mais rigorosa.

## Convenções

Este repositório adota Conventional Commits para o histórico do Git. Cada projeto novo é introduzido com uma sequência de commits progressivos, permitindo ao leitor acompanhar o processo de construção. Todo código é escrito em Python 3.12 salvo exceção documentada. Todos os projetos seguem a estrutura Cookiecutter Data Science v2 adaptada. Nenhum projeto contém emojis em código, documentação ou logs; marcadores de log em produção usam os padrões `[OK]`, `[AVISO]` e `[ERRO]`.

## Reprodução

O procedimento de reprodução é local a cada projeto. Cada pasta `projeto_NN_*` contém um `README.md` com instruções específicas de criação do ambiente virtual, instalação de dependências, execução dos notebooks e testes. O padrão geral é:

```powershell
cd projeto_NN_nome
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest tests/
jupyter notebook
```

## Referências

Data Science Academy. *Curso de Pós-Graduação em Ciência de Dados*. Disponível em: [datascienceacademy.com.br](https://www.datascienceacademy.com.br). Os projetos originais aqui adaptados são de autoria da DSA e utilizados no âmbito acadêmico como base para exercícios de reengenharia com fins de aprendizagem e composição de portfólio.

Referências científicas específicas a cada projeto (metodologias, papers, datasets públicos) constam nos READMEs individuais de cada pasta.

## Licença

O código autoral deste repositório é distribuído sob a licença MIT — ver arquivo `LICENSE`. Datasets públicos utilizados mantêm suas licenças originais, referenciadas nos respectivos projetos.
