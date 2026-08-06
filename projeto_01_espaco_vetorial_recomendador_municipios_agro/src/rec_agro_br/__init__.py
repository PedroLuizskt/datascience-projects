"""rec_agro_br — Recomendador de Municípios Brasileiros por Perfil Agropecuário.

Pacote de estudo desenvolvido como adaptação do projeto Cap08 da pós-graduação
em Ciência de Dados da Data Science Academy (DSA). O projeto original ensina
conceitos de espaço vetorial aplicados a sistemas de recomendação; esta
adaptação os aplica ao domínio agropecuário brasileiro, usando dados abertos
do IBGE (PPM/SIDRA).

Módulos
-------
config
    Constantes, paths e parâmetros configuráveis do pipeline.

Notas
-----
Autor: Pedro Luiz Rocha Vaz de Melo
Licença: MIT
Repositório: https://github.com/PedroLuizskt/datascience-projects
"""

from __future__ import annotations

from rec_agro_br.config import PROJECT_VERSION as __version__

__all__ = ["__version__"]
