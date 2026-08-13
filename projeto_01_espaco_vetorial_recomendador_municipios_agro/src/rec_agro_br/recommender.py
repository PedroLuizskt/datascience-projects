"""Sistema de recomendação de municípios brasileiros por perfil agropecuário.

Este módulo empacota os módulos :mod:`features`, :mod:`vectorize` e
:mod:`similarity` em uma API amigável e coesa para consumo por notebooks,
CLI e (eventualmente) uma aplicação web. É a camada final do pipeline
implementado nas Fases 1.A a 1.D.

Interface principal
-------------------
A classe :class:`MunicipioRecommender` carrega o dataset processado e a
matriz de features vetorizadas em memória (custo único), e expõe métodos
para consulta:

- :meth:`recommend_by_name` — recomenda por nome do município
- :meth:`recommend_by_code` — recomenda por código IBGE (7 dígitos)
- :meth:`recommend_by_tags` — recomenda a partir de uma string de tags
  customizada, útil para queries hipotéticas ("me mostre municípios com
  perfil bovinocultura alta e sem suínos")
- :meth:`explain` — decompõe uma recomendação em features compartilhadas
- :meth:`search` — resolve nome parcial de município (busca fuzzy simples)

Cada método de recomendação devolve uma lista de :class:`RecomendacaoResult`,
que agrupa o município recomendado, a similaridade cosseno e metadados
úteis para exibição.

Análogo com o projeto DSA original
-----------------------------------
O projeto Cap08 da DSA implementa uma função ``recomendar_filmes(titulo, top_n)``
que recebe o título de um filme e devolve os N mais similares. Este módulo
é a versão amadurecida disso: em vez de uma função solta, uma classe com
estado (carregamentos caros feitos uma vez), múltiplos métodos de consulta,
tratamento de erro e uma facilidade adicional (``explain``) que o projeto
original não tem.

Exemplos
--------
Uso programático::

    from rec_agro_br.recommender import MunicipioRecommender
    rec = MunicipioRecommender.load()
    resultados = rec.recommend_by_name("Cambuquira", uf="MG", k=5)
    for r in resultados:
        print(f"{r.similaridade:.4f}  {r.nome} ({r.uf})")

Uso via CLI::

    python -m rec_agro_br.recommender "Cambuquira" --uf MG --k 5
    python -m rec_agro_br.recommender --code 3111606 --k 10
    python -m rec_agro_br.recommender --search "Cambu"
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from rec_agro_br import config, features, similarity, vectorize

logger = logging.getLogger(__name__)


# =============================================================================
# Modelo de resultado
# =============================================================================
@dataclass
class RecomendacaoResult:
    """Resultado individual de uma recomendação.

    Attributes
    ----------
    id_municipio : int
        Código IBGE de 7 dígitos.
    nome : str
        Nome do município.
    uf : str
        Sigla da UF (SP, MG, ...).
    mesorregiao : str
        Nome da mesorregião IBGE.
    regiao : str
        Nome da região (Norte, Nordeste, Sudeste, Sul, Centro-Oeste).
    especializacao : str
        Atividade agropecuária dominante identificada pelo pipeline.
    similaridade : float
        Similaridade cosseno em :math:`[0, 1]` com o município consultado.
    tags : str
        Campo de tags completo, disponível para inspeção via :meth:`explain`.
    """

    id_municipio: int
    nome: str
    uf: str
    mesorregiao: str
    regiao: str
    especializacao: str
    similaridade: float
    tags: str = field(repr=False)


@dataclass
class Explicacao:
    """Decomposição de uma recomendação em fatores explicativos."""

    query_nome: str
    query_uf: str
    recomendado_nome: str
    recomendado_uf: str
    similaridade_cosseno: float
    distancia_euclidiana: float
    distancia_manhattan: float
    tokens_em_comum: list[str]
    tokens_so_query: list[str]
    tokens_so_recomendado: list[str]


# =============================================================================
# Recomendador
# =============================================================================
class MunicipioNaoEncontradoError(ValueError):
    """Levantada quando um município não é encontrado pelo critério dado."""


class NomeAmbiguoError(ValueError):
    """Levantada quando um nome de município tem múltiplos matches."""


class MunicipioRecommender:
    """Sistema de recomendação content-based para municípios brasileiros.

    Carrega em memória o dataset processado (5571 municípios com features
    agropecuárias) e a matriz de features vetorizadas (bag-of-words das
    tags). Consultas são resolvidas em tempo O(n) sobre 5571 municípios,
    tipicamente em milissegundos.

    A instância é construída via método de classe :meth:`load` que faz o
    carregamento canônico dos artefatos em ``data/processed/``. Para uso
    em testes ou notebooks avançados, o construtor direto aceita os
    objetos já carregados.

    Attributes
    ----------
    df : pandas.DataFrame
        Dataset processado de :func:`features.load_features_dataset`.
    vectorizer : CountVectorizer
        Vectorizer ajustado, de :func:`vectorize.load_vectorizer`.
    X : scipy.sparse.csr_matrix
        Matriz de features, de :func:`vectorize.load_matrix`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vectorizer: object,
        X: sparse.csr_matrix,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.X = X
        self._validar_consistencia()

    def _validar_consistencia(self) -> None:
        """Sanity check: dataset e matriz precisam ter o mesmo número de linhas."""
        if len(self.df) != self.X.shape[0]:
            raise ValueError(
                f"Inconsistência: dataset tem {len(self.df)} linhas mas matriz "
                f"tem {self.X.shape[0]}. Refaça build-features e vectorize "
                "para regenerar ambos em sincronia."
            )
        for col in ("id_municipio", "nome_municipio", "sigla_uf", "tags"):
            if col not in self.df.columns:
                raise ValueError(f"Coluna obrigatória ausente no dataset: {col}")

    @classmethod
    def load(cls) -> "MunicipioRecommender":
        """Carrega todos os artefatos do disco e devolve uma instância pronta.

        Raises
        ------
        FileNotFoundError
            Se algum dos três artefatos (dataset processado, vectorizer,
            matriz) não estiver em disco. Rode antes:
            ``python -m rec_agro_br.features`` e
            ``python -m rec_agro_br.vectorize``.
        """
        df = features.load_features_dataset()
        vec = vectorize.load_vectorizer()
        X = vectorize.load_matrix()
        return cls(df=df, vectorizer=vec, X=X)

    # -------------------------------------------------------------------------
    # Resolução de município (nome → índice)
    # -------------------------------------------------------------------------
    def _locate_by_name(self, nome: str, uf: str | None = None) -> int:
        """Resolve nome de município para índice na tabela.

        Faz match case-insensitive. Se ``uf`` for passado, filtra por ela.
        Se houver múltiplos matches sem UF especificada, levanta
        :class:`NomeAmbiguoError` com sugestão de desambiguação.

        Parameters
        ----------
        nome : str
            Nome do município (case-insensitive, whitespace tolerado).
        uf : str, optional
            Sigla da UF para desambiguar homônimos.

        Returns
        -------
        int
            Índice da linha no ``self.df``.

        Raises
        ------
        MunicipioNaoEncontradoError
            Se nenhum município bater com o nome (opcionalmente + UF).
        NomeAmbiguoError
            Se múltiplos municípios baterem e nenhuma UF foi passada.
        """
        nome_norm = nome.strip().casefold()
        mask = self.df["nome_municipio"].str.casefold() == nome_norm
        if uf is not None:
            mask &= self.df["sigla_uf"] == uf.upper()

        matches = self.df[mask]

        if len(matches) == 0:
            sugestoes = self.search(nome, max_results=5)
            msg = f"Nenhum município encontrado com nome '{nome}'"
            if uf:
                msg += f" na UF {uf.upper()}"
            if sugestoes:
                msg += f". Sugestões: {[f'{s[0]} ({s[1]})' for s in sugestoes]}"
            raise MunicipioNaoEncontradoError(msg)

        if len(matches) > 1:
            ufs = matches["sigla_uf"].unique().tolist()
            raise NomeAmbiguoError(
                f"Múltiplos municípios com nome '{nome}': "
                f"presente em {ufs}. Especifique a UF."
            )

        return int(matches.index[0])

    def _locate_by_code(self, code: int | str) -> int:
        """Resolve código IBGE (7 dígitos) para índice na tabela."""
        try:
            code_int = int(code)
        except (ValueError, TypeError) as e:
            raise MunicipioNaoEncontradoError(f"Código inválido: {code}") from e

        matches = self.df[self.df["id_municipio"] == code_int]
        if len(matches) == 0:
            raise MunicipioNaoEncontradoError(
                f"Nenhum município com código IBGE {code_int}. "
                "Códigos válidos têm 7 dígitos (ex: 3111606 = Cambuquira/MG)."
            )
        return int(matches.index[0])

    def search(
        self,
        parcial: str,
        max_results: int = 10,
    ) -> list[tuple[str, str]]:
        """Busca municípios cujo nome contenha ou seja parecido com ``parcial``.

        Primeiro tenta match por substring case-insensitive. Se pouco
        retornar, complementa com fuzzy matching via
        :func:`difflib.get_close_matches`.

        Parameters
        ----------
        parcial : str
            Fragmento ou variação do nome.
        max_results : int
            Limite de resultados.

        Returns
        -------
        list of (nome_municipio, sigla_uf)
            Lista ordenada por relevância (matches exatos primeiro,
            depois substring, depois fuzzy).
        """
        parcial_norm = parcial.strip().casefold()
        if not parcial_norm:
            return []

        nomes = self.df["nome_municipio"].astype(str)
        ufs = self.df["sigla_uf"].astype(str)

        # Etapa 1: substring
        contem = nomes.str.casefold().str.contains(parcial_norm, na=False)
        matches_substring = list(zip(nomes[contem], ufs[contem], strict=False))

        # Etapa 2: fuzzy (se sobrar espaço no orçamento)
        resultados: list[tuple[str, str]] = matches_substring[:max_results]
        if len(resultados) < max_results:
            todos_nomes = nomes.unique().tolist()
            fuzzy_hits = difflib.get_close_matches(
                parcial, todos_nomes, n=max_results, cutoff=0.7
            )
            for hit in fuzzy_hits:
                for uf in ufs[nomes == hit]:
                    tupla = (hit, uf)
                    if tupla not in resultados:
                        resultados.append(tupla)
                        if len(resultados) >= max_results:
                            break

        return resultados[:max_results]

    # -------------------------------------------------------------------------
    # Recomendação
    # -------------------------------------------------------------------------
    def _recommend_by_index(
        self,
        indice: int,
        k: int = 5,
        excluir_mesmo_uf: bool = False,
    ) -> list[RecomendacaoResult]:
        """Método base: dado índice na tabela, retorna top-k similares."""
        if k <= 0:
            raise ValueError(f"k deve ser positivo, recebido {k}")

        # Query: uma única linha da matriz de features
        query_vec = self.X[indice]

        # Similaridade cosseno em lote contra toda a matriz
        # Resultado é (1, n); pegamos a linha 0
        scores = similarity.cosine_similarity_matrix(query_vec, self.X)[0]

        excluir = [indice]
        if excluir_mesmo_uf:
            uf_query = self.df.iloc[indice]["sigla_uf"]
            excluir.extend(
                self.df.index[self.df["sigla_uf"] == uf_query].tolist()
            )

        top = similarity.top_k_similares(
            scores, k=k, excluir_indices=excluir
        )
        return [self._make_result(idx, score) for idx, score in top]

    def recommend_by_name(
        self,
        nome: str,
        uf: str | None = None,
        k: int = 5,
        excluir_mesmo_uf: bool = False,
    ) -> list[RecomendacaoResult]:
        """Recomenda os ``k`` municípios mais similares ao dado por nome.

        Parameters
        ----------
        nome : str
            Nome do município. Case e whitespace tolerados.
        uf : str, optional
            Sigla da UF para desambiguar homônimos (São Paulo, São José,
            existem em várias UFs).
        k : int
            Quantos resultados retornar.
        excluir_mesmo_uf : bool
            Se ``True``, remove recomendações da mesma UF do consultado
            — útil para buscar análogos em outras regiões, aplicação
            típica em benchmarking interestadual.

        Returns
        -------
        list of RecomendacaoResult
            Ordenados por similaridade decrescente.
        """
        indice = self._locate_by_name(nome, uf=uf)
        return self._recommend_by_index(indice, k=k, excluir_mesmo_uf=excluir_mesmo_uf)

    def recommend_by_code(
        self,
        code: int | str,
        k: int = 5,
        excluir_mesmo_uf: bool = False,
    ) -> list[RecomendacaoResult]:
        """Recomenda por código IBGE (7 dígitos)."""
        indice = self._locate_by_code(code)
        return self._recommend_by_index(indice, k=k, excluir_mesmo_uf=excluir_mesmo_uf)

    def recommend_by_tags(
        self,
        tags: str,
        k: int = 5,
    ) -> list[RecomendacaoResult]:
        """Recomenda a partir de uma string de tags customizada.

        Vetoriza a string usando o mesmo ``vectorizer`` ajustado (sem
        refit) e devolve os municípios mais próximos. Útil para queries
        hipotéticas: você monta manualmente uma tag como
        ``"sudeste alta_bovinocultura media_avicultura"`` e o sistema
        devolve municípios que casam com esse perfil.

        Parameters
        ----------
        tags : str
            String de tags no mesmo formato produzido pelo pipeline
            de features (tokens separados por espaço).
        k : int
            Quantos resultados retornar.
        """
        query_series = pd.Series([tags])
        query_vec = vectorize.transform(self.vectorizer, query_series)

        if query_vec.nnz == 0:
            raise ValueError(
                "Nenhum token da query bate com o vocabulário do vectorizer. "
                "Verifique se as tags usam o mesmo formato do dataset "
                "(ex: 'sudeste alta_bovinocultura media_avicultura')."
            )

        scores = similarity.cosine_similarity_matrix(query_vec, self.X)[0]
        top = similarity.top_k_similares(scores, k=k)
        return [self._make_result(idx, score) for idx, score in top]

    # -------------------------------------------------------------------------
    # Explicação
    # -------------------------------------------------------------------------
    def explain(
        self,
        query_ref: str | int,
        recomendado_ref: str | int,
        query_uf: str | None = None,
        recomendado_uf: str | None = None,
    ) -> Explicacao:
        """Decompõe uma recomendação em tokens compartilhados e distintos.

        Retorna as três métricas de distância (cosseno, euclidiana, manhattan)
        e o conjunto de tokens em comum vs. exclusivos. Útil para responder
        "por que este município foi recomendado?"

        Parameters
        ----------
        query_ref : str or int
            Nome (str) ou código IBGE (int) do município consultado.
        recomendado_ref : str or int
            Idem para o recomendado.
        query_uf, recomendado_uf : str, optional
            UF para desambiguação se necessário.
        """
        idx_q = self._resolver_ref(query_ref, query_uf)
        idx_r = self._resolver_ref(recomendado_ref, recomendado_uf)

        row_q = self.df.iloc[idx_q]
        row_r = self.df.iloc[idx_r]

        vec_q = self.X[idx_q]
        vec_r = self.X[idx_r]

        # Métricas
        cos = similarity.cosine_similarity_pair(vec_q, vec_r)
        euc = similarity.euclidean_distance_pair(vec_q, vec_r)
        man = similarity.manhattan_distance_pair(vec_q, vec_r)

        # Tokens
        tokens_q = set(row_q["tags"].split())
        tokens_r = set(row_r["tags"].split())
        em_comum = sorted(tokens_q & tokens_r)
        so_q = sorted(tokens_q - tokens_r)
        so_r = sorted(tokens_r - tokens_q)

        return Explicacao(
            query_nome=str(row_q["nome_municipio"]),
            query_uf=str(row_q["sigla_uf"]),
            recomendado_nome=str(row_r["nome_municipio"]),
            recomendado_uf=str(row_r["sigla_uf"]),
            similaridade_cosseno=cos,
            distancia_euclidiana=euc,
            distancia_manhattan=man,
            tokens_em_comum=em_comum,
            tokens_so_query=so_q,
            tokens_so_recomendado=so_r,
        )

    def _resolver_ref(
        self,
        ref: str | int,
        uf: str | None = None,
    ) -> int:
        """Resolve referência (nome ou código) para índice."""
        if isinstance(ref, int):
            return self._locate_by_code(ref)
        if isinstance(ref, str) and ref.isdigit():
            return self._locate_by_code(int(ref))
        return self._locate_by_name(str(ref), uf=uf)

    # -------------------------------------------------------------------------
    # Utilitário para converter linha do DataFrame em RecomendacaoResult
    # -------------------------------------------------------------------------
    def _make_result(self, indice: int, similaridade: float) -> RecomendacaoResult:
        row = self.df.iloc[indice]
        return RecomendacaoResult(
            id_municipio=int(row["id_municipio"]),
            nome=str(row["nome_municipio"]),
            uf=str(row["sigla_uf"]),
            mesorregiao=str(row.get("nome_mesorregiao", "")),
            regiao=str(row.get("nome_regiao", "")),
            especializacao=str(row.get("especializacao", "")),
            similaridade=float(similaridade),
            tags=str(row["tags"]),
        )


# =============================================================================
# CLI
# =============================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_recomendacoes(
    query_desc: str,
    resultados: list[RecomendacaoResult],
) -> None:
    """Imprime recomendações em tabela formatada."""
    print(f"\n>>> TOP {len(resultados)} municípios similares a {query_desc}:\n")
    header = f"  {'sim':<6}  {'código':<8}  {'UF':<3}  {'município':<35}  {'mesorregião':<35}  especialização"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in resultados:
        nome_uf = f"{r.nome[:33]}"
        meso = f"{r.mesorregiao[:33]}"
        esp = r.especializacao.replace("especializado_em_", "").replace("_", " ")
        print(
            f"  {r.similaridade:<6.4f}  {r.id_municipio:<8}  {r.uf:<3}  "
            f"{nome_uf:<35}  {meso:<35}  {esp}"
        )


def _print_busca(parcial: str, hits: list[tuple[str, str]]) -> None:
    if not hits:
        print(f"\n[AVISO] Nenhum município encontrado para '{parcial}'.")
        return
    print(f"\n>>> {len(hits)} municípios com nome contendo '{parcial}':\n")
    for nome, uf in hits:
        print(f"  [{uf}] {nome}")


def _print_explicacao(exp: Explicacao) -> None:
    print(f"\n>>> Explicação: {exp.query_nome} ({exp.query_uf})  vs  "
          f"{exp.recomendado_nome} ({exp.recomendado_uf})\n")
    print(f"  Similaridade cosseno:  {exp.similaridade_cosseno:.4f}")
    print(f"  Distância euclidiana:  {exp.distancia_euclidiana:.4f}")
    print(f"  Distância Manhattan:   {exp.distancia_manhattan:.4f}")
    print(f"\n  Tokens em comum ({len(exp.tokens_em_comum)}):")
    for t in exp.tokens_em_comum:
        print(f"    - {t}")
    if exp.tokens_so_query:
        print(f"\n  Tokens só em {exp.query_nome} ({len(exp.tokens_so_query)}):")
        for t in exp.tokens_so_query:
            print(f"    - {t}")
    if exp.tokens_so_recomendado:
        print(f"\n  Tokens só em {exp.recomendado_nome} "
              f"({len(exp.tokens_so_recomendado)}):")
        for t in exp.tokens_so_recomendado:
            print(f"    - {t}")


def _cmd_recommend(args: argparse.Namespace) -> int:
    rec = MunicipioRecommender.load()

    if args.search:
        hits = rec.search(args.search, max_results=args.k)
        _print_busca(args.search, hits)
        return 0

    if args.code:
        resultados = rec.recommend_by_code(
            args.code, k=args.k, excluir_mesmo_uf=args.excluir_mesmo_uf
        )
        query_desc = f"código IBGE {args.code}"
    elif args.tags:
        resultados = rec.recommend_by_tags(args.tags, k=args.k)
        query_desc = f'tags "{args.tags[:60]}..."'
    elif args.nome:
        resultados = rec.recommend_by_name(
            args.nome,
            uf=args.uf,
            k=args.k,
            excluir_mesmo_uf=args.excluir_mesmo_uf,
        )
        query_desc = f"{args.nome}" + (f"/{args.uf}" if args.uf else "")
    else:
        print("[ERRO] Passe --nome, --code, --tags ou --search.", file=sys.stderr)
        return 2

    _print_recomendacoes(query_desc, resultados)

    if args.explicar and not args.search and resultados:
        # Explica a top-1
        top1 = resultados[0]
        ref_query = args.code or args.nome
        exp = rec.explain(
            query_ref=ref_query,
            recomendado_ref=top1.id_municipio,
            query_uf=args.uf,
        )
        _print_explicacao(exp)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rec_agro_br.recommender",
        description=(
            "Sistema de recomendação de municípios brasileiros por perfil "
            "agropecuário. Consulta por nome, código IBGE, tags customizadas, "
            "ou busca por nome parcial."
        ),
    )
    parser.add_argument("nome", nargs="?", help="Nome do município (default).")
    parser.add_argument("--uf", help="Sigla da UF (para desambiguar homônimos).")
    parser.add_argument("--code", help="Consulta por código IBGE de 7 dígitos.")
    parser.add_argument(
        "--tags",
        help='Consulta por string de tags (ex: "sudeste alta_bovinocultura").',
    )
    parser.add_argument(
        "--search",
        help="Busca municípios cujo nome contenha o fragmento dado.",
    )
    parser.add_argument("--k", type=int, default=5, help="Número de resultados.")
    parser.add_argument(
        "--excluir-mesmo-uf",
        action="store_true",
        help="Remove recomendações da mesma UF (busca análogos interestaduais).",
    )
    parser.add_argument(
        "--explicar",
        action="store_true",
        help="Após listar recomendações, explica a top-1 em detalhe.",
    )
    parser.set_defaults(func=_cmd_recommend)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        logger.error(
            "[ERRO] Artefato ausente: %s. Rode antes: "
            "python -m rec_agro_br.features && python -m rec_agro_br.vectorize",
            e,
        )
        return 3
    except MunicipioNaoEncontradoError as e:
        logger.error("[ERRO] %s", e)
        return 4
    except NomeAmbiguoError as e:
        logger.error("[ERRO] %s", e)
        return 5
    except Exception as e:
        logger.exception("[ERRO] Falha inesperada: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
