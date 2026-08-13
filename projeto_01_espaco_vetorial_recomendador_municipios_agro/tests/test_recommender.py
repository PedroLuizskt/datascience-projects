"""Testes do módulo `rec_agro_br.recommender`.

Cobre construção da classe, resolução de nome/código, busca fuzzy, todos
os métodos de recomendação (`by_name`, `by_code`, `by_tags`), explicação
e tratamento de erros (município ausente, nome ambíguo, artefato ausente).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rec_agro_br import recommender, vectorize
from rec_agro_br.recommender import (
    Explicacao,
    MunicipioNaoEncontradoError,
    MunicipioRecommender,
    NomeAmbiguoError,
    RecomendacaoResult,
)


# =============================================================================
# Fixtures locais: dataset e matriz sintéticos
# =============================================================================
@pytest.fixture
def df_para_recommender() -> pd.DataFrame:
    """DataFrame com 6 municípios com tags plausíveis."""
    return pd.DataFrame(
        {
            "id_municipio": [3111606, 3550308, 4314902, 1302603, 3143302, 3170107],
            "nome_municipio": [
                "Cambuquira",
                "São Paulo",
                "Porto Alegre",
                "Manaus",
                "Sao Paulo",  # Homônimo em MG para testar ambiguidade
                "Uberlândia",
            ],
            "sigla_uf": ["MG", "SP", "RS", "AM", "MG", "MG"],
            "nome_mesorregiao": [
                "Sul/Sudoeste de Minas",
                "Metropolitana de São Paulo",
                "Metropolitana de Porto Alegre",
                "Centro Amazonense",
                "Sul/Sudoeste de Minas",
                "Triângulo Mineiro",
            ],
            "nome_regiao": ["Sudeste", "Sudeste", "Sul", "Norte", "Sudeste", "Sudeste"],
            "especializacao": [
                "especializado_em_avicultura",
                "sem_producao_pecuaria",
                "especializado_em_bovinocultura",
                "especializado_em_bovinocultura",
                "especializado_em_avicultura",
                "especializado_em_bovinocultura",
            ],
            "tags": [
                "sudeste mg sul_sudoeste_de_minas alta_avicultura media_bovinocultura",
                "sudeste sp metropolitana_de_são_paulo sem_producao_pecuaria",
                "sul rs metropolitana_de_porto_alegre alta_bovinocultura",
                "norte am centro_amazonense alta_bovinocultura",
                "sudeste mg sul_sudoeste_de_minas alta_avicultura baixa_bovinocultura",
                "sudeste mg triângulo_mineiro alta_bovinocultura alta_avicultura",
            ],
        }
    )


@pytest.fixture
def recommender_pronto(df_para_recommender: pd.DataFrame) -> MunicipioRecommender:
    """Recommender construído e pronto a usar com o dataset sintético."""
    vec, X = vectorize.fit_and_transform(
        df_para_recommender["tags"], use_stemming=False
    )
    return MunicipioRecommender(df=df_para_recommender, vectorizer=vec, X=X)


# =============================================================================
# Construção e validação
# =============================================================================
class TestConstrucao:
    def test_instancia_valida(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        assert len(recommender_pronto.df) == 6
        assert recommender_pronto.X.shape[0] == 6

    def test_inconsistencia_de_tamanhos_levanta_erro(
        self, df_para_recommender: pd.DataFrame
    ) -> None:
        vec, X = vectorize.fit_and_transform(
            df_para_recommender["tags"], use_stemming=False
        )
        # Remove uma linha do df para introduzir inconsistência
        df_pequeno = df_para_recommender.head(3)
        with pytest.raises(ValueError, match="Inconsistência"):
            MunicipioRecommender(df=df_pequeno, vectorizer=vec, X=X)

    def test_coluna_obrigatoria_ausente_levanta_erro(
        self, df_para_recommender: pd.DataFrame
    ) -> None:
        vec, X = vectorize.fit_and_transform(
            df_para_recommender["tags"], use_stemming=False
        )
        df_sem_tags = df_para_recommender.drop(columns=["tags"])
        with pytest.raises(ValueError, match="obrigatória ausente"):
            MunicipioRecommender(df=df_sem_tags, vectorizer=vec, X=X)


# =============================================================================
# Resolução de município por nome
# =============================================================================
class TestLocateByName:
    def test_nome_exato(self, recommender_pronto: MunicipioRecommender) -> None:
        idx = recommender_pronto._locate_by_name("Cambuquira")
        assert recommender_pronto.df.iloc[idx]["nome_municipio"] == "Cambuquira"

    def test_case_insensitive(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        idx = recommender_pronto._locate_by_name("CAMBUQUIRA")
        assert recommender_pronto.df.iloc[idx]["nome_municipio"] == "Cambuquira"

    def test_whitespace_tolerado(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        idx = recommender_pronto._locate_by_name("  Cambuquira  ")
        assert recommender_pronto.df.iloc[idx]["nome_municipio"] == "Cambuquira"

    def test_nome_inexistente_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        with pytest.raises(MunicipioNaoEncontradoError, match="Cidade Inexistente"):
            recommender_pronto._locate_by_name("Cidade Inexistente")

    def test_nome_ambiguo_sem_uf_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        # 'São Paulo' existe em SP e 'Sao Paulo' em MG (nomes distintos casefolded)
        # Vamos usar caso real: adicionar homônimo verdadeiro na fixture.
        # A fixture atual tem 'São Paulo' apenas em SP e 'Sao Paulo' apenas em MG.
        # Casefold trata acentos igual? Não — "São" != "Sao" mesmo com casefold.
        # Vou fazer um caso ambíguo direto:
        df = recommender_pronto.df.copy()
        df.at[3, "nome_municipio"] = "Cambuquira"  # Manaus vira Cambuquira
        rec = MunicipioRecommender(
            df=df, vectorizer=recommender_pronto.vectorizer, X=recommender_pronto.X
        )
        with pytest.raises(NomeAmbiguoError, match="Múltiplos"):
            rec._locate_by_name("Cambuquira")

    def test_ambiguo_com_uf_desambigua(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        df = recommender_pronto.df.copy()
        df.at[3, "nome_municipio"] = "Cambuquira"
        df.at[3, "sigla_uf"] = "AM"
        rec = MunicipioRecommender(
            df=df, vectorizer=recommender_pronto.vectorizer, X=recommender_pronto.X
        )
        idx_mg = rec._locate_by_name("Cambuquira", uf="MG")
        idx_am = rec._locate_by_name("Cambuquira", uf="AM")
        assert rec.df.iloc[idx_mg]["sigla_uf"] == "MG"
        assert rec.df.iloc[idx_am]["sigla_uf"] == "AM"


# =============================================================================
# Resolução por código IBGE
# =============================================================================
class TestLocateByCode:
    def test_codigo_valido(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        idx = recommender_pronto._locate_by_code(3111606)
        assert recommender_pronto.df.iloc[idx]["nome_municipio"] == "Cambuquira"

    def test_codigo_como_string(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        idx = recommender_pronto._locate_by_code("3111606")
        assert recommender_pronto.df.iloc[idx]["nome_municipio"] == "Cambuquira"

    def test_codigo_inexistente_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        with pytest.raises(MunicipioNaoEncontradoError, match="9999999"):
            recommender_pronto._locate_by_code(9999999)

    def test_codigo_invalido_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        with pytest.raises(MunicipioNaoEncontradoError, match="inválido"):
            recommender_pronto._locate_by_code("nao_e_numero")


# =============================================================================
# Busca (search)
# =============================================================================
class TestSearch:
    def test_substring_encontra(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        hits = recommender_pronto.search("Cambu")
        nomes = [nome for nome, _ in hits]
        assert "Cambuquira" in nomes

    def test_case_insensitive(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        hits = recommender_pronto.search("CAMBU")
        assert len(hits) > 0

    def test_string_vazia_retorna_vazio(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        assert recommender_pronto.search("") == []

    def test_fuzzy_encontra_variacao(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        # 'Uberlandia' sem acento deve ainda achar 'Uberlândia'
        hits = recommender_pronto.search("Uberlandia")
        nomes = [nome for nome, _ in hits]
        assert any("Uberlândia" in n for n in nomes)


# =============================================================================
# recommend_by_name
# =============================================================================
class TestRecommendByName:
    def test_retorna_k_resultados(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_name("Cambuquira", k=3)
        assert len(results) == 3
        assert all(isinstance(r, RecomendacaoResult) for r in results)

    def test_nao_inclui_o_proprio_municipio(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_name("Cambuquira", k=5)
        nomes_ufs = [(r.nome, r.uf) for r in results]
        assert ("Cambuquira", "MG") not in nomes_ufs

    def test_ordenado_por_similaridade_decrescente(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_name("Cambuquira", k=5)
        similaridades = [r.similaridade for r in results]
        assert similaridades == sorted(similaridades, reverse=True)

    def test_similares_dentro_da_mesma_mesorregiao_vencem(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        """Cambuquira (Sul/Sudoeste de Minas) deve ter Sao Paulo/MG
        (mesma mesorregião) na top-1."""
        results = recommender_pronto.recommend_by_name("Cambuquira", k=1)
        assert results[0].nome == "Sao Paulo"  # o de MG
        assert results[0].uf == "MG"

    def test_excluir_mesmo_uf(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_name(
            "Cambuquira", k=5, excluir_mesmo_uf=True
        )
        assert all(r.uf != "MG" for r in results)

    def test_k_zero_ou_negativo_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        with pytest.raises(ValueError, match="positivo"):
            recommender_pronto.recommend_by_name("Cambuquira", k=0)


# =============================================================================
# recommend_by_code
# =============================================================================
class TestRecommendByCode:
    def test_por_codigo(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_code(3111606, k=3)
        assert len(results) == 3
        # Não deve incluir Cambuquira (código 3111606)
        assert all(r.id_municipio != 3111606 for r in results)


# =============================================================================
# recommend_by_tags
# =============================================================================
class TestRecommendByTags:
    def test_tags_customizadas(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        results = recommender_pronto.recommend_by_tags(
            "sul rs alta_bovinocultura", k=2
        )
        # Porto Alegre tem tags que casam
        nomes = [r.nome for r in results]
        assert "Porto Alegre" in nomes

    def test_tags_vazias_ou_desconhecidas_levanta_erro(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        with pytest.raises(ValueError, match="vocabulário"):
            recommender_pronto.recommend_by_tags(
                "token_totalmente_inexistente", k=1
            )


# =============================================================================
# explain
# =============================================================================
class TestExplain:
    def test_explicacao_retorna_dataclass(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        exp = recommender_pronto.explain(
            query_ref="Cambuquira",
            recomendado_ref="Sao Paulo",  # o de MG (mesma mesorregião)
            query_uf="MG",
            recomendado_uf="MG",
        )
        assert isinstance(exp, Explicacao)

    def test_tokens_em_comum_incluem_contexto(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        exp = recommender_pronto.explain(
            query_ref="Cambuquira",
            recomendado_ref="Sao Paulo",
            query_uf="MG",
            recomendado_uf="MG",
        )
        # Ambos são da mesma mesorregião
        assert "sul_sudoeste_de_minas" in exp.tokens_em_comum
        assert "sudeste" in exp.tokens_em_comum
        assert "mg" in exp.tokens_em_comum

    def test_metricas_computadas(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        exp = recommender_pronto.explain(
            query_ref="Cambuquira",
            recomendado_ref="Porto Alegre",
            query_uf="MG",
        )
        assert 0.0 <= exp.similaridade_cosseno <= 1.0
        assert exp.distancia_euclidiana >= 0.0
        assert exp.distancia_manhattan >= 0.0

    def test_referencia_por_codigo_ibge(
        self, recommender_pronto: MunicipioRecommender
    ) -> None:
        exp = recommender_pronto.explain(
            query_ref=3111606,  # Cambuquira
            recomendado_ref=4314902,  # Porto Alegre
        )
        assert exp.query_nome == "Cambuquira"
        assert exp.recomendado_nome == "Porto Alegre"


# =============================================================================
# CLI (parser)
# =============================================================================
class TestParserRecommender:
    def test_nome_positional(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["Cambuquira"])
        assert args.nome == "Cambuquira"
        assert args.k == 5

    def test_nome_com_uf_e_k(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["Cambuquira", "--uf", "MG", "--k", "10"])
        assert args.nome == "Cambuquira"
        assert args.uf == "MG"
        assert args.k == 10

    def test_code(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["--code", "3111606"])
        assert args.code == "3111606"

    def test_tags(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["--tags", "sudeste alta_bovinocultura"])
        assert args.tags == "sudeste alta_bovinocultura"

    def test_search(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["--search", "Cambu"])
        assert args.search == "Cambu"

    def test_excluir_mesmo_uf_flag(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["Cambuquira", "--excluir-mesmo-uf"])
        assert args.excluir_mesmo_uf is True

    def test_explicar_flag(self) -> None:
        parser = recommender._build_parser()
        args = parser.parse_args(["Cambuquira", "--explicar"])
        assert args.explicar is True
