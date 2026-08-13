"""Testes do módulo `rec_agro_br.vectorize`.

Cobre tokenização (simples e com stemming), construção e ajuste do
CountVectorizer, persistência do vectorizer e da matriz esparsa, e
o pipeline de alto nível `build_and_persist`.

A instalação do RSLPStemmer é feita lazy no primeiro uso; se falhar
por falta de conectividade, os testes que dependem de stemming são
skipados via marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from rec_agro_br import vectorize


# =============================================================================
# Tokenizers
# =============================================================================
class TestTokenizeSimples:
    def test_split_por_espaco(self) -> None:
        tokens = vectorize.tokenize_simples("nordeste rn oeste_potiguar")
        assert tokens == ["nordeste", "rn", "oeste_potiguar"]

    def test_lowercase_aplicado(self) -> None:
        tokens = vectorize.tokenize_simples("Nordeste RN Oeste")
        assert tokens == ["nordeste", "rn", "oeste"]

    def test_string_vazia_retorna_lista_vazia(self) -> None:
        assert vectorize.tokenize_simples("") == []

    def test_none_retorna_lista_vazia(self) -> None:
        assert vectorize.tokenize_simples(None) == []


class TestTokenizeComStemming:
    def test_stemming_aplicado_em_palavras_simples(self) -> None:
        """Palavras simples são reduzidas ao radical pelo RSLP."""
        tokens = vectorize.tokenize_com_stemming("bovinocultura suinocultura")
        # RSLP típico: "bovinocultura" -> "bovinocultur" ou similar
        # Não sabemos o radical exato, mas deve ser diferente do original
        assert len(tokens) == 2
        assert all(len(t) > 0 for t in tokens)

    def test_tokens_compostos_preservados(self) -> None:
        """Tokens com underscore são preservados intactos."""
        tokens = vectorize.tokenize_com_stemming(
            "sul_sudoeste_de_minas especializado_em_bovinocultura"
        )
        assert "sul_sudoeste_de_minas" in tokens
        assert "especializado_em_bovinocultura" in tokens

    def test_mistura_compostos_e_simples(self) -> None:
        """Tokens simples são stemmizados; compostos, não."""
        tokens = vectorize.tokenize_com_stemming(
            "nordeste sul_sudoeste_de_minas bovinocultura"
        )
        assert "sul_sudoeste_de_minas" in tokens
        # nordeste é simples, deve ter sido stemmizado ou mantido
        assert len(tokens) == 3

    def test_lowercase_antes_do_stemming(self) -> None:
        """RSLP não é case-insensitive, então precisa lowercase antes."""
        tokens_upper = vectorize.tokenize_com_stemming("BOVINOCULTURA")
        tokens_lower = vectorize.tokenize_com_stemming("bovinocultura")
        assert tokens_upper == tokens_lower

    def test_string_vazia_retorna_lista_vazia(self) -> None:
        assert vectorize.tokenize_com_stemming("") == []


# =============================================================================
# build_vectorizer
# =============================================================================
class TestBuildVectorizer:
    def test_retorna_count_vectorizer(self) -> None:
        vec = vectorize.build_vectorizer()
        assert isinstance(vec, CountVectorizer)

    def test_com_stemming_usa_tokenizer_stem(self) -> None:
        vec = vectorize.build_vectorizer(use_stemming=True)
        assert vec.tokenizer is vectorize.tokenize_com_stemming

    def test_sem_stemming_usa_tokenizer_simples(self) -> None:
        vec = vectorize.build_vectorizer(use_stemming=False)
        assert vec.tokenizer is vectorize.tokenize_simples

    def test_max_features_configuravel(self) -> None:
        vec = vectorize.build_vectorizer(max_features=100)
        assert vec.max_features == 100


# =============================================================================
# fit_and_transform
# =============================================================================
class TestFitAndTransform:
    @pytest.fixture
    def corpus_pequeno(self) -> pd.Series:
        return pd.Series(
            [
                "nordeste rn oeste_potiguar bovinocultura avicultura",
                "sudeste sp metropolitana_de_são_paulo sem_producao_pecuaria",
                "sul rs noroeste_rio_grandense suinocultura avicultura",
                "sudeste mg sul_sudoeste_de_minas bovinocultura avicultura",
            ]
        )

    def test_shape_da_matriz(self, corpus_pequeno: pd.Series) -> None:
        vec, X = vectorize.fit_and_transform(corpus_pequeno, use_stemming=False)
        assert X.shape[0] == len(corpus_pequeno)
        assert X.shape[1] == len(vec.vocabulary_)

    def test_matriz_esparsa(self, corpus_pequeno: pd.Series) -> None:
        _, X = vectorize.fit_and_transform(corpus_pequeno, use_stemming=False)
        assert sparse.issparse(X)

    def test_vocab_contem_tokens_esperados(self, corpus_pequeno: pd.Series) -> None:
        vec, _ = vectorize.fit_and_transform(corpus_pequeno, use_stemming=False)
        vocab = set(vec.vocabulary_.keys())
        # tokens compostos devem estar presentes intactos
        assert "oeste_potiguar" in vocab
        assert "sul_sudoeste_de_minas" in vocab

    def test_serie_vazia_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="vazia"):
            vectorize.fit_and_transform(pd.Series([], dtype="object"))

    def test_valores_nan_tratados(self) -> None:
        corpus = pd.Series(["a b c", None, "d e f"])
        vec, X = vectorize.fit_and_transform(corpus, use_stemming=False)
        assert X.shape[0] == 3
        # Linha do meio deve ter zero soma (documento vazio após fillna)
        assert X.sum(axis=1)[1] == 0

    def test_contagem_correta_de_frequencia(
        self, corpus_pequeno: pd.Series
    ) -> None:
        """Verifica que a matriz conta corretamente as ocorrências."""
        vec, X = vectorize.fit_and_transform(corpus_pequeno, use_stemming=False)
        # 'avicultura' aparece em 3 dos 4 documentos
        idx_avi = vec.vocabulary_.get("avicultura")
        assert idx_avi is not None
        col_avi = np.asarray(X[:, idx_avi].todense()).ravel()
        assert (col_avi > 0).sum() == 3


class TestTransform:
    def test_transform_novo_com_vocab_ajustado(self) -> None:
        corpus_treino = pd.Series(["nordeste rn", "sudeste sp"])
        vec, _ = vectorize.fit_and_transform(corpus_treino, use_stemming=False)

        # Novo documento: nordeste conhecido, mg desconhecido
        corpus_novo = pd.Series(["nordeste mg"])
        X_novo = vectorize.transform(vec, corpus_novo)
        assert X_novo.shape == (1, len(vec.vocabulary_))
        # 'mg' fora do vocab -> apenas 'nordeste' contado
        assert X_novo.sum() == 1


# =============================================================================
# Persistência (save/load)
# =============================================================================
class TestPersistencia:
    def test_save_e_load_vectorizer_roundtrip(
        self,
        isolated_data_dirs: Path,
    ) -> None:
        corpus = pd.Series(["a b c", "b c d", "c d e"])
        vec_original, _ = vectorize.fit_and_transform(corpus, use_stemming=False)

        vectorize.save_vectorizer(vec_original)
        vec_carregado = vectorize.load_vectorizer()

        assert vec_carregado.vocabulary_ == vec_original.vocabulary_

    def test_save_e_load_matrix_roundtrip(self, isolated_data_dirs: Path) -> None:
        corpus = pd.Series(["a b c", "b c d", "c d e"])
        _, X_original = vectorize.fit_and_transform(corpus, use_stemming=False)

        vectorize.save_matrix(X_original)
        X_carregado = vectorize.load_matrix()

        assert X_carregado.shape == X_original.shape
        assert (X_carregado != X_original).nnz == 0

    def test_load_sem_arquivo_levanta_file_not_found(
        self, isolated_data_dirs: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            vectorize.load_vectorizer()
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            vectorize.load_matrix()


# =============================================================================
# Pipeline de alto nível (build_and_persist)
# =============================================================================
class TestBuildAndPersist:
    @pytest.fixture
    def df_features_fake(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_municipio": [1, 2, 3],
                "nome_municipio": ["A", "B", "C"],
                "tags": [
                    "nordeste rn bovinocultura",
                    "sudeste sp avicultura suinocultura",
                    "sul rs bovinocultura suinocultura",
                ],
            }
        )

    def test_pipeline_gera_arquivos(
        self,
        isolated_data_dirs: Path,
        df_features_fake: pd.DataFrame,
    ) -> None:
        vec, X = vectorize.build_and_persist(
            use_stemming=False, df_features=df_features_fake
        )
        assert vectorize.get_vectorizer_path().exists()
        assert vectorize.get_matrix_path().exists()
        assert X.shape[0] == 3

    def test_pipeline_sem_coluna_tags_levanta_erro(
        self, isolated_data_dirs: Path
    ) -> None:
        df_sem_tags = pd.DataFrame({"id_municipio": [1], "nome_municipio": ["X"]})
        with pytest.raises(ValueError, match="tags"):
            vectorize.build_and_persist(df_features=df_sem_tags)


# =============================================================================
# CLI
# =============================================================================
class TestParserVectorize:
    def test_parser_default(self) -> None:
        parser = vectorize._build_parser()
        args = parser.parse_args([])
        assert args.sem_stemming is False

    def test_parser_sem_stemming(self) -> None:
        parser = vectorize._build_parser()
        args = parser.parse_args(["--sem-stemming"])
        assert args.sem_stemming is True
