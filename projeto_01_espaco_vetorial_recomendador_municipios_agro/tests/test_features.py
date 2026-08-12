"""Testes do módulo `rec_agro_br.features`.

Cobre cada estágio do pipeline de feature engineering isoladamente
(clean, pivot, merge, derive_*, build_tags) e o pipeline end-to-end
(build_features_dataset). Todos são unit tests offline usando as
fixtures ``ppm_raw_minimo`` e ``localidades_minimo`` do conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_agro_br import features


# =============================================================================
# Utilitários de normalização (funções privadas)
# =============================================================================
class TestNormalizarTexto:
    def test_strip_e_colapso_de_whitespace(self) -> None:
        assert features._normalizar_texto("  Ovino  ") == "Ovino"
        assert features._normalizar_texto("Suíno\t-\ntotal") == "Suíno - total"

    def test_nan_vira_string_vazia(self) -> None:
        assert features._normalizar_texto(np.nan) == ""
        assert features._normalizar_texto(None) == ""


class TestToSnake:
    def test_espacos_viram_underscore(self) -> None:
        assert features._to_snake("Rio Grande do Sul") == "rio_grande_do_sul"

    def test_barra_e_separadores_normalizados(self) -> None:
        assert features._to_snake("Sul/Sudoeste de Minas") == "sul_sudoeste_de_minas"

    def test_case_lowered(self) -> None:
        assert features._to_snake("MINAS GERAIS") == "minas_gerais"

    def test_acentos_preservados(self) -> None:
        assert features._to_snake("São Paulo") == "são_paulo"

    def test_nan_vira_string_vazia(self) -> None:
        assert features._to_snake(np.nan) == ""


# =============================================================================
# clean_ppm
# =============================================================================
class TestCleanPPM:
    def test_shape_apos_filtragem_de_redundantes(
        self, ppm_raw_minimo: pd.DataFrame
    ) -> None:
        """Fixture tem 9 linhas; 1 é 'Suíno - matrizes desuínos' (ignorada)."""
        df = features.clean_ppm(ppm_raw_minimo)
        # 9 linhas - 1 matriz de suínos = 8
        assert len(df) == 8

    def test_colunas_finais(self, ppm_raw_minimo: pd.DataFrame) -> None:
        df = features.clean_ppm(ppm_raw_minimo)
        assert set(df.columns) == {"id_municipio", "ano", "atividade", "valor"}

    def test_tipos_das_colunas(self, ppm_raw_minimo: pd.DataFrame) -> None:
        df = features.clean_ppm(ppm_raw_minimo)
        assert str(df["id_municipio"].dtype) == "int64"
        assert str(df["ano"].dtype) == "int64"
        assert str(df["valor"].dtype) == "Float64"

    def test_hifen_convertido_para_zero(
        self, ppm_raw_minimo: pd.DataFrame
    ) -> None:
        """São Paulo (SP) tem valor '-' para bovinos, deve virar 0.0."""
        df = features.clean_ppm(ppm_raw_minimo)
        sp_bovinos = df[
            (df["id_municipio"] == 3550308) & (df["atividade"] == "bovinocultura")
        ]
        assert len(sp_bovinos) == 1
        assert sp_bovinos["valor"].iloc[0] == 0.0

    def test_atividades_mapeadas_para_nomes_canonicos(
        self, ppm_raw_minimo: pd.DataFrame
    ) -> None:
        df = features.clean_ppm(ppm_raw_minimo)
        atividades = set(df["atividade"].unique())
        assert atividades.issubset(
            {"bovinocultura", "suinocultura", "avicultura"}
        )
        assert "matrizes" not in " ".join(atividades)

    def test_whitespace_irregular_tolerado(
        self, ppm_raw_minimo: pd.DataFrame
    ) -> None:
        """Cambuquira tem ' Bovino ' (com espaços) na fixture — deve mapear."""
        df = features.clean_ppm(ppm_raw_minimo)
        cambu_bovinos = df[
            (df["id_municipio"] == 3111606) & (df["atividade"] == "bovinocultura")
        ]
        assert len(cambu_bovinos) == 1
        assert cambu_bovinos["valor"].iloc[0] == 10000.0

    def test_ausencia_de_coluna_esperada_levanta_value_error(self) -> None:
        df_incompleto = pd.DataFrame({"NC": ["6"], "V": ["100"]})
        with pytest.raises(ValueError, match="Colunas esperadas ausentes"):
            features.clean_ppm(df_incompleto)


# =============================================================================
# pivot_ppm_wide
# =============================================================================
class TestPivotWide:
    def test_shape_wide(self, ppm_raw_minimo: pd.DataFrame) -> None:
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        # 3 municípios únicos, 1 col id + 3 atividades (bov, suin, avi)
        assert df_wide.shape == (3, 4)

    def test_colunas_incluem_id_e_atividades(
        self, ppm_raw_minimo: pd.DataFrame
    ) -> None:
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        assert "id_municipio" in df_wide.columns
        assert "bovinocultura" in df_wide.columns
        assert "suinocultura" in df_wide.columns
        assert "avicultura" in df_wide.columns

    def test_ausencias_viram_zero(self, ppm_raw_minimo: pd.DataFrame) -> None:
        """Cambuquira não tem linha PPM para avicultura na fixture — deve virar 0."""
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        cambu = df_wide[df_wide["id_municipio"] == 3111606]
        # Cambuquira tem avicultura=5000 na fixture. Vamos testar outra ausência:
        # SP não tem linha para avicultura → deve virar 0
        sp = df_wide[df_wide["id_municipio"] == 3550308]
        assert sp["avicultura"].iloc[0] == 0.0

    def test_valores_preservados(self, ppm_raw_minimo: pd.DataFrame) -> None:
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        acre = df_wide[df_wide["id_municipio"] == 1200013]
        assert acre["bovinocultura"].iloc[0] == 50000.0
        assert acre["suinocultura"].iloc[0] == 3000.0
        assert acre["avicultura"].iloc[0] == 80000.0


# =============================================================================
# merge_com_localidades
# =============================================================================
class TestMerge:
    def test_left_join_preserva_todas_localidades(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        """Localidades tem 4 municípios; PPM só cobre 3. Merge deve ter 4."""
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        df_merged = features.merge_com_localidades(df_wide, localidades_minimo)
        assert len(df_merged) == 4
        assert set(df_merged["id_municipio"]) == {
            1200013,
            3111606,
            3550308,
            1302603,
        }

    def test_municipio_orfao_recebe_zeros_nas_atividades(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        """Manaus (1302603) não aparece na PPM — atividades devem ser 0."""
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        df_merged = features.merge_com_localidades(df_wide, localidades_minimo)
        manaus = df_merged[df_merged["id_municipio"] == 1302603].iloc[0]
        for atv in ["bovinocultura", "suinocultura", "avicultura"]:
            assert manaus[atv] == 0.0, f"{atv} em Manaus deveria ser 0"

    def test_colunas_de_contexto_preservadas(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        df_merged = features.merge_com_localidades(df_wide, localidades_minimo)
        for col in features.COLUNAS_CONTEXTO_TERRITORIAL:
            assert col in df_merged.columns


# =============================================================================
# derive_perfis_agropecuarios
# =============================================================================
class TestDerivePerfis:
    @pytest.fixture
    def df_com_atividades(self) -> pd.DataFrame:
        """DataFrame sintético com 10 municípios e 1 atividade."""
        return pd.DataFrame(
            {
                "id_municipio": range(1, 11),
                "bovinocultura": [0, 0, 100, 200, 500, 1000, 5000, 10000, 50000, 100000],
            }
        )

    def test_zero_vira_sem(self, df_com_atividades: pd.DataFrame) -> None:
        df = features.derive_perfis_agropecuarios(
            df_com_atividades, atividades=["bovinocultura"]
        )
        assert (df.loc[df["bovinocultura"] == 0, "perfil_bovinocultura"] == "sem_bovinocultura").all()

    def test_categorias_geradas(self, df_com_atividades: pd.DataFrame) -> None:
        df = features.derive_perfis_agropecuarios(
            df_com_atividades, atividades=["bovinocultura"]
        )
        assert "perfil_bovinocultura" in df.columns
        categorias = set(df["perfil_bovinocultura"].unique())
        assert categorias.issubset(
            {"sem_bovinocultura", "baixa_bovinocultura",
             "media_bovinocultura", "alta_bovinocultura"}
        )

    def test_alta_para_topo(self, df_com_atividades: pd.DataFrame) -> None:
        df = features.derive_perfis_agropecuarios(
            df_com_atividades, atividades=["bovinocultura"], quantis=(0.33, 0.66)
        )
        # Valor máximo (100000) deve ser 'alta'
        max_row = df.loc[df["bovinocultura"].idxmax()]
        assert max_row["perfil_bovinocultura"] == "alta_bovinocultura"

    def test_poucos_dados_fallback_binario(self) -> None:
        """Menos de 3 municípios com dados > 0: fallback presente/sem."""
        df_pequeno = pd.DataFrame(
            {
                "id_municipio": [1, 2],
                "bovinocultura": [0, 100],
            }
        )
        df = features.derive_perfis_agropecuarios(
            df_pequeno, atividades=["bovinocultura"]
        )
        assert set(df["perfil_bovinocultura"].unique()).issubset(
            {"sem_bovinocultura", "presente_bovinocultura"}
        )

    def test_atividade_ausente_e_ignorada(self) -> None:
        df = pd.DataFrame({"id_municipio": [1, 2, 3], "bovinocultura": [0, 100, 500]})
        result = features.derive_perfis_agropecuarios(
            df, atividades=["bovinocultura", "atividade_inexistente"]
        )
        assert "perfil_bovinocultura" in result.columns
        assert "perfil_atividade_inexistente" not in result.columns


# =============================================================================
# derive_especializacao
# =============================================================================
class TestDeriveEspecializacao:
    def test_atividade_com_maior_percentil_vence(self) -> None:
        """Município com percentil 1.0 em bovinos e 0.5 em suínos: bovinocultura."""
        df = pd.DataFrame(
            {
                "id_municipio": [1, 2, 3, 4],
                "bovinocultura": [1, 2, 3, 100],  # município 4 tem percentil 1.0
                "suinocultura": [50, 50, 50, 40],  # município 4 tem percentil baixo
            }
        )
        result = features.derive_especializacao(
            df, atividades_principais=["bovinocultura", "suinocultura"]
        )
        assert result.loc[3, "especializacao"] == "especializado_em_bovinocultura"

    def test_sem_producao_vira_sem_producao_pecuaria(self) -> None:
        df = pd.DataFrame(
            {
                "id_municipio": [1, 2],
                "bovinocultura": [100, 0],
                "suinocultura": [50, 0],
            }
        )
        result = features.derive_especializacao(
            df, atividades_principais=["bovinocultura", "suinocultura"]
        )
        assert result.loc[1, "especializacao"] == "sem_producao_pecuaria"

    def test_todas_atividades_ausentes_marca_sem_producao(self) -> None:
        df = pd.DataFrame({"id_municipio": [1, 2], "outra_coluna": [10, 20]})
        result = features.derive_especializacao(
            df, atividades_principais=["bovinocultura"]
        )
        assert (result["especializacao"] == "sem_producao_pecuaria").all()

    def test_zero_em_atividade_nao_causa_especializacao_falsa(self) -> None:
        """Regressão: valor 0 não deve competir por especialização.

        Antes da correção, um município com bovinocultura=100 e
        suinocultura=0 poderia ser rotulado 'especializado_em_suinocultura'
        porque o rank(pct) dos zeros dava percentil ~0.5. Agora zeros
        viram NaN antes do rank.
        """
        df = pd.DataFrame(
            {
                "id_municipio": [1, 2, 3, 4, 5],
                "bovinocultura": [100, 200, 300, 400, 500],
                "suinocultura": [0, 0, 0, 0, 1000],  # só o município 5 tem
            }
        )
        result = features.derive_especializacao(
            df, atividades_principais=["bovinocultura", "suinocultura"]
        )
        # Município 1-4: só têm bovinocultura, devem ser especializados nela
        for idx in [0, 1, 2, 3]:
            assert result.loc[idx, "especializacao"] == "especializado_em_bovinocultura", (
                f"Município {idx+1}: {result.loc[idx, 'especializacao']}"
            )
        # Município 5: tem ambos; suinocultura ganha (percentil 1.0 vs bov percentil 1.0)
        # (empate, idxmax pega o primeiro; verifica só que não é sem_producao)
        assert result.loc[4, "especializacao"] != "sem_producao_pecuaria"


# =============================================================================
# derive_diversidade
# =============================================================================
class TestDeriveDiversidade:
    def test_conta_atividades_positivas(self) -> None:
        df = pd.DataFrame(
            {
                "id_municipio": [1, 2, 3],
                "bovinocultura": [100, 0, 200],
                "suinocultura": [0, 50, 100],
                "avicultura": [1000, 0, 0],
            }
        )
        result = features.derive_diversidade(
            df, atividades=["bovinocultura", "suinocultura", "avicultura"]
        )
        assert result.loc[0, "n_atividades"] == 2  # bov + avi
        assert result.loc[1, "n_atividades"] == 1  # só suin
        assert result.loc[2, "n_atividades"] == 2  # bov + suin

    def test_atividades_presentes_string(self) -> None:
        df = pd.DataFrame(
            {
                "id_municipio": [1],
                "bovinocultura": [100],
                "suinocultura": [0],
                "avicultura": [1000],
            }
        )
        result = features.derive_diversidade(
            df, atividades=["bovinocultura", "suinocultura", "avicultura"]
        )
        s = result.loc[0, "atividades_presentes"]
        assert "bovinocultura" in s
        assert "avicultura" in s
        assert "suinocultura" not in s


# =============================================================================
# build_tags
# =============================================================================
class TestBuildTags:
    @pytest.fixture
    def df_pronto_para_tags(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> pd.DataFrame:
        df_clean = features.clean_ppm(ppm_raw_minimo)
        df_wide = features.pivot_ppm_wide(df_clean)
        df_merged = features.merge_com_localidades(df_wide, localidades_minimo)
        df_perfis = features.derive_perfis_agropecuarios(df_merged)
        df_esp = features.derive_especializacao(df_perfis)
        return features.derive_diversidade(df_esp)

    def test_coluna_tags_criada(self, df_pronto_para_tags: pd.DataFrame) -> None:
        result = features.build_tags(df_pronto_para_tags)
        assert "tags" in result.columns

    def test_tags_nao_vazias(self, df_pronto_para_tags: pd.DataFrame) -> None:
        result = features.build_tags(df_pronto_para_tags)
        assert (result["tags"].str.len() > 0).all()

    def test_tags_contem_contexto_geografico(
        self, df_pronto_para_tags: pd.DataFrame
    ) -> None:
        """Cambuquira deve ter 'sudeste', 'mg', 'sul_sudoeste_de_minas'."""
        result = features.build_tags(df_pronto_para_tags)
        cambu_tags = result[result["id_municipio"] == 3111606]["tags"].iloc[0]
        assert "sudeste" in cambu_tags
        assert "mg" in cambu_tags
        assert "sul_sudoeste_de_minas" in cambu_tags

    def test_tags_contem_perfis(self, df_pronto_para_tags: pd.DataFrame) -> None:
        result = features.build_tags(df_pronto_para_tags)
        cambu_tags = result[result["id_municipio"] == 3111606]["tags"].iloc[0]
        # Deve ter algum token de perfil (bovinocultura, suinocultura, ...)
        assert (
            "bovinocultura" in cambu_tags
            or "sem_bovinocultura" in cambu_tags
        )

    def test_tags_contem_especializacao(
        self, df_pronto_para_tags: pd.DataFrame
    ) -> None:
        result = features.build_tags(df_pronto_para_tags)
        sp_tags = result[result["id_municipio"] == 3550308]["tags"].iloc[0]
        assert "sem_producao_pecuaria" in sp_tags

    def test_tokens_snake_case_no_contexto(
        self, df_pronto_para_tags: pd.DataFrame
    ) -> None:
        """Nomes com espaço devem virar tokens únicos com underscore."""
        result = features.build_tags(df_pronto_para_tags)
        # 'Metropolitana de São Paulo' → 'metropolitana_de_são_paulo'
        sp_tags = result[result["id_municipio"] == 3550308]["tags"].iloc[0]
        assert "metropolitana_de_são_paulo" in sp_tags

    def test_multiplos_espacos_colapsados(
        self, df_pronto_para_tags: pd.DataFrame
    ) -> None:
        result = features.build_tags(df_pronto_para_tags)
        # Nenhuma tag deve ter dois espaços em sequência
        assert not (result["tags"].str.contains(r"\s\s")).any()


# =============================================================================
# Pipeline end-to-end
# =============================================================================
class TestBuildFeaturesDataset:
    def test_pipeline_end_to_end(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        df = features.build_features_dataset(
            df_ppm_raw=ppm_raw_minimo, df_localidades=localidades_minimo
        )
        assert len(df) == 4  # 4 municípios das localidades
        assert "tags" in df.columns
        assert "especializacao" in df.columns
        assert "n_atividades" in df.columns
        assert "perfil_bovinocultura" in df.columns

    def test_pipeline_todas_localidades_no_output(
        self,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        df = features.build_features_dataset(
            df_ppm_raw=ppm_raw_minimo, df_localidades=localidades_minimo
        )
        assert set(df["id_municipio"]) == {
            1200013,
            3111606,
            3550308,
            1302603,
        }


# =============================================================================
# save/load
# =============================================================================
class TestSaveLoad:
    def test_save_e_load_roundtrip(
        self,
        isolated_data_dirs: Path,
        ppm_raw_minimo: pd.DataFrame,
        localidades_minimo: pd.DataFrame,
    ) -> None:
        df = features.build_features_dataset(
            df_ppm_raw=ppm_raw_minimo, df_localidades=localidades_minimo
        )
        path = features.save_features_dataset(df)
        assert path.exists()

        df_carregado = features.load_features_dataset()
        assert len(df_carregado) == len(df)
        assert set(df_carregado.columns) == set(df.columns)

    def test_load_sem_arquivo_levanta_file_not_found(
        self, isolated_data_dirs: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            features.load_features_dataset()


# =============================================================================
# CLI
# =============================================================================
class TestParserFeatures:
    def test_parser_sem_argumentos(self) -> None:
        parser = features._build_parser()
        args = parser.parse_args([])
        assert args.ano is None

    def test_parser_com_ano(self) -> None:
        parser = features._build_parser()
        args = parser.parse_args(["--ano", "2023"])
        assert args.ano == 2023
