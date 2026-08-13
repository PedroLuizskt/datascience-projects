"""Testes do módulo `rec_agro_br.dataset`.

Organização
-----------
- **Unit tests** (default): validam parsing, tipagem, cache e tratamento de
  erro. Não fazem chamada de rede — usam fixtures sintéticas e um cliente
  SIDRA falso injetado. Rodam em milissegundos.
- **Integration tests** (`@pytest.mark.network`): fazem chamada real às
  APIs do IBGE. Por default são **excluídos** da execução (ver marker no
  pyproject.toml). Para rodá-los explicitamente:

      pytest tests/test_dataset.py -m network -v

Se você não tem internet ou o IBGE está fora do ar, os unit tests continuam
passando normalmente.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rec_agro_br import config, dataset


# =============================================================================
# _flatten_localidade — parsing de município aninhado em plano
# =============================================================================
class TestFlattenLocalidade:
    def test_municipio_completo_extrai_12_chaves(
        self, loc_municipio_completo: dict[str, Any]
    ) -> None:
        flat = dataset._flatten_localidade(loc_municipio_completo)
        chaves_esperadas = {
            "id_municipio",
            "nome_municipio",
            "id_microrregiao",
            "nome_microrregiao",
            "id_mesorregiao",
            "nome_mesorregiao",
            "id_uf",
            "sigla_uf",
            "nome_uf",
            "id_regiao",
            "sigla_regiao",
            "nome_regiao",
        }
        assert set(flat.keys()) == chaves_esperadas

    def test_municipio_completo_valores(
        self, loc_municipio_completo: dict[str, Any]
    ) -> None:
        flat = dataset._flatten_localidade(loc_municipio_completo)
        assert flat["id_municipio"] == 3550308
        assert flat["nome_municipio"] == "São Paulo"
        assert flat["sigla_uf"] == "SP"
        assert flat["nome_regiao"] == "Sudeste"
        assert flat["sigla_regiao"] == "SE"

    def test_municipio_cambuquira_valores(
        self, loc_municipio_cambuquira: dict[str, Any]
    ) -> None:
        flat = dataset._flatten_localidade(loc_municipio_cambuquira)
        assert flat["id_municipio"] == 3111606
        assert flat["nome_municipio"] == "Cambuquira"
        assert flat["sigla_uf"] == "MG"
        assert flat["nome_mesorregiao"] == "Sul/Sudoeste de Minas"

    def test_municipio_sem_hierarquia_nao_quebra(
        self, loc_municipio_sem_hierarquia: dict[str, Any]
    ) -> None:
        """Município degenerado retorna dict com Nones nos campos ausentes."""
        flat = dataset._flatten_localidade(loc_municipio_sem_hierarquia)
        assert flat["id_municipio"] == 9999999
        assert flat["nome_municipio"] == "Município Fantasma"
        assert flat["sigla_uf"] is None
        assert flat["nome_regiao"] is None


# =============================================================================
# _localidades_to_dataframe — montagem do DataFrame achatado
# =============================================================================
class TestLocalidadesToDataFrame:
    def test_lista_ampla_gera_dataframe_com_shape_correto(
        self, loc_lista_ampla: list[dict[str, Any]]
    ) -> None:
        df = dataset._localidades_to_dataframe(loc_lista_ampla)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 12)

    def test_lista_vazia_levanta_value_error(self) -> None:
        with pytest.raises(ValueError, match="vazia"):
            dataset._localidades_to_dataframe([])

    def test_colunas_de_id_sao_int64_nullable(
        self, loc_lista_ampla: list[dict[str, Any]]
    ) -> None:
        df = dataset._localidades_to_dataframe(loc_lista_ampla)
        for col in [
            "id_municipio",
            "id_microrregiao",
            "id_mesorregiao",
            "id_uf",
            "id_regiao",
        ]:
            assert str(df[col].dtype) == "Int64", f"{col} não é Int64"

    def test_colunas_de_texto_sao_string(
        self, loc_lista_ampla: list[dict[str, Any]]
    ) -> None:
        df = dataset._localidades_to_dataframe(loc_lista_ampla)
        for col in [
            "nome_municipio",
            "nome_microrregiao",
            "nome_uf",
            "sigla_regiao",
        ]:
            assert str(df[col].dtype) == "string", f"{col} não é string"

    def test_ids_de_regiao_dentro_do_esperado(
        self, loc_lista_ampla: list[dict[str, Any]]
    ) -> None:
        """Regiões brasileiras têm ID de 1 a 5."""
        df = dataset._localidades_to_dataframe(loc_lista_ampla)
        assert set(df["id_regiao"].dropna().unique()).issubset({1, 2, 3, 4, 5})


# =============================================================================
# download_localidades — orquestração + cache
# =============================================================================
class _FakeResponse:
    """Response falso para injetar em requests.Session."""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"Status {self.status_code}")


class _FakeSession:
    """Session falsa que retorna um payload pré-configurado."""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self._status = status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return _FakeResponse(self._payload, self._status)


class TestDownloadLocalidades:
    def test_download_com_session_falsa_grava_dois_arquivos(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session = _FakeSession(loc_lista_ampla)
        df = dataset.download_localidades(session=session)

        assert dataset.get_localidades_raw_path().exists()
        assert dataset.get_localidades_interim_path().exists()
        assert df.shape == (4, 12)

    def test_download_faz_uma_chamada_HTTP(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session)
        assert len(session.calls) == 1
        url, _ = session.calls[0]
        assert "localidades/municipios" in url

    def test_cache_evita_segunda_chamada_HTTP(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session1 = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session1)
        assert len(session1.calls) == 1

        session2 = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session2)
        assert len(session2.calls) == 0, "Segunda chamada não deveria tocar a rede"

    def test_force_true_refaz_download(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session1 = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session1)
        assert len(session1.calls) == 1

        session2 = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session2, force=True)
        assert len(session2.calls) == 1, "Com force=True deveria ter refeito"

    def test_json_bruto_persiste_estrutura_original(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session = _FakeSession(loc_lista_ampla)
        dataset.download_localidades(session=session)

        raw_path = dataset.get_localidades_raw_path()
        raw_content = json.loads(raw_path.read_text(encoding="utf-8"))
        assert isinstance(raw_content, list)
        assert len(raw_content) == 4
        # A estrutura aninhada deve ser preservada
        assert "microrregiao" in raw_content[0]


class TestLoadLocalidades:
    def test_load_sem_arquivo_levanta_file_not_found(
        self,
        isolated_data_dirs: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            dataset.load_localidades()

    def test_load_apos_download_le_do_disco(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
    ) -> None:
        session = _FakeSession(loc_lista_ampla)
        df_baixado = dataset.download_localidades(session=session)
        df_lido = dataset.load_localidades()
        pd.testing.assert_frame_equal(df_baixado, df_lido)


# =============================================================================
# _resolver_periodo_ppm — lógica de resolução de ano
# =============================================================================
class TestResolverPeriodoPPM:
    def test_ano_none_devolve_sentinela(self) -> None:
        periodo, cache_key = dataset._resolver_periodo_ppm(None)
        assert periodo == dataset.PPM_ULTIMO_DISPONIVEL
        assert cache_key == dataset.PPM_ULTIMO_DISPONIVEL

    def test_ano_inteiro_devolve_string_e_int(self) -> None:
        periodo, cache_key = dataset._resolver_periodo_ppm(2023)
        assert periodo == "2023"
        assert cache_key == 2023


# =============================================================================
# _dividir_em_chunks — chunking por tamanho fixo (substituiu _agrupar_por_uf
# como estratégia principal de fatiamento, ver docstring do dataset.py)
# =============================================================================
class TestDividirEmChunks:
    def test_lista_menor_que_chunk_gera_um_chunk(self) -> None:
        chunks = dataset._dividir_em_chunks(["a", "b", "c"], chunk_size=10)
        assert chunks == [["a", "b", "c"]]

    def test_lista_exata_gera_n_chunks(self) -> None:
        codigos = [str(i) for i in range(10)]
        chunks = dataset._dividir_em_chunks(codigos, chunk_size=5)
        assert len(chunks) == 2
        assert chunks[0] == ["0", "1", "2", "3", "4"]
        assert chunks[1] == ["5", "6", "7", "8", "9"]

    def test_lista_com_resto_gera_ultimo_chunk_menor(self) -> None:
        codigos = [str(i) for i in range(11)]
        chunks = dataset._dividir_em_chunks(codigos, chunk_size=5)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 1

    def test_5570_codigos_com_chunk_500_gera_12_chunks(self) -> None:
        """Cenário real: Brasil com chunk_size default (500)."""
        codigos = [str(i) for i in range(5570)]
        chunks = dataset._dividir_em_chunks(codigos, chunk_size=500)
        assert len(chunks) == 12  # 11 × 500 + 70

    def test_chunk_size_zero_ou_negativo_levanta_erro(self) -> None:
        with pytest.raises(ValueError, match="positivo"):
            dataset._dividir_em_chunks(["a"], chunk_size=0)


# =============================================================================
# _validar_cobertura_ppm — check crítico contra falhas silenciosas do SIDRA
# =============================================================================
class TestValidarCoberturaPPM:
    def test_cobertura_completa_nao_levanta_warning(self, caplog) -> None:
        df = pd.DataFrame({"D1C": ["1", "2", "3"]})
        dataset._validar_cobertura_ppm(df, {"1", "2", "3"})
        # Deve logar apenas INFO, nunca WARNING
        avisos = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not avisos

    def test_perda_moderada_gera_warning(self, caplog) -> None:
        import logging as _logging

        caplog.set_level(_logging.WARNING)
        df = pd.DataFrame({"D1C": [str(i) for i in range(90)]})
        solicitados = {str(i) for i in range(100)}  # 10% de perda
        dataset._validar_cobertura_ppm(df, solicitados, limite_aviso=0.02)
        assert any("Cobertura PPM" in r.message for r in caplog.records)

    def test_perda_catastrofica_levanta_runtime_error(self) -> None:
        """Regressão do bug MG/SP: perda >50% deve derrubar o pipeline."""
        df = pd.DataFrame({"D1C": [str(i) for i in range(30)]})
        solicitados = {str(i) for i in range(100)}  # 70% de perda
        with pytest.raises(RuntimeError, match="catastroficamente"):
            dataset._validar_cobertura_ppm(df, solicitados)

    def test_ausencia_de_D1C_apenas_avisa(self, caplog) -> None:
        import logging as _logging

        caplog.set_level(_logging.WARNING)
        df = pd.DataFrame({"outra_coluna": ["a"]})
        dataset._validar_cobertura_ppm(df, {"1", "2"})
        assert any("D1C ausente" in r.message for r in caplog.records)


# =============================================================================
# _agrupar_municipios_por_uf — mantido como utilitário mas não mais usado
# no pipeline principal de download
# =============================================================================
class TestAgruparMunicipiosPorUF:
    def test_agrupamento_gera_dicionario_por_uf(
        self, loc_lista_ampla: list[dict[str, Any]]
    ) -> None:
        df = dataset._localidades_to_dataframe(loc_lista_ampla)
        agrupado = dataset._agrupar_municipios_por_uf(df)
        assert set(agrupado.keys()) == {"SP", "MG", "RS", "AM"}

    def test_uf_ausente_e_descartada(self) -> None:
        df = pd.DataFrame(
            {
                "sigla_uf": ["SP", None, "MG"],
                "id_municipio": [3550308, 9999999, 3111606],
            }
        )
        agrupado = dataset._agrupar_municipios_por_uf(df)
        assert set(agrupado.keys()) == {"SP", "MG"}


# =============================================================================
# download_ppm_efetivo_rebanhos — chunking por tamanho fixo
# =============================================================================
class TestDownloadPPM:
    def test_download_com_cliente_falso_grava_parquet(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_client: "Any",
    ) -> None:
        df = dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        raw_path = dataset.get_ppm_raw_path(2023)
        assert raw_path.exists()

    def test_parametros_passados_ao_sidra_estao_corretos(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_client: "Any",
    ) -> None:
        dataset.download_ppm_efetivo_rebanhos(
            ano=2022, sidra_client=fake_sidra_client
        )
        call = fake_sidra_client.last_call
        assert call is not None
        assert call["table_code"] == config.PPM_TABLE_CODE
        assert call["territorial_level"] == config.SIDRA_TERRITORIAL_LEVEL_MUNICIPIO
        assert call["ibge_territorial_code"] != "all"
        assert call["variable"] == config.PPM_VARIABLE_CODE
        assert call["period"] == "2022"
        assert call["header"] == "n"
        assert call["format"] == "pandas"

    def test_ano_none_usa_sentinela_no_sidra(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_client: "Any",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(config, "PPM_ANO", None)
        dataset.download_ppm_efetivo_rebanhos(sidra_client=fake_sidra_client)
        assert fake_sidra_client.last_call["period"] == dataset.PPM_ULTIMO_DISPONIVEL

    def test_chunking_por_tamanho_gera_multiplas_chamadas(
        self,
        localidades_gravadas_multi_uf: tuple[pd.DataFrame, list],
        fake_sidra_client: "Any",
    ) -> None:
        """Com chunk_size=2 e 8 municípios totais, esperam-se 4 chamadas SIDRA.

        localidades_gravadas_multi_uf tem 4 UFs × 2 municípios = 8 municípios.
        """
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=2
        )
        assert fake_sidra_client.call_count == 4

    def test_chunk_size_maior_que_dataset_gera_uma_chamada(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_client: "Any",
    ) -> None:
        """Com chunk_size folgado, todos os municípios cabem em 1 chamada."""
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=500
        )
        # localidades_gravadas_uma_uf tem 3 municípios de SP; 500 > 3 → 1 chunk
        assert fake_sidra_client.call_count == 1

    def test_cache_evita_novo_batch_completo(
        self,
        localidades_gravadas_multi_uf: tuple[pd.DataFrame, list],
        fake_sidra_client: "Any",
    ) -> None:
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=2
        )
        primeira_execucao = fake_sidra_client.call_count
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=2
        )
        assert fake_sidra_client.call_count == primeira_execucao

    def test_force_true_refaz_batch_completo(
        self,
        localidades_gravadas_multi_uf: tuple[pd.DataFrame, list],
        fake_sidra_client: "Any",
    ) -> None:
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=2
        )
        primeira_execucao = fake_sidra_client.call_count
        dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client, chunk_size=2, force=True
        )
        assert fake_sidra_client.call_count == 2 * primeira_execucao

    def test_sidra_vazio_em_todos_chunks_levanta_runtime_error(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_empty: "Any",
    ) -> None:
        with pytest.raises(RuntimeError, match="Nenhum dos"):
            dataset.download_ppm_efetivo_rebanhos(
                ano=2023, sidra_client=fake_sidra_empty
            )

    def test_localidades_ausentes_dispara_download_automatico(
        self,
        isolated_data_dirs: Path,
        loc_lista_ampla: list[dict[str, Any]],
        fake_sidra_client: "Any",
    ) -> None:
        session = _FakeSession(loc_lista_ampla)
        import rec_agro_br.dataset as ds_module

        original_build = ds_module._build_session
        ds_module._build_session = lambda: session  # type: ignore
        try:
            assert not dataset.get_localidades_interim_path().exists()
            dataset.download_ppm_efetivo_rebanhos(
                ano=2023, sidra_client=fake_sidra_client
            )
            assert dataset.get_localidades_interim_path().exists()
            assert len(session.calls) == 1
        finally:
            ds_module._build_session = original_build

    def test_falha_em_chunk_isolado_nao_derruba_pipeline(
        self,
        localidades_gravadas_multi_uf: tuple[pd.DataFrame, list],
    ) -> None:
        """Simula: primeiro chunk falha (JSONDecodeError), demais passam.

        Regressão do bug MG/SP: falhas de chunk devem ser toleradas
        individualmente, desde que a maioria dos chunks tenha sucesso.
        """
        import json

        from tests.conftest import FakeSidraClient

        # Estende FakeSidraClient para falhar na primeira chamada
        class FakeParcial(FakeSidraClient):
            def get_table(self, **kwargs):
                if self.call_count == 0:
                    self.call_count += 1  # conta a tentativa falha
                    raise json.JSONDecodeError("Expecting value", "", 0)
                return super().get_table(**kwargs)

        client = FakeParcial()
        # 8 municípios, chunk_size=4 → 2 chunks. Primeiro falha, 1 sucede.
        # 1 chunk × 4 municípios × 2 rebanhos = 8 linhas
        # 4 municípios / 8 solicitados = 50% cobertura → passa (>= 50% limiar)
        df = dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=client, chunk_size=4
        )
        assert client.call_count == 2
        assert len(df) == 8  # 1 chunk × 4 municípios × 2 rebanhos


class TestLoadPPM:
    def test_load_sem_arquivo_levanta_file_not_found(
        self, isolated_data_dirs: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            dataset.load_ppm_efetivo_rebanhos(ano=2023)

    def test_load_apos_download_le_igual(
        self,
        localidades_gravadas_uma_uf: pd.DataFrame,
        fake_sidra_client: "Any",
    ) -> None:
        df_baixado = dataset.download_ppm_efetivo_rebanhos(
            ano=2023, sidra_client=fake_sidra_client
        )
        df_lido = dataset.load_ppm_efetivo_rebanhos(ano=2023)
        pd.testing.assert_frame_equal(df_baixado, df_lido)


# =============================================================================
# CLI (parser)
# =============================================================================
class TestParser:
    def test_parser_aceita_localidades(self) -> None:
        parser = dataset._build_parser()
        args = parser.parse_args(["localidades"])
        assert args.command == "localidades"
        assert args.force is False

    def test_parser_aceita_ppm_com_ano(self) -> None:
        parser = dataset._build_parser()
        args = parser.parse_args(["ppm", "--ano", "2023"])
        assert args.command == "ppm"
        assert args.ano == 2023

    def test_parser_aceita_all_com_force(self) -> None:
        parser = dataset._build_parser()
        args = parser.parse_args(["all", "--force"])
        assert args.command == "all"
        assert args.force is True

    def test_parser_falta_subcomando_erro(self) -> None:
        parser = dataset._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# =============================================================================
# Integration tests (marcados com @pytest.mark.network)
# =============================================================================
# Estes testes só rodam quando explicitamente pedidos com:
#     pytest -m network
#
# Eles fazem chamadas HTTP reais ao IBGE. Podem levar 30-60s cada. Não são
# incluídos por default para manter a suíte rápida e independente de rede.

@pytest.mark.network
class TestIntegracaoRedeReal:
    """Testes que efetivamente contatam a API do IBGE."""

    def test_baixa_localidades_reais(self, isolated_data_dirs: Path) -> None:
        df = dataset.download_localidades()
        # O Brasil tem 5570 municípios (dado consolidado, IBGE 2022+)
        assert 5500 <= len(df) <= 5600, f"Esperado ~5570, obtido {len(df)}"
        # Todas as 27 UFs devem estar presentes
        assert df["sigla_uf"].nunique() == 27

    @pytest.mark.slow
    def test_baixa_ppm_real_ultimo_ano(self, isolated_data_dirs: Path) -> None:
        df = dataset.download_ppm_efetivo_rebanhos()
        assert not df.empty
        assert len(df) > 10000

    @pytest.mark.slow
    def test_regressao_mg_sp_cobertura(self, isolated_data_dirs: Path) -> None:
        """Regressão do bug de chunking: MG e SP DEVEM ter dados.

        Bug detectado após Fase 1.D: chunking por UF gerava URLs muito
        grandes para MG (853 municípios) e SP (645 municípios), causando
        falha silenciosa do SIDRA. Este teste garante que a correção
        (chunking por tamanho fixo de 500) resolveu o problema.
        """
        df = dataset.download_ppm_efetivo_rebanhos()
        # Código IBGE de MG começa com "31", SP com "35"
        mg_municipios = df[df["D1C"].astype(str).str.startswith("31")][
            "D1C"
        ].nunique()
        sp_municipios = df[df["D1C"].astype(str).str.startswith("35")][
            "D1C"
        ].nunique()
        # MG tem 853 municípios, SP tem 645. Aceita perda de até 5%.
        assert mg_municipios >= 800, f"Cobertura MG: só {mg_municipios} municípios"
        assert sp_municipios >= 610, f"Cobertura SP: só {sp_municipios} municípios"
