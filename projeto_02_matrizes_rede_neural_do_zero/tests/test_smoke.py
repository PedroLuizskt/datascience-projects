"""Testes de sanidade básica: imports, versões, config."""

from __future__ import annotations


class TestImports:
    def test_import_pacote(self) -> None:
        import rna_matrizes  # noqa: F401

    def test_import_core(self) -> None:
        from rna_matrizes import core  # noqa: F401

    def test_import_config(self) -> None:
        from rna_matrizes import config  # noqa: F401

    def test_classe_principal_exportada_no_top_level(self) -> None:
        from rna_matrizes import RedeNeuralBinaria
        assert RedeNeuralBinaria is not None


class TestVersao:
    def test_versao_definida(self) -> None:
        import rna_matrizes
        assert rna_matrizes.__version__ is not None
        assert isinstance(rna_matrizes.__version__, str)
        assert len(rna_matrizes.__version__) > 0


class TestConfig:
    def test_diretorios_definidos(self) -> None:
        from rna_matrizes import config
        assert config.PROJECT_ROOT.exists()
        assert config.DATA_DIR.name == "data"

    def test_ensure_directories_cria_estrutura(self, tmp_path, monkeypatch) -> None:
        from rna_matrizes import config

        monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
        monkeypatch.setattr(config, "INTERIM_DATA_DIR", tmp_path / "interim")
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
        monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", tmp_path / "external")
        monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")

        config.ensure_directories()

        for name in ("raw", "interim", "processed", "external", "figures"):
            assert (tmp_path / name).exists()
            assert (tmp_path / name).is_dir()

    def test_defaults_hiperparametros(self) -> None:
        from rna_matrizes import config
        assert config.DEFAULT_TAXA_APRENDIZADO > 0
        assert config.DEFAULT_NUM_ITERACOES > 0
        assert config.DEFAULT_TOLERANCIA >= 0
        assert 0 < config.DEFAULT_LIMIAR_DECISAO < 1
