"""
Unit tests for Hydra-based configuration management module.

Tests configuration loading and config group composition.
"""

import pytest
import tempfile
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from glyph.config.config import (
    load_config,
    save_config,
    print_config,
    get_config_path,
)


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        cfg = load_config(overrides=[
            "seed=999",
            "device=cuda"
        ])

        assert cfg.seed == 999
        assert cfg.device == "cuda"

    def test_load_config_with_data_overrides(self):
        """Test loading config with data path overrides."""
        cfg = load_config(overrides=[
            "data.data_path=/test/path",
            "data.metadata=/test/metadata.csv",
            "data.input_cols=[x,y]",
            "data.outcome_cols=[z]"
        ])

        assert cfg.data.data_path == "/test/path"
        assert cfg.data.metadata == "/test/metadata.csv"
        assert cfg.data.input_cols == ["x", "y"]
        assert cfg.data.outcome_cols == ["z"]

    def test_config_is_dictconfig(self):
        """Test that loaded config is a DictConfig."""
        cfg = load_config(overrides=["seed=42"])
        assert isinstance(cfg, DictConfig)

    def test_config_supports_dot_notation(self):
        """Test that config supports dot notation access."""
        cfg = load_config(overrides=["seed=123"])
        assert cfg.seed == 123
        assert hasattr(cfg, 'sae')
        assert hasattr(cfg.sae, 'learning_rate')

    def test_config_supports_dict_access(self):
        """Test that config supports dict-style access."""
        cfg = load_config(overrides=["seed=456"])
        assert cfg['seed'] == 456
        assert cfg['sae']['learning_rate'] == 0.0003


class TestConfigGroups:
    """Test cases for config group selection."""

    def test_sae_default_group(self):
        """Test default SAE config group."""
        cfg = load_config()
        assert cfg.sae.top_ks == [5, 10, 20, 40, 80]
        assert cfg.sae.selection.r2_targets == [0.85, 0.90, 0.95]
        assert cfg.sae.selection.extra_sparsity_per_target == 1.0
        assert cfg.sae.matryoshka == [128, 512, 2048, 8192]
        assert cfg.sae.learning_rate == 0.0003

    def test_phewas_default_group(self):
        """Test default PheWAS config group."""
        cfg = load_config()
        assert cfg.phewas.min_activations == 25
        assert cfg.phewas.selection_criteria == "top_20"

    def test_autointerp_default_group(self):
        """Test default AutoInterp config group."""
        cfg = load_config()
        assert cfg.autointerp.llm.provider == "gemini_api"
        assert cfg.autointerp.llm.model == "gemini-flash-lite-latest"
        assert cfg.autointerp.llm.temperature == 0.0
        assert cfg.autointerp.llm.max_tokens == 1024
        assert cfg.autointerp.n_fit_samples == 40
        assert cfg.autointerp.n_test_samples == 40

    def test_metrics_default_group(self):
        """Test default metrics config group."""
        cfg = load_config()
        assert cfg.metrics.compute_auc is True
        assert cfg.metrics.compute_accuracy is True
        assert cfg.metrics.threshold == 0.5
        assert cfg.metrics.average == "binary"

    def test_logging_none_group(self):
        """Test none logging config group."""
        cfg = load_config(overrides=["logging=none"])
        assert cfg.logging.use_wandb is False

    def test_logging_wandb_group(self):
        """Test wandb logging config group."""
        cfg = load_config(overrides=["logging=wandb"])
        assert cfg.logging.use_wandb is True
        assert cfg.logging.wandb_project == "glyph"

    def test_visualization_default_group(self):
        cfg = load_config()
        assert cfg.visualization.auto_generate is False
        assert cfg.visualization.interactive is True
        assert cfg.visualization.static_jpg is False


class TestConfigOverrides:
    """Test cases for config overrides."""

    def test_override_top_level_settings(self):
        """Test overriding top-level settings."""
        cfg = load_config(overrides=[
            "seed=999",
            "device=cpu",
            "path_to_embeddings=/path/to/emb"
        ])
        assert cfg.seed == 999
        assert cfg.device == "cpu"
        assert cfg.path_to_embeddings == "/path/to/emb"

    def test_override_nested_sae_settings(self):
        """Test overriding nested SAE settings."""
        cfg = load_config(overrides=[
            "sae.learning_rate=0.001",
            "sae.max_steps=10000"
        ])
        assert cfg.sae.learning_rate == 0.001
        assert cfg.sae.max_steps == 10000

    def test_override_phewas_settings(self):
        """Test overriding PheWAS settings."""
        cfg = load_config(overrides=[
            "phewas.min_activations=50",
            "phewas.selection_criteria=top_10"
        ])
        assert cfg.phewas.min_activations == 50
        assert cfg.phewas.selection_criteria == "top_10"

    def test_override_metrics_settings(self):
        """Test overriding metrics settings."""
        cfg = load_config(overrides=[
            "metrics.threshold=0.7",
            "metrics.compute_auc=false"
        ])
        assert cfg.metrics.threshold == 0.7
        assert cfg.metrics.compute_auc is False


class TestSaveConfig:
    """Test cases for save_config function."""

    def test_save_config_creates_file(self):
        """Test that save_config creates output file."""
        cfg = load_config(overrides=["seed=123"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_config(cfg, tmpdir)

            config_path = Path(tmpdir) / "config.yaml"
            assert config_path.exists()

            # Load and verify contents
            loaded = OmegaConf.load(config_path)
            assert loaded.seed == 123

    def test_save_config_creates_directory(self):
        """Test that save_config creates output directory if it doesn't exist."""
        cfg = load_config(overrides=["seed=456"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "nested"
            save_config(cfg, str(output_dir))

            assert output_dir.exists()
            assert (output_dir / "config.yaml").exists()

    def test_save_config_custom_filename(self, tmp_path):
        cfg = load_config(overrides=["seed=789"])
        save_config(cfg, tmp_path, filename="my_config.yaml")
        assert (tmp_path / "my_config.yaml").exists()

    def test_save_config_preserves_all_groups(self):
        """Test that save_config preserves all config groups."""
        cfg = load_config(overrides=["seed=999"])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_config(cfg, tmpdir)

            loaded = OmegaConf.load(Path(tmpdir) / "config.yaml")
            assert hasattr(loaded, 'data')
            assert hasattr(loaded, 'sae')
            assert hasattr(loaded, 'phewas')
            assert hasattr(loaded, 'autointerp')
            assert hasattr(loaded, 'metrics')
            assert hasattr(loaded, 'logging')


class TestPrintConfig:
    """Test cases for print_config function."""

    def test_print_config_runs(self, capsys):
        """Test that print_config executes without error."""
        cfg = load_config(overrides=["seed=999"])
        print_config(cfg)

        captured = capsys.readouterr()
        assert "seed: 999" in captured.out


class TestGetConfigPath:
    """Test cases for get_config_path function."""

    def test_get_config_path_returns_path(self):
        """Test that get_config_path returns a Path object."""
        path = get_config_path()
        assert isinstance(path, Path)

    def test_get_config_path_points_to_config(self):
        """Test that config path points to the packaged conf directory."""
        path = get_config_path()
        assert path.name == "conf"
        assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
