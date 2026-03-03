# -*- coding: utf-8 -*-
"""
Configuration Management
------------------------
Centralized configuration for the toolkit.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional
from pathlib import Path
import yaml


@dataclass
class ParserConfig:
    """Configuration for log parsers."""

    default_separator: str = ","
    default_encoding: str = "utf-8"
    firewall_columns: List[str] = field(
        default_factory=lambda: [
            "ipsrc",
            "ipdst",
            "portdst",
            "proto",
            "action",
            "date",
            "regle",
        ]
    )


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    admin_ports: Set[int] = field(default_factory=lambda: {21, 22, 3389, 3306})
    port_threshold: int = 1024
    default_group_by: str = "ipsrc"


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    random_state: int = 42
    default_cv_folds: int = 5
    positive_label: str = "positive"
    target_column: str = "risk"


@dataclass
class AppConfig:
    """Configuration for Streamlit app."""

    title: str = "Intrusion Detection Dashboard"
    icon: str = "🛡️"
    layout: str = "wide"


@dataclass
class Config:
    """Main configuration container."""

    parser: ParserConfig = field(default_factory=ParserConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    app: AppConfig = field(default_factory=AppConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("saved_models"))

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "parser" in data:
            config.parser = ParserConfig(**data["parser"])
        if "features" in data:
            feat_data = data["features"]
            if "admin_ports" in feat_data:
                feat_data["admin_ports"] = set(feat_data["admin_ports"])
            config.features = FeatureConfig(**feat_data)
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "app" in data:
            config.app = AppConfig(**data["app"])
        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])
        if "models_dir" in data:
            config.models_dir = Path(data["models_dir"])

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            "parser": {
                "default_separator": self.parser.default_separator,
                "default_encoding": self.parser.default_encoding,
                "firewall_columns": self.parser.firewall_columns,
            },
            "features": {
                "admin_ports": list(self.features.admin_ports),
                "port_threshold": self.features.port_threshold,
                "default_group_by": self.features.default_group_by,
            },
            "model": {
                "random_state": self.model.random_state,
                "default_cv_folds": self.model.default_cv_folds,
                "positive_label": self.model.positive_label,
                "target_column": self.model.target_column,
            },
            "app": {
                "title": self.app.title,
                "icon": self.app.icon,
                "layout": self.app.layout,
            },
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def load_config(path: str) -> Config:
    """Load and set global configuration from file."""
    config = Config.from_yaml(path)
    set_config(config)
    return config
