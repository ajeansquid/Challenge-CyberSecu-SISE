# -*- coding: utf-8 -*-
"""I/O utility functions."""

import json
from pathlib import Path
from typing import Any, Union

import yaml


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Union[str, Path]) -> dict:
    """
    Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Dictionary to save
        path: Output path
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

try:
    import pandas as pd  # noqa: F401 (optional, only needed for parquet functions)

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


def save_parquet(
    df: Any,
    path: Union[str, Path],
    compression: str = "snappy",
    index: bool = False,
) -> Path:
    """
    Save a pandas DataFrame to a Parquet file.

    Args:
        df: DataFrame to persist.
        path: Destination file path (will be created if needed).
        compression: Parquet compression codec ('snappy', 'gzip', 'brotli', None).
        index: Whether to include the DataFrame index.

    Returns:
        Resolved Path of the written file.

    Raises:
        ImportError: If neither pyarrow nor fastparquet is installed.
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, compression=compression, index=index)
    return path


def load_parquet(
    path: Union[str, Path],
    columns: list = None,
) -> Any:
    """
    Load a Parquet file into a pandas DataFrame.

    Args:
        path: Path to the Parquet file.
        columns: Optional list of column names to load (projection push-down).

    Returns:
        DataFrame with the file's contents.
    """
    import pandas as pd  # local import to keep module lightweight

    kwargs = {}
    if columns:
        kwargs["columns"] = columns
    return pd.read_parquet(Path(path), **kwargs)
