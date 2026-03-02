# -*- coding: utf-8 -*-
"""
Data Service
------------
High-level data loading and management.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from core.exceptions import ValidationError, ServiceError
from core.config import get_config
from parsers import ParserFactory, FirewallParser
from utils.io import load_parquet


class DataService:
    """
    Service for loading and managing data.
    Provides high-level interface for data operations.
    """

    def __init__(self):
        config = get_config()
        self.data_dir = config.data_dir
        self._loaded_data: Dict[str, pd.DataFrame] = {}

    def load_raw_logs(
        self,
        source: str,
        parser_type: str = 'firewall',
        **parser_kwargs
    ) -> pd.DataFrame:
        """
        Load and parse raw log file.

        Args:
            source: File path
            parser_type: Parser type ('firewall', 'csv', 'syslog')
            **parser_kwargs: Parser arguments

        Returns:
            Parsed DataFrame
        """
        parser = ParserFactory.create(parser_type, **parser_kwargs)
        df = parser.parse(source)

        # Validate
        is_valid, errors = parser.validate(df)
        if not is_valid:
            raise ValidationError(
                f"Data validation failed: {errors}",
                errors=errors
            )

        self._loaded_data['raw_logs'] = df
        return df

    def load_parquet_file(
        self,
        source: str,
        drop_cols: Optional[List[str]] = None,
        date_filter: Optional[tuple] = None,
        date_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Load a pre-processed Parquet file (e.g. produced by 00_transform_raw_data).

        Args:
            source: Path to the ``.parquet`` file.
            drop_cols: Optional list of columns to drop after loading (e.g. ``["fw"]``).
            date_filter: Optional ``(date_from, date_to)`` tuple of strings
                ``"YYYY-MM-DD"`` to keep only rows in that range.
            date_col: Name of the datetime column used for filtering
                (default ``"timestamp"``).

        Returns:
            Parsed DataFrame.
        """
        path = Path(source)
        if not path.exists():
            raise ServiceError(f"Parquet file not found: {source}", "data_service")

        df = load_parquet(path)

        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        if date_filter is not None:
            date_from, date_to = date_filter
            if date_col in df.columns:
                mask = (df[date_col] >= date_from) & (df[date_col] <= date_to)
                df = df.loc[mask].reset_index(drop=True)

        self._loaded_data['raw_logs'] = df
        return df

    def load_labeled_data(
        self,
        source: str,
        index_col: int = 0
    ) -> pd.DataFrame:
        """
        Load labeled dataset (Excel/CSV).

        Args:
            source: File path
            index_col: Index column

        Returns:
            DataFrame with labels
        """
        path = Path(source)

        if not path.exists():
            raise ServiceError(f"File not found: {source}", "data_service")

        if path.suffix == '.xlsx':
            df = pd.read_excel(source, index_col=index_col)
        elif path.suffix == '.csv':
            df = pd.read_csv(source, index_col=index_col)
        else:
            raise ServiceError(
                f"Unsupported file type: {path.suffix}",
                "data_service"
            )

        self._loaded_data['labeled'] = df
        return df

    def load_features(
        self,
        source: str,
        index_col: int = 0
    ) -> pd.DataFrame:
        """
        Load pre-computed features.

        Args:
            source: File path
            index_col: Index column

        Returns:
            Feature DataFrame
        """
        return self.load_labeled_data(source, index_col)

    def get_data(self, key: str) -> Optional[pd.DataFrame]:
        """Get loaded data by key."""
        return self._loaded_data.get(key)

    def save_data(
        self,
        df: pd.DataFrame,
        destination: str,
        format: str = 'csv'
    ) -> Path:
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            destination: File path
            format: Output format ('csv', 'xlsx', 'parquet')

        Returns:
            Path to saved file
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            df.to_csv(path)
        elif format == 'xlsx':
            df.to_excel(path)
        elif format == 'parquet':
            df.to_parquet(path)
        else:
            raise ServiceError(
                f"Unsupported format: {format}",
                "data_service"
            )

        return path

    def validate_features(
        self,
        df: pd.DataFrame,
        required_cols: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate feature DataFrame.

        Args:
            df: Feature DataFrame
            required_cols: Required column names

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if df.empty:
            errors.append("DataFrame is empty")

        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                errors.append(f"Missing required columns: {missing}")

        # Check for NaN
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            errors.append(f"Columns with NaN values: {nan_cols}")

        return len(errors) == 0, errors

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data summary statistics.

        Args:
            df: DataFrame to summarize

        Returns:
            Dict with summary statistics
        """
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict()
        }

    def clear(self) -> None:
        """Clear all loaded data."""
        self._loaded_data.clear()
