# -*- coding: utf-8 -*-
"""
Generic CSV Parser
------------------
Flexible parser for CSV files with configurable options.
"""

import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple

from .base import BaseParser


class GenericCSVParser(BaseParser):
    """
    Generic CSV parser with extensive configuration options.
    """

    def __init__(
        self,
        columns: List[str] = None,
        separator: str = ",",
        encoding: str = "utf-8",
        skiprows: int = 0,
        date_columns: List[str] = None,
        date_format: str = None,
        dtype: Dict[str, type] = None,
        transformations: Dict[str, Callable] = None,
        na_values: List[str] = None
    ):
        super().__init__(separator=separator, encoding=encoding, columns=columns)
        self.skiprows = skiprows
        self.date_columns = date_columns or []
        self.date_format = date_format
        self.dtype = dtype or {}
        self.transformations = transformations or {}
        self.na_values = na_values or ["", "NA", "N/A", "-", "null", "NULL"]

    def _do_parse(self, source: str, encoding: str) -> pd.DataFrame:
        """Parse CSV with all configured options."""
        kwargs = {
            'sep': self.separator,
            'encoding': encoding,
            'skiprows': self.skiprows,
            'na_values': self.na_values,
            'on_bad_lines': 'warn'
        }

        if self._columns:
            kwargs['names'] = self._columns
            kwargs['header'] = None

        if self.dtype:
            kwargs['dtype'] = self.dtype

        return pd.read_csv(source, **kwargs)

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply date parsing and transformations."""
        # Parse date columns
        for col in self.date_columns:
            if col in df.columns:
                if self.date_format:
                    df[col] = pd.to_datetime(
                        df[col],
                        format=self.date_format,
                        errors='coerce'
                    )
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        # Apply custom transformations
        for col, func in self.transformations.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Basic validation."""
        errors = []

        if df.empty:
            errors.append("DataFrame is empty")

        if self._columns:
            missing = set(self._columns) - set(df.columns)
            if missing:
                errors.append(f"Missing columns: {missing}")

        return len(errors) == 0, errors
