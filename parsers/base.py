# -*- coding: utf-8 -*-
"""
Base Parser
-----------
Abstract base implementation for parsers with common functionality.
"""

import pandas as pd
from typing import List, Tuple
from pathlib import Path

from core.interfaces import Parser
from core.exceptions import ParsingError
from core.config import get_config


class BaseParser(Parser):
    """
    Base parser with common functionality.
    Extend this class to implement specific parsers.
    """

    def __init__(
        self,
        separator: str = None,
        encoding: str = None,
        columns: List[str] = None
    ):
        config = get_config()
        self.separator = separator or config.parser.default_separator
        self.encoding = encoding or config.parser.default_encoding
        self._columns = columns or []

    @property
    def expected_columns(self) -> List[str]:
        return self._columns

    def parse(self, source: str) -> pd.DataFrame:
        """Parse file with automatic encoding detection."""
        path = Path(source)

        if not path.exists():
            raise ParsingError(f"File not found: {source}", source=source)

        # Try multiple encodings
        encodings = [self.encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for enc in encodings:
            try:
                df = self._do_parse(source, enc)
                return self._post_process(df)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ParsingError(f"Failed to parse: {e}", source=source)

        raise ParsingError(
            "Could not parse file with any encoding",
            source=source
        )

    def _do_parse(self, source: str, encoding: str) -> pd.DataFrame:
        """
        Actual parsing implementation.
        Override in subclasses for custom parsing logic.
        """
        kwargs = {
            'sep': self.separator,
            'encoding': encoding,
            'on_bad_lines': 'warn'
        }

        if self._columns:
            kwargs['names'] = self._columns
            kwargs['header'] = None

        return pd.read_csv(source, **kwargs)

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-process parsed DataFrame.
        Override in subclasses for custom transformations.
        """
        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate parsed DataFrame has expected columns."""
        errors = []

        if not self.expected_columns:
            return True, errors

        missing = set(self.expected_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        return len(errors) == 0, errors
