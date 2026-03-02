# -*- coding: utf-8 -*-
"""
Syslog Parser
-------------
Parser for standard syslog format.
"""

import re
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime
from dateutil import parser as date_parser

from .base import BaseParser
from core.exceptions import ParsingError


class SyslogParser(BaseParser):
    """
    Parser for syslog format logs.
    """

    # Standard syslog pattern
    DEFAULT_PATTERN = r'^(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+):\s+(.*)$'

    def __init__(
        self,
        pattern: str = None,
        encoding: str = "utf-8"
    ):
        super().__init__(encoding=encoding)
        self.pattern = pattern or self.DEFAULT_PATTERN
        self._compiled_pattern = re.compile(self.pattern)

    @property
    def expected_columns(self) -> List[str]:
        return ['timestamp', 'hostname', 'process', 'message']

    def _do_parse(self, source: str, encoding: str) -> pd.DataFrame:
        """Parse syslog file line by line."""
        records = []

        with open(source, 'r', encoding=encoding, errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                record = self._parse_line(line.strip(), line_num)
                if record:
                    records.append(record)

        if not records:
            raise ParsingError("No valid syslog entries found", source=source)

        return pd.DataFrame(records)

    def _parse_line(self, line: str, line_num: int) -> Optional[dict]:
        """Parse a single syslog line."""
        match = self._compiled_pattern.match(line)

        if match:
            return {
                'timestamp': self._parse_timestamp(match.group(1)),
                'hostname': match.group(2),
                'process': match.group(3),
                'message': match.group(4)
            }
        else:
            # Store unparsed lines with just message
            return {
                'timestamp': None,
                'hostname': None,
                'process': None,
                'message': line
            }

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse syslog timestamp (doesn't include year)."""
        if not ts_str:
            return None
        try:
            current_year = datetime.now().year
            return date_parser.parse(f"{ts_str} {current_year}")
        except:
            return None

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate syslog data."""
        errors = []

        if 'message' not in df.columns:
            errors.append("Missing 'message' column")

        # Check parse success rate
        if 'timestamp' in df.columns:
            null_pct = df['timestamp'].isnull().mean() * 100
            if null_pct > 50:
                errors.append(f"{null_pct:.1f}% of timestamps could not be parsed")

        return len(errors) == 0, errors
