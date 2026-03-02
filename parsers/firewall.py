# -*- coding: utf-8 -*-
"""
Firewall Log Parser
-------------------
Parser for firewall logs in course format.
Format: ipsrc,ipdst,portdst,proto,action,date,regle
"""

import pandas as pd
from typing import List, Tuple

from .base import BaseParser
from core.config import get_config


class FirewallParser(BaseParser):
    """
    Parser for firewall logs.
    Expected format: ipsrc,ipdst,portdst,proto,action,date,regle
    """

    def __init__(
        self,
        separator: str = ",",
        columns: List[str] = None,
        parse_dates: bool = True
    ):
        config = get_config()
        cols = columns or config.parser.firewall_columns
        super().__init__(separator=separator, columns=cols)
        self.parse_dates = parse_dates

    @property
    def expected_columns(self) -> List[str]:
        return ['ipsrc', 'ipdst', 'portdst', 'action']

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert types and parse dates."""
        # Ensure port is numeric
        if 'portdst' in df.columns:
            df['portdst'] = pd.to_numeric(df['portdst'], errors='coerce')

        # Parse dates
        if self.parse_dates and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Standardize action values
        if 'action' in df.columns:
            df['action'] = df['action'].str.strip().str.capitalize()

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate firewall log structure."""
        is_valid, errors = super().validate(df)

        # Check action values
        if 'action' in df.columns:
            valid_actions = {'Permit', 'Deny'}
            actual_actions = set(df['action'].dropna().unique())
            invalid = actual_actions - valid_actions
            if invalid:
                errors.append(f"Invalid action values: {invalid}")
                is_valid = False

        # Check for valid IPs (basic check)
        if 'ipsrc' in df.columns:
            null_count = df['ipsrc'].isnull().sum()
            if null_count > 0:
                errors.append(f"{null_count} rows with null source IP")

        return is_valid, errors
