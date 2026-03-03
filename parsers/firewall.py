# -*- coding: utf-8 -*-
"""
Firewall Log Parsers
--------------------
FirewallParser        : ancien format CSV générique (séparateur configurable)
FirewallExportParser  : format export CSV (`;` séparé, sans en-tête)
                        timestamp;src_ip;dst_ip;proto;src_port;dst_port;rule;action;interface_in;interface_out;fw
KibanaExportParser    : export Kibana/Elasticsearch (`,` séparé, avec en-tête)
                        Colonnes dupliquées `.keyword`, timestamp "Mar 2, 2026 @ HH:MM:SS.mmm"
"""

import pandas as pd
from typing import List, Tuple, Optional

from .base import BaseParser
from core.config import get_config

# Colonnes du format export challenge (`;` séparé)
EXPORT_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "proto",
    "src_port",
    "dst_port",
    "rule",
    "action",
    "interface_in",
    "interface_out",
    "fw",
]

# Mapping colonnes Kibana → noms standards
_KIBANA_RENAME = {
    "@timestamp": "timestamp",
    "action": "action",
    "dstport": "dst_port",
    "interface": "interface_in",
    "interface_out": "interface_out",
    "ipdst": "dst_ip",
    "ipsrc": "src_ip",
    "policyid": "rule",
    "proto": "proto",
    "srcport": "src_port",
}

# Colonnes présentes dans le raw parser mais absentes du format Kibana
_KIBANA_MISSING_COLS = ["hostname", "fw", "len", "ttl", "id", "df", "window", "flags"]


class FirewallParser(BaseParser):
    """
    Parser for firewall logs.
    Expected format: ipsrc,ipdst,portdst,proto,action,date,regle
    """

    def __init__(
        self, separator: str = ",", columns: List[str] = None, parse_dates: bool = True
    ):
        config = get_config()
        cols = columns or config.parser.firewall_columns
        super().__init__(separator=separator, columns=cols)
        self.parse_dates = parse_dates

    @property
    def expected_columns(self) -> List[str]:
        return ["ipsrc", "ipdst", "portdst", "action"]

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert types and parse dates."""
        # Ensure port is numeric
        if "portdst" in df.columns:
            df["portdst"] = pd.to_numeric(df["portdst"], errors="coerce")

        # Parse dates
        if self.parse_dates and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Standardize action values
        if "action" in df.columns:
            df["action"] = df["action"].str.strip().str.capitalize()

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate firewall log structure."""
        is_valid, errors = super().validate(df)

        # Check action values
        if "action" in df.columns:
            valid_actions = {"Permit", "Deny"}
            actual_actions = set(df["action"].dropna().unique())
            invalid = actual_actions - valid_actions
            if invalid:
                errors.append(f"Invalid action values: {invalid}")
                is_valid = False

        # Check for valid IPs (basic check)
        if "ipsrc" in df.columns:
            null_count = df["ipsrc"].isnull().sum()
            if null_count > 0:
                errors.append(f"{null_count} rows with null source IP")

        return is_valid, errors


class FirewallExportParser(BaseParser):
    """
    Parser for ';'-separated export format.

    Expected columns (no header in file):
        timestamp ; src_ip ; dst_ip ; proto ; src_port ; dst_port ;
        rule ; action ; interface_in ; interface_out ; fw

    Example line:
        2025-03-20 01:29:24;94.102.61.47;159.84.146.99;TCP;52502;3178;999;DENY;eth0;;6
    """

    def __init__(self, parse_dates: bool = True):
        super().__init__(separator=";", columns=EXPORT_COLUMNS)
        self.parse_dates = parse_dates

    @property
    def expected_columns(self) -> List[str]:
        return ["timestamp", "src_ip", "dst_ip", "action"]

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate types."""
        # Timestamp
        if self.parse_dates and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Numeric columns
        for col in ("src_port", "dst_port", "rule", "fw"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Standardise action  (DENY / PERMIT → uppercase category)
        if "action" in df.columns:
            df["action"] = df["action"].str.strip().str.upper().astype("category")

        # Protocol as category
        if "proto" in df.columns:
            df["proto"] = df["proto"].str.strip().astype("category")

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate the parsed export data."""
        errors: List[str] = []

        for col in self.expected_columns:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")

        if "timestamp" in df.columns:
            null_pct = df["timestamp"].isnull().mean() * 100
            if null_pct > 5:
                errors.append(f"{null_pct:.1f}% of timestamps could not be parsed")

        if "action" in df.columns:
            valid = {"DENY", "PERMIT"}
            found = set(df["action"].dropna().astype(str).unique())
            bad = found - valid
            if bad:
                errors.append(f"Unexpected action values: {bad}")

        if "src_ip" in df.columns:
            null_count = df["src_ip"].isnull().sum()
            if null_count > 0:
                errors.append(f"{null_count} rows with null source IP")

        return len(errors) == 0, errors


class KibanaExportParser(BaseParser):
    """
    Parser for Kibana / Elasticsearch CSV exports.

    The export contains:
    - A header row with column names (some duplicated as `.keyword`)
    - Meta columns: `_id`, `_ignored`, `_index`, `_score`
    - Timestamp format: "Mar 2, 2026 @ 20:45:01.000"
    - Empty values represented as "(empty)"

    The produced DataFrame matches the column schema of KernelFirewallParser.
    Columns not available in the Kibana export (len, ttl, df, window, flags…)
    are added as NaN so the schema stays consistent.

    Example header:
        @timestamp,_id,_ignored,_index,_score,action,action.keyword,datetime,...
    """

    def __init__(self, parse_dates: bool = True):
        super().__init__(separator=",")
        self.parse_dates = parse_dates

    @property
    def expected_columns(self) -> List[str]:
        return ["timestamp", "src_ip", "dst_ip", "action"]

    def _do_parse(self, source: str, encoding: str) -> pd.DataFrame:
        """Read CSV and immediately drop .keyword duplicates and meta columns."""
        df = pd.read_csv(
            source,
            encoding=encoding,
            on_bad_lines="warn",
        )

        # Drop Elasticsearch meta columns
        meta = {"_id", "_ignored", "_index", "_score"}
        df = df.drop(columns=[c for c in df.columns if c in meta], errors="ignore")

        # Drop all `.keyword` duplicate columns
        df = df.drop(
            columns=[c for c in df.columns if c.endswith(".keyword")], errors="ignore"
        )

        # Drop redundant datetime column (already in @timestamp)
        df = df.drop(columns=["datetime"], errors="ignore")

        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename, clean and cast columns to the standard schema."""
        # Rename to standard names
        df = df.rename(columns=_KIBANA_RENAME)

        # Replace "(empty)" sentinel with proper NaN
        df = df.replace("(empty)", pd.NA)

        # Timestamp: "Mar 2, 2026 @ 20:45:01.000"
        if self.parse_dates and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="%b %d, %Y @ %H:%M:%S.%f",
                errors="coerce",
            )

        # Numeric columns
        for col in ("src_port", "dst_port", "rule"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Action → uppercase category
        if "action" in df.columns:
            df["action"] = df["action"].str.strip().str.upper().astype("category")

        # Protocol → category
        if "proto" in df.columns:
            df["proto"] = df["proto"].str.strip().astype("category")

        # Add columns missing from Kibana export to align schema with raw parser
        #   df=False by default (Don't Fragment flag unknown), rest NaN
        for col in _KIBANA_MISSING_COLS:
            if col not in df.columns:
                if col == "df":
                    df["df"] = False
                elif col in ("len", "ttl", "id", "window"):
                    df[col] = pd.array([pd.NA] * len(df), dtype="Int64")
                else:
                    df[col] = pd.NA  # hostname, flags → string NA

        # Reorder to match raw parser column order
        ordered = [
            "timestamp",
            "hostname",
            "action",
            "fw",
            "rule",
            "interface_in",
            "interface_out",
            "src_ip",
            "dst_ip",
            "len",
            "ttl",
            "id",
            "df",
            "proto",
            "src_port",
            "dst_port",
            "window",
            "flags",
        ]
        present = [c for c in ordered if c in df.columns]
        extra = [c for c in df.columns if c not in ordered]
        df = df[present + extra]

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate the parsed Kibana export."""
        errors: List[str] = []

        for col in self.expected_columns:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")

        if "timestamp" in df.columns:
            null_pct = df["timestamp"].isnull().mean() * 100
            if null_pct > 5:
                errors.append(f"{null_pct:.1f}% of timestamps could not be parsed")

        if "action" in df.columns:
            valid = {"DENY", "PERMIT"}
            found = set(df["action"].dropna().astype(str).unique())
            bad = found - valid
            if bad:
                errors.append(f"Unexpected action values: {bad}")

        if "src_ip" in df.columns:
            null_count = df["src_ip"].isnull().sum()
            if null_count > 0:
                errors.append(f"{null_count} rows with null source IP")

        return len(errors) == 0, errors
