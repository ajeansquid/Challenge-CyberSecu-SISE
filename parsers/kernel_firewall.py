# -*- coding: utf-8 -*-
"""
Kernel Firewall Log Parser
--------------------------
Parser for raw Linux kernel iptables/netfilter syslog lines.

Expected format (one line example):
    Mar 20 01:29:24 159.84.146.99 kernel: [54783294.108218] DENY FW=6 RULE=999
    IN=eth0 OUT= MAC=... SRC=94.102.61.47 DST=159.84.146.99 LEN=44 TOS=0x00
    PREC=0x00 TTL=238 ID=54321 PROTO=TCP SPT=52502 DPT=3178 WINDOW=65535
    RES=0x00 SYN URGP=0

Produced columns:
    timestamp, hostname, action, fw, rule, interface_in, interface_out,
    src_ip, dst_ip, len, tos, prec, ttl, id, df, proto,
    src_port, dst_port, window, flags
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional

from .base import BaseParser
from core.exceptions import ParsingError

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Syslog header:  "Mar 20 01:29:24 159.84.146.99 kernel: [uptime]"
_HEADER_RE = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})"
    r"\s+(?P<hostname>\S+)\s+kernel:\s+\[[\d.]+\]"
    r"\s+(?P<action>DENY|PERMIT|ACCEPT|DROP|REJECT)"
    r"\s+(?P<rest>.+)$"
)

# Abbreviated month name → month number
_MONTH_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# Key=value extractor (handles empty values like OUT= )
_KV_RE = re.compile(r"(\w+)=(\S*)")

# TCP/UDP flag keywords that appear without a value
_FLAG_KEYWORDS = {"SYN", "ACK", "FIN", "RST", "URG", "PSH", "ECE", "CWR"}


def _parse_flags(text: str) -> str:
    """Extract flag keywords present in a log line fragment."""
    tokens = text.upper().split()
    return ",".join(t for t in tokens if t in _FLAG_KEYWORDS) or ""


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class KernelFirewallParser(BaseParser):
    """
    Parser for Linux kernel netfilter/iptables syslog entries.

    Each line is parsed into structured fields mirroring the raw log.
    The resulting DataFrame can be saved to Parquet via ``save_parquet``
    from ``utils.io``.
    """

    def __init__(self, year: Optional[int] = None, encoding: str = "utf-8"):
        super().__init__(encoding=encoding)
        # Syslog timestamps have no year; use the provided year or current year
        self.year = year or datetime.now().year

    @property
    def expected_columns(self) -> List[str]:
        return ["timestamp", "hostname", "action", "src_ip", "dst_ip"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_parse(self, source: str, encoding: str) -> pd.DataFrame:
        """Read file line-by-line and parse each entry.

        The year is inferred automatically: it starts at ``self.year`` and is
        incremented each time the month rolls back (e.g. December → January),
        which handles log files that span a year boundary.
        """
        records = []
        current_year = self.year
        last_month: Optional[int] = None

        with open(source, "r", encoding=encoding, errors="replace") as fh:
            for line_num, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue

                match = _HEADER_RE.match(line)
                if not match:
                    continue

                # Detect year rollover from the month sequence
                month_num = _MONTH_NUM.get(match.group("month"))
                if month_num is not None and last_month is not None:
                    if month_num < last_month:
                        current_year += 1
                if month_num is not None:
                    last_month = month_num

                record = self._parse_line(match, current_year)
                if record:
                    records.append(record)

        if not records:
            raise ParsingError("No valid kernel firewall entries found", source=source)

        return pd.DataFrame(records)

    def _parse_line(self, match: re.Match, year: int) -> Optional[dict]:
        """Build a record dict from a pre-matched header regex and a resolved year."""
        g = match.groupdict()
        rest = g["rest"]

        # Build timestamp
        ts_str = f"{g['month']} {g['day']} {g['time']} {year}"
        try:
            timestamp = datetime.strptime(ts_str, "%b %d %H:%M:%S %Y")
        except ValueError:
            timestamp = None

        # Extract key=value pairs
        kv = {k: v for k, v in _KV_RE.findall(rest)}

        # Extract TCP/UDP flags (standalone tokens without =)
        flags = _parse_flags(rest)

        return {
            "timestamp": timestamp,
            "hostname": g["hostname"],
            "action": g["action"].upper(),
            "fw": _to_int(kv.get("FW")),
            "rule": _to_int(kv.get("RULE")),
            "interface_in": kv.get("IN", ""),
            "interface_out": kv.get("OUT", ""),
            "src_ip": kv.get("SRC", ""),
            "dst_ip": kv.get("DST", ""),
            "len": _to_int(kv.get("LEN")),
            "ttl": _to_int(kv.get("TTL")),
            "id": _to_int(kv.get("ID")),
            "df": "DF" in rest.upper().split(),
            "proto": kv.get("PROTO", ""),
            "src_port": _to_int(kv.get("SPT")),
            "dst_port": _to_int(kv.get("DPT")),
            "window": _to_int(kv.get("WINDOW")),
            "flags": flags,
        }

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to appropriate types."""
        int_cols = ["fw", "rule", "len", "ttl", "id", "src_port", "dst_port", "window"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if "action" in df.columns:
            df["action"] = df["action"].astype("category")

        if "proto" in df.columns:
            df["proto"] = df["proto"].astype("category")

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate parsed firewall data."""
        errors: List[str] = []

        for col in self.expected_columns:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")

        if "timestamp" in df.columns:
            null_pct = df["timestamp"].isnull().mean() * 100
            if null_pct > 20:
                errors.append(f"{null_pct:.1f}% of timestamps could not be parsed")

        if "action" in df.columns:
            valid = {"DENY", "PERMIT", "ACCEPT", "DROP", "REJECT"}
            found = set(df["action"].dropna().astype(str).unique())
            bad = found - valid
            if bad:
                errors.append(f"Unexpected action values: {bad}")

        return len(errors) == 0, errors
