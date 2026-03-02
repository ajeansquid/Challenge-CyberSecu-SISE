# -*- coding: utf-8 -*-
"""Helper functions and utilities."""

import re
import time
import logging
from functools import wraps
from typing import Optional, Callable, Any


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def validate_ip(ip: str) -> bool:
    """
    Validate an IP address format.

    Args:
        ip: IP address string

    Returns:
        True if valid IPv4 address
    """
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False

    parts = ip.split('.')
    return all(0 <= int(part) <= 255 for part in parts)


def validate_port(port: int) -> bool:
    """
    Validate a port number.

    Args:
        port: Port number

    Returns:
        True if valid port (0-65535)
    """
    return 0 <= port <= 65535


def normalize_action(action: str) -> str:
    """
    Normalize firewall action strings.

    Args:
        action: Raw action string

    Returns:
        Normalized action ('permit' or 'deny')
    """
    action_lower = str(action).lower().strip()

    permit_variants = ['permit', 'allow', 'accept', 'pass', 'allowed', '1', 'true']
    deny_variants = ['deny', 'drop', 'block', 'reject', 'denied', '0', 'false']

    if action_lower in permit_variants:
        return 'permit'
    elif action_lower in deny_variants:
        return 'deny'
    else:
        return action_lower


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_number(n: float, precision: int = 2) -> str:
    """
    Format a number for display with thousand separators.

    Args:
        n: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if isinstance(n, int) or n == int(n):
        return f"{int(n):,}"
    return f"{n:,.{precision}f}"


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")

        return result

    return wrapper


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between levels

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
