# -*- coding: utf-8 -*-
"""
Parser Factory
--------------
Factory for creating parser instances.
"""

from typing import Dict, Type

from core.interfaces import Parser
from core.exceptions import ConfigurationError
from .firewall import FirewallParser
from .generic import GenericCSVParser
from .syslog import SyslogParser
from .kernel_firewall import KernelFirewallParser


class ParserFactory:
    """
    Factory for creating parser instances.
    Register custom parsers to extend functionality.
    """

    _parsers: Dict[str, Type[Parser]] = {
        "firewall": FirewallParser,
        "csv": GenericCSVParser,
        "syslog": SyslogParser,
        "kernel_firewall": KernelFirewallParser,
    }

    @classmethod
    def register(cls, name: str, parser_class: Type[Parser]) -> None:
        """
        Register a new parser type.

        Args:
            name: Parser identifier
            parser_class: Parser class (must implement Parser interface)
        """
        if not issubclass(parser_class, Parser):
            raise ConfigurationError(
                f"{parser_class.__name__} must implement Parser interface"
            )
        cls._parsers[name] = parser_class

    @classmethod
    def create(cls, parser_type: str, **kwargs) -> Parser:
        """
        Create a parser instance.

        Args:
            parser_type: Type of parser ('firewall', 'csv', 'syslog')
            **kwargs: Parser-specific arguments

        Returns:
            Configured parser instance
        """
        if parser_type not in cls._parsers:
            available = list(cls._parsers.keys())
            raise ConfigurationError(
                f"Unknown parser type: {parser_type}. Available: {available}"
            )

        return cls._parsers[parser_type](**kwargs)

    @classmethod
    def list_parsers(cls) -> list:
        """List available parser types."""
        return list(cls._parsers.keys())


# Convenience function
def get_parser(parser_type: str, **kwargs) -> Parser:
    """
    Get a parser instance.

    Args:
        parser_type: Type of parser
        **kwargs: Parser arguments

    Returns:
        Parser instance
    """
    return ParserFactory.create(parser_type, **kwargs)
