"""Orderflow package."""

try:
    from orderflow._version import __version__
except ImportError:
    __version__ = "0.6.0.dev0"

__all__ = ["__version__"]
