"""Canonical storage service facade."""

from importlib import import_module


def __getattr__(name: str):
	module = import_module("orderflow.storage.storage_service")
	return getattr(module, name)

