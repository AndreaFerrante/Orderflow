import importlib
from pathlib import Path
import tomllib

import pytest


LEGACY_MODULES = {
    "orderflow._volume_factory": {
        "get_market_evening_session",
        "get_tickers_in_folder_mem_optim",
    },
    "orderflow.auctions": {
        "aggregate_auctions",
        "compute_forward_outcomes",
        "get_valid_blocks",
    },
    "orderflow.configuration": set(),
    "orderflow.volume_profile": {
        "get_daily_high_and_low_by_session",
        "get_daily_session_moving_POC",
        "get_dynamic_cumulative_delta_per_session",
        "get_volume_profile_areas",
        "get_volume_profile_node_volume",
        "get_volume_profile_peaks_valleys",
    },
    "orderflow.volume_profile_kde": {
        "gaussian_kde_numba_parallel",
        "get_kde_high_low_price_peaks",
    },
    "orderflow.vwap": {"get_vwap"},
}


@pytest.mark.parametrize("module_name", sorted(LEGACY_MODULES))
def test_legacy_runner_module_exports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    missing = sorted(
        symbol for symbol in LEGACY_MODULES[module_name] if not hasattr(module, symbol)
    )
    assert not missing, f"{module_name} missing {missing}"


def test_pytz_is_declared_runtime_dependency() -> None:
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]

    assert any(dependency.startswith("pytz") for dependency in dependencies)