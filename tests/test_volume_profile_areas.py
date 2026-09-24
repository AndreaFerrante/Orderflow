import threading

import pandas as pd
import pytest

from orderflow.market.profiles.volume_profile import get_volume_profile_areas


@pytest.mark.parametrize(
    "volumes",
    [
        [1, 1],  # tie: second level matches the first, never beats it
        [2, 1],  # second level smaller than the first
    ],
)
def test_first_tick_seeds_the_poc_price(volumes):
    # Tick 0 is the session's POC. Tick 1 at a different price that does not out-trade it
    # must be placed relative to tick 0's price, not relative to an unset POC of 0.00.
    data = pd.DataFrame(
        {"Price": [5000.00, 5000.25], "Volume": volumes, "SessionType": ["RTH", "RTH"]}
    )

    # 'PO' is 'POC' truncated by the 2-char label array — the documented VA_Areas label.
    assert list(get_volume_profile_areas(data)) == ["PO", "VA"]


def test_zero_volume_tick_after_session_reset_seeds_the_poc():
    # The RTH->ETH reset clears the POC; a zero-volume first tick can't out-trade the cleared 0,
    # so it must still be taken as the new session's POC like tick 0 of the frame is.
    data = pd.DataFrame(
        {
            "Price": [5000.00, 5000.00, 5001.00],
            "Volume": [1, 1, 0],
            "SessionType": ["RTH", "RTH", "ETH"],
        }
    )

    assert list(get_volume_profile_areas(data)) == ["PO", "PO", "PO"]


def test_zero_volume_level_does_not_stall_value_area_expansion():
    # At tick 3 the value area needs > int(0.68 * 5) = 3 contracts: POC 5000.25 (2) + 5000.00 (1)
    # is not enough, and the only level left is 5000.75 — beyond the empty 5000.50 level.
    # Expansion must step over the empty level and reach it instead of looping forever.
    data = pd.DataFrame(
        {
            "Price": [5000.00, 5000.25, 5000.50, 5000.75],
            "Volume": [1, 2, 0, 2],
            "SessionType": ["RTH"] * 4,
        }
    )
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("areas", list(get_volume_profile_areas(data))),
        daemon=True,  # a stuck worker must not keep the test process alive
    )
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "value-area expansion never terminated"
    assert result["areas"] == ["PO", "PO", "na", "VA"]
