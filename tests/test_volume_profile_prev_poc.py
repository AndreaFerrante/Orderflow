import pandas as pd

from orderflow.market.profiles.volume_profile import get_daily_session_moving_POC


def test_prev_poc_carries_friday_session_across_the_weekend():
    # Three sessions, one dominant price each. The Sunday-evening session opens ~49h after Friday's
    # close; its Prev_POC must still be the session that just ended (Friday's), on every tick.
    rows = [
        ("2025-01-08 17:00", "ETH", 100.0, 1), ("2025-01-09 10:00", "RTH", 100.0, 9),  # Wed eve -> Thu
        ("2025-01-09 17:00", "ETH", 200.0, 1), ("2025-01-10 10:00", "RTH", 200.0, 9),  # Thu eve -> Fri
        ("2025-01-12 17:00", "ETH", 300.0, 1), ("2025-01-13 10:00", "RTH", 300.0, 9),  # Sun eve -> Mon
    ]
    data = pd.DataFrame(rows, columns=["Datetime", "SessionType", "Price", "Volume"])
    # Same Datetime representation the enrichment runner hands this function.
    data["Datetime"] = pd.Series(pd.to_datetime(data["Datetime"]).dt.to_pydatetime(), dtype=object)

    _, prev_poc = get_daily_session_moving_POC(data)

    assert list(prev_poc) == [0.0, 0.0, 100.0, 100.0, 200.0, 200.0]
