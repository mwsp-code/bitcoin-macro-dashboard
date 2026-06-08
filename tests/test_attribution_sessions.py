import numpy as np
import pandas as pd

from btc_dashboard.features import _native_macro_changes


def test_previous_macro_change_uses_previous_business_session():
    index = pd.date_range("2026-06-04", periods=5, freq="D")
    data = pd.DataFrame(index=index)
    data["NASDAQ"] = [100.0, 102.0, 102.0, 102.0, 103.0]
    data["NASDAQ_OBSERVED"] = [True, True, False, False, True]

    current, previous, _ = _native_macro_changes(data, "NASDAQ")

    friday_return = np.log(102.0 / 100.0)
    monday_return = np.log(103.0 / 102.0)
    assert np.isclose(current.loc["2026-06-06"], friday_return)
    assert np.isclose(current.loc["2026-06-08"], monday_return)
    assert np.isclose(previous.loc["2026-06-08"], friday_return)
