import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_active_macro_session_mask():
    app_path = PROJECT_ROOT / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "active_macro_session_mask"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(app_path), "exec"),
        namespace,
    )
    return namespace["active_macro_session_mask"]


def test_weekend_forward_fills_are_not_active_macro_sessions():
    index = pd.to_datetime(["2026-06-05", "2026-06-06", "2026-06-07"])
    returns = pd.DataFrame(
        {
            "NASDAQ_ret": [0.01, 0.0, 0.0],
            "DXY_ret": [-0.002, 0.0, 0.0],
            "GOLD_ret": [0.003, 0.0, 0.0],
            "OIL_ret": [-0.004, 0.0, 0.0],
            "REAL_YIELD_chg": [0.02, 0.0, 0.0],
        },
        index=index,
    )

    mask = load_active_macro_session_mask()(returns)

    assert mask.to_dict() == {
        index[0]: True,
        index[1]: False,
        index[2]: False,
    }
