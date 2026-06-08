import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_fallback_functions():
    app_path = PROJECT_ROOT / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    required = {
        "normalize_daily_index",
        "clean_real_yield",
        "read_real_yield_fallback",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    namespace = {
        "pd": pd,
        "REAL_YIELD_CACHE_FILE": PROJECT_ROOT / "missing-runtime-cache.csv",
        "REAL_YIELD_SEED_FILE": PROJECT_ROOT / "data" / "real_yield_seed.csv",
        "CACHE_FILE": PROJECT_ROOT / "missing-complete-cache.csv",
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(app_path), "exec"),
        namespace,
    )
    return namespace["read_real_yield_fallback"]


def test_bundled_seed_survives_streamlit_cold_start():
    read_fallback = load_fallback_functions()
    series, label, errors = read_fallback()

    assert label == "bundled official seed"
    assert series.name == "REAL_YIELD"
    assert series.index[-1] == pd.Timestamp("2026-06-05")
    assert float(series.iloc[-1]) == 2.19
    assert not errors
