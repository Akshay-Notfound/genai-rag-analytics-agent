"""
query_executor.py
------------------
Executes a generated pandas expression against the active dataframe in a
restricted namespace (no builtins, no imports) so the agent can safely turn
LLM output into an actual computed result instead of just returning text.
"""

from __future__ import annotations

import pandas as pd


class QueryExecutionError(Exception):
    pass


def run_pandas_expression(expression: str, df: pd.DataFrame):
    """
    Evaluate a pandas expression like "df['revenue'].mean()" against `df`.
    Restricted to a minimal namespace to avoid executing arbitrary code.
    """
    safe_globals = {"__builtins__": {"len": len, "sum": sum, "min": min, "max": max, "round": round}}
    safe_locals = {"df": df, "pd": pd}

    try:
        return eval(expression, safe_globals, safe_locals)  # noqa: S307 (scoped eval)
    except Exception as exc:  # noqa: BLE001
        raise QueryExecutionError(f"Could not execute generated query: {expression!r} ({exc})") from exc
