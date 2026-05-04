"""Streamlit dashboard: equity curve, recent trades, AI decision log.

Run with `streamlit run src/dashboard/app.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="ai-trader", layout="wide")
    st.title("ai-trader — live dashboard")

    eq_path = settings.project_root / settings.equity_csv
    dec_path = settings.project_root / settings.decisions_csv

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Equity curve")
        eq = _read_csv(eq_path)
        if eq.empty:
            st.info("No equity snapshots yet.")
        else:
            eq["ts"] = pd.to_datetime(eq["ts"])
            st.line_chart(eq.set_index("ts")["equity_usd"])

    with col2:
        st.subheader("AI decisions (last 50)")
        dec = _read_csv(dec_path)
        if dec.empty:
            st.info("No decisions logged yet.")
        else:
            st.dataframe(dec.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
