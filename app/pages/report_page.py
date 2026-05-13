"""Report page — renders Markdown reports from the reports/ directory."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

st.title("Report")

reports = sorted(REPORTS_DIR.glob("*.md")) if REPORTS_DIR.exists() else []

if not reports:
    st.info("No reports found. Add `.md` files to the `reports/` directory.")
else:
    names = {p.stem.replace("-", " ").replace("_", " ").title(): p for p in reports}
    choice = st.selectbox("Select report", list(names))
    st.markdown(names[choice].read_text(encoding="utf-8"))
