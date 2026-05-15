"""About page — project information."""

from __future__ import annotations

import streamlit as st

st.title("About")

st.markdown(
    """
## Incident Report Classifier

This application is the **Incident Report Classifier Model** developed as part of the
**CITS5206 Capstone Project** at the University of Western Australia.

---

### Note on this Web UI

This web interface is a **Proof of Concept (PoC)** and a minimal working product.
As such, it exposes only a subset of the available model training options:

- **Hyperparameter search** is not available through the UI.
- **Detailed training options** — such as overriding specific configs or advanced
  training parameters — cannot be configured here.

For full control over training, including hyperparameter tuning and config overrides,
run the training code directly from the code base inside `experiment_setup` directory.

`main_notebook.py` provides the documentation on how to run hyperparameter search.
"""
)
