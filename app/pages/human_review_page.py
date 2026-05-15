"""Human Review page — three-tier review workflow with partial-save / resume support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.app import list_trained_models, save_uploaded_file
from app.components.banner import show_instruction_banner

_LABELS_PATH = Path(__file__).resolve().parents[2] / "app" / "labels.json"
_BATCH_SIZE = 3
_ENERGY_DECISION_COL = "human_review_energy_decision"
_DAMAGE_DECISION_COL = "human_review_damage_decision"


@st.cache_data
def _load_labels() -> dict[str, list[str]]:
    with open(_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def _score_col(df: pd.DataFrame) -> str | None:
    if "damage_score" in df.columns:
        return "damage_score"
    if "energy_score" in df.columns:
        return "energy_score"
    return None


def _energy_pred_info(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pred = "predicted_energy_type" if "predicted_energy_type" in df.columns else None
    conf = "energy_confidence" if "energy_confidence" in df.columns else None
    return pred, conf


def _damage_pred_info(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pred = "predicted_damage_potential" if "predicted_damage_potential" in df.columns else None
    conf = "damage_confidence" if "damage_confidence" in df.columns else None
    return pred, conf


def _text_col_guess(df: pd.DataFrame) -> str:
    for cand in ("incident_description", "description", "text"):
        if cand in df.columns:
            return cand
    return df.columns[0]


def classify_rows(df: pd.DataFrame) -> dict:
    """Partition rows into review buckets based on score and fatal flag."""
    score_col = _score_col(df)
    fatal_mask = pd.Series([False] * len(df), index=df.index)
    if "fatal_flag" in df.columns:
        fatal_mask = df["fatal_flag"] == "YES"

    fatal_df = df[fatal_mask].copy()

    if score_col:
        non_fatal = df[~fatal_mask]
        high_df = non_fatal[non_fatal[score_col] == "HIGH"].copy()
        _sample_frac = st.session_state.get("settings_high_sample_pct", 10) / 100
        high_sample = high_df.sample(frac=_sample_frac, random_state=42) if len(high_df) > 0 else high_df
        medium_df = non_fatal[non_fatal[score_col] == "MEDIUM"].copy()
        low_df = non_fatal[non_fatal[score_col] == "LOW"].copy()
        high_total = len(high_df)
    else:
        high_sample = pd.DataFrame(columns=df.columns)
        medium_df = pd.DataFrame(columns=df.columns)
        low_df = df[~fatal_mask].copy()
        high_total = 0

    return {
        "fatal": fatal_df,
        "high_sample": high_sample,
        "high_total": high_total,
        "medium": medium_df,
        "low": low_df,
    }


# ── State helpers ──────────────────────────────────────────────────────────────

def _confirmed_rows() -> set[int]:
    return st.session_state.setdefault("confirmed_rows", set())


def _restore_state_from_df(df: pd.DataFrame) -> None:
    """Populate session state from a previously saved CSV."""
    confirmed: set[int] = set()
    col_prefix_pairs = []
    if _ENERGY_DECISION_COL in df.columns:
        col_prefix_pairs.append((_ENERGY_DECISION_COL, "energy"))
    if _DAMAGE_DECISION_COL in df.columns:
        col_prefix_pairs.append((_DAMAGE_DECISION_COL, "damage"))

    for col, prefix in col_prefix_pairs:
        for row_idx, row in df.iterrows():
            raw = row.get(col, "")
            if pd.isna(raw):
                continue
            val = str(raw).strip()
            if not val:
                continue
            confirmed.add(row_idx)
            if val.startswith("Overridden: "):
                st.session_state[f"{prefix}_decision_{row_idx}"] = "Override"
                st.session_state[f"{prefix}_override_{row_idx}"] = val[len("Overridden: "):]
            else:
                st.session_state[f"{prefix}_decision_{row_idx}"] = "Accept"
    st.session_state["confirmed_rows"] = confirmed


def _decision_for(row_idx: int, prefix: str) -> str:
    """Return the saved decision string for a confirmed row and prediction type."""
    decision = st.session_state.get(f"{prefix}_decision_{row_idx}")
    if not decision:
        return ""
    if decision == "Override":
        label = st.session_state.get(f"{prefix}_override_{row_idx}", "")
        return f"Overridden: {label}"
    return "Accepted"


def _build_export_df(review_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a decision column for each prediction type present in the data."""
    result = review_df.copy()
    confirmed = _confirmed_rows()
    has_energy = "predicted_energy_type" in review_df.columns
    has_damage = "predicted_damage_potential" in review_df.columns
    if has_energy:
        result[_ENERGY_DECISION_COL] = [
            _decision_for(idx, "energy") if idx in confirmed else "" for idx in result.index
        ]
    if has_damage:
        result[_DAMAGE_DECISION_COL] = [
            _decision_for(idx, "damage") if idx in confirmed else "" for idx in result.index
        ]
    return result


# ── Review section ─────────────────────────────────────────────────────────────

def _review_section(
    section_key: str,
    section_df: pd.DataFrame,
    text_col: str,
    score_col: str | None,
    energy_pred_col: str | None,
    energy_conf_col: str | None,
    energy_labels: list[str],
    damage_pred_col: str | None,
    damage_conf_col: str | None,
    damage_labels: list[str],
) -> None:
    """Show the next _BATCH_SIZE unconfirmed rows with Accept / Override widgets."""
    confirmed = _confirmed_rows()
    pending = section_df[~section_df.index.isin(confirmed)]

    if pending.empty:
        st.success("All rows in this section have been reviewed.")
        return

    total_pending = len(pending)
    batch_rows = pending.iloc[:_BATCH_SIZE]
    batch_end = len(batch_rows)

    st.caption(f"{total_pending} remaining · showing next {batch_end}")

    for row_idx in batch_rows.index:
        row = batch_rows.loc[row_idx]
        full_text = str(row.get(text_col, row_idx))
        score_value = str(row.get(score_col, "—")) if score_col else "—"
        fatal_value = str(row.get("fatal_flag", "")) if "fatal_flag" in section_df.columns else ""

        with st.container(border=True):
            st.markdown("**Incident**")
            st.write(full_text)

            # Info bar: energy pred + conf, damage pred + conf, score, fatal flag
            info_cols = st.columns(4)
            ci = 0
            if energy_pred_col:
                with info_cols[ci]:
                    st.markdown(f"**Energy:** `{row.get(energy_pred_col, '—')}`")
                ci += 1
                if energy_conf_col:
                    e_conf = row.get(energy_conf_col)
                    with info_cols[ci % 4]:
                        try:
                            st.markdown(f"**E. Conf:** `{float(e_conf):.2%}`")
                        except (TypeError, ValueError):
                            st.markdown(f"**E. Conf:** `{e_conf}`")
                    ci += 1
            if damage_pred_col:
                with info_cols[ci % 4]:
                    st.markdown(f"**Damage:** `{row.get(damage_pred_col, '—')}`")
                ci += 1
                if damage_conf_col:
                    d_conf = row.get(damage_conf_col)
                    with info_cols[ci % 4]:
                        try:
                            st.markdown(f"**D. Conf:** `{float(d_conf):.2%}`")
                        except (TypeError, ValueError):
                            st.markdown(f"**D. Conf:** `{d_conf}`")
                    ci += 1
            with info_cols[min(ci, 3)]:
                st.markdown(f"**Score:** `{score_value}`")
            if fatal_value == "YES":
                with info_cols[min(ci + 1, 3)]:
                    st.markdown("⚠️ **Fatal flagged**")

            # Decision widget(s)
            def _decision_widget(prefix: str, label_header: str, override_labels: list[str]) -> None:
                st.markdown(f"**{label_header} decision**")
                dec_col, lbl_col = st.columns([1, 2])
                with dec_col:
                    decision = st.selectbox(
                        label_header,
                        ["Accept", "Override"],
                        key=f"{prefix}_decision_{row_idx}",
                        index=None,
                        placeholder="— choose —",
                        label_visibility="collapsed",
                    )
                with lbl_col:
                    if decision == "Override":
                        st.selectbox(
                            "Correct label",
                            override_labels,
                            key=f"{prefix}_override_{row_idx}",
                            label_visibility="collapsed",
                        )
                    elif decision == "Accept":
                        st.caption(f"✓ Accepting {label_header.lower()} prediction")
                    else:
                        st.caption(f"☐ {label_header} not yet reviewed")

            if energy_pred_col:
                _decision_widget("energy", "Energy", energy_labels)
            if damage_pred_col:
                _decision_widget("damage", "Damage", damage_labels)

    if st.button("Confirm & Next →", key=f"confirm_{section_key}", type="primary"):
        confirmed.update(batch_rows.index.tolist())
        st.session_state["confirmed_rows"] = confirmed
        st.rerun()


def _already_reviewed_section(
    review_df: pd.DataFrame,
    text_col: str,
    energy_pred_col: str | None,
    damage_pred_col: str | None,
) -> None:
    confirmed = _confirmed_rows()
    if not confirmed:
        return

    rows = review_df[review_df.index.isin(confirmed)].copy()
    display_cols = [text_col]
    if energy_pred_col and energy_pred_col in rows.columns:
        display_cols.append(energy_pred_col)
        rows[_ENERGY_DECISION_COL] = [_decision_for(i, "energy") for i in rows.index]
        display_cols.append(_ENERGY_DECISION_COL)
    if damage_pred_col and damage_pred_col in rows.columns:
        display_cols.append(damage_pred_col)
        rows[_DAMAGE_DECISION_COL] = [_decision_for(i, "damage") for i in rows.index]
        display_cols.append(_DAMAGE_DECISION_COL)

    with st.expander(f"✅ Already Reviewed ({len(rows)} rows)", expanded=False):
        st.dataframe(rows[display_cols], use_container_width=True)


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("Human Review")
show_instruction_banner("human_review")

labels = _load_labels()

st.subheader("Load Inference Results")
input_mode = st.radio("Input mode", ["Upload CSV", "Run inference inline"])

if input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload inference results CSV", type=["csv"])
    if uploaded and st.session_state.get("_upload_file_id") != uploaded.file_id:
        # Only process on a genuinely new upload — not on every rerun triggered
        # by widget interactions (Confirm & Next, etc.), which would wipe confirmed_rows.
        st.session_state["_upload_file_id"] = uploaded.file_id
        df_loaded = pd.read_csv(save_uploaded_file(uploaded))
        st.session_state["review_df"] = df_loaded
        for key in ("batch_fatal", "batch_high", "batch_medium", "batch_low"):
            st.session_state.pop(key, None)
        resume_cols = [c for c in (_ENERGY_DECISION_COL, _DAMAGE_DECISION_COL) if c in df_loaded.columns]
        if resume_cols:
            _restore_state_from_df(df_loaded)
            already = df_loaded[resume_cols[0]].dropna().astype(str).str.strip().ne("").sum()
            st.success(f"Loaded {len(df_loaded)} rows — resuming with {already} already reviewed.")
        else:
            st.session_state["confirmed_rows"] = set()
            st.success(f"Loaded {len(df_loaded)} rows.")

else:
    _SORT_OPTIONS = ["val_f1_macro", "test_f1_macro", "val_accuracy", "training_time_sec"]
    _NONE_LABEL = "— none —"

    sort_by = st.selectbox("Sort models by", _SORT_OPTIONS, key="hr_sort")

    def _model_select(label: str, task_filter: str) -> str | None:
        entries = list_trained_models(task_filter=task_filter)
        options = [_NONE_LABEL] + [e.label for e in entries]
        paths = {e.label: str(e.path) for e in entries}
        sel = st.selectbox(label, options, key=f"hr_{task_filter}_model")
        return paths.get(sel) if sel != _NONE_LABEL else None

    energy_dir = _model_select("Energy Model", "energy")
    damage_dir = _model_select("Damage Model", "damage")
    inline_csv = st.file_uploader("Input CSV", type=["csv"], key="hr_inline_csv")
    text_col_inline = st.text_input("Text column", value="description", key="hr_text_col")

    if st.button("Run Inference for Review"):
        if energy_dir is None and damage_dir is None:
            st.error("Select at least one model.")
            st.stop()
        if inline_csv is None:
            st.error("Upload an input CSV.")
            st.stop()
        import api

        with st.spinner("Running inference…"):
            try:
                result_df = api.infer(
                    dataset_path=str(save_uploaded_file(inline_csv)),
                    energy_model_dir=energy_dir,
                    damage_model_dir=damage_dir,
                    text_col=text_col_inline,
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()
        st.session_state["review_df"] = result_df
        st.session_state["confirmed_rows"] = set()
        st.session_state["_upload_file_id"] = None
        for key in ("batch_fatal", "batch_high", "batch_medium", "batch_low"):
            st.session_state.pop(key, None)
        st.success(f"Inference complete — {len(result_df)} rows loaded.")

# ── Guard ──────────────────────────────────────────────────────────────────────

review_df: pd.DataFrame | None = st.session_state.get("review_df")

if review_df is None:
    st.info("Load or run inference results above to begin the review workflow.")
    st.stop()

tiers = classify_rows(review_df)
fatal_df = tiers["fatal"]
high_sample = tiers["high_sample"]
high_total = tiers["high_total"]
medium_df = tiers["medium"]
low_df = tiers["low"]

text_col = _text_col_guess(review_df)
score_col = _score_col(review_df)
energy_pred_col, energy_conf_col = _energy_pred_info(review_df)
damage_pred_col, damage_conf_col = _damage_pred_info(review_df)
energy_labels: list[str] = labels.get("energy", labels["damage"])
damage_labels: list[str] = labels.get("damage", labels["damage"])

# ── Summary bar ────────────────────────────────────────────────────────────────

confirmed_count = len(_confirmed_rows())
pending_count = len(fatal_df) + len(high_sample) + len(medium_df) + len(low_df) - confirmed_count

summary_cols = st.columns(6)
with summary_cols[0]:
    st.metric("Total rows", len(review_df))
with summary_cols[1]:
    st.metric("Fatal flagged", len(fatal_df))
with summary_cols[2]:
    st.metric("HIGH tier", high_total)
with summary_cols[3]:
    st.metric("MEDIUM tier", len(medium_df))
with summary_cols[4]:
    st.metric("LOW tier", len(low_df))
with summary_cols[5]:
    st.metric("Reviewed", confirmed_count)

# ── Save progress (always accessible) ─────────────────────────────────────────

save_col, _ = st.columns([1, 3])
with save_col:
    progress_csv = _build_export_df(review_df).to_csv(index=False)
    st.download_button(
        "💾 Save Progress",
        data=progress_csv,
        file_name="review_in_progress.csv",
        mime="text/csv",
        help="Download current state. Re-upload this file tomorrow to resume where you left off.",
    )

st.divider()

# ── Already reviewed ───────────────────────────────────────────────────────────

_already_reviewed_section(review_df, text_col, energy_pred_col, damage_pred_col)

# ── Pending review sections ────────────────────────────────────────────────────

_section_kwargs = dict(
    text_col=text_col,
    score_col=score_col,
    energy_pred_col=energy_pred_col,
    energy_conf_col=energy_conf_col,
    energy_labels=energy_labels,
    damage_pred_col=damage_pred_col,
    damage_conf_col=damage_conf_col,
    damage_labels=damage_labels,
)

with st.expander(f"🔴 Fatal — Mandatory Review ({len(fatal_df)} rows)", expanded=True):
    _review_section("fatal", fatal_df, **_section_kwargs)

_high_pct = st.session_state.get("settings_high_sample_pct", 10)
with st.expander(
    f"🟡 HIGH Confidence — Spot-check Sample ({len(high_sample)} rows, {_high_pct}% of {high_total} total)",
    expanded=True,
):
    if high_total > 0:
        st.caption(
            f"Showing {len(high_sample)} rows sampled from {high_total} HIGH-confidence predictions ({_high_pct}% spot-check)."
        )
    _review_section("high", high_sample, **_section_kwargs)

with st.expander(f"🟠 MEDIUM Confidence — Light Review ({len(medium_df)} rows)", expanded=True):
    _review_section("medium", medium_df, **_section_kwargs)

with st.expander(f"🔵 LOW Confidence — Full Manual Classification ({len(low_df)} rows)", expanded=True):
    _review_section("low", low_df, **_section_kwargs)

# ── Final export ───────────────────────────────────────────────────────────────

st.divider()
if st.button("Export Final Results", type="primary"):
    exported = _build_export_df(review_df)
    st.download_button(
        "Download reviewed CSV",
        data=exported.to_csv(index=False),
        file_name="reviewed_output.csv",
        mime="text/csv",
    )
