"""Inference helpers that stitch preprocessing, dataloaders, and calibrated outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from api.train import _rename_columns

from implementations.simple_bi_gru import build_bigru_dataloader
from implementations.tf_idf import build_tfidf_dataloader
from modules.data_loader.bert_loader import df_to_bert_dataloader
from modules.embedding.bert_config import BertEmbeddingConfig
from modules.embedding.bert_tokenizer import BertTokenizerWrapper
from modules.embedding.safety_bert_static import get_contextual_embeddings
from modules.inference import run_inference

from .loader import load_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FATAL_LABELS = frozenset({"Fatality Potential", "Fatal"})


def _load_confidence_thresholds() -> tuple[float, float]:
    with open(_REPO_ROOT / "config_requirement_check.json", encoding="utf-8") as rf:
        data = json.load(rf)
    high = float(data["confidence_threshold"]["high"])
    med = float(data["confidence_threshold"]["medium"])
    if high > 1.0 or med > 1.0:
        high /= 100.0
        med /= 100.0
    return high, med


def _confidence_tier(score: float, high: float, med: float) -> str:
    if score >= high:
        return "HIGH"
    if score >= med:
        return "MEDIUM"
    return "LOW"


def _default_action(score_label: str) -> str:
    if score_label == "HIGH":
        return "Auto-classified"
    if score_label == "MEDIUM":
        return "Review Required Before Accepting"
    return "Manual Classification Required"


def _damage_action(pred_label: str, tier_label: str) -> str:
    if pred_label in _FATAL_LABELS:
        return "Manual Classification Required [Fatal Potential]"
    return _default_action(tier_label)


def _need_length(mt: str) -> bool:
    return str(mt).lower().startswith("bigru")


def _infer_run_config(bundle: dict) -> dict:
    cfg_saved = bundle["config"]
    return {
        "model": bundle["model"],
        "device": bundle["device"],
        "energy_model": bundle["energy_model"],
        "num_classes": bundle["num_classes"],
        "need_length": _need_length(bundle["model_type"]),
        "temperature": float(cfg_saved.get("temperature", 1.0)),
        "use_temperature": bool(cfg_saved.get("use_temperature", False)),
    }


def _empty_labels(n_rows: int) -> torch.Tensor:
    return torch.zeros(max(n_rows, 0), dtype=torch.long)


def _tfidf_feature_matrix(proc_df: pd.DataFrame, text_col: str, art: dict) -> torch.Tensor:
    lemmas = bool(art.get("lemma_config"))
    tokens_col = f"{text_col}_tokens_lemma" if lemmas else f"{text_col}_tokens"
    docs = proc_df[tokens_col].tolist()

    repr_name = art.get("feature_representation", "tfidf")
    vectorizer = art["vectorizer"]

    if repr_name == "tfidf":
        return vectorizer.transform(docs)
    if repr_name == "tfidf_embed_avg":
        from modules.embedding.safety_bert_static import get_safety_bert_embedding_matrix

        emb_model = art.get("embedding_model_name", "adanish91/safetybert")
        emb_mat = get_safety_bert_embedding_matrix(
            vectorizer.vocab,
            model_name=emb_model,
            verbose=False,
        )
        weighted = vectorizer.transform_weighted_average_embeddings(docs, embedding_matrix=emb_mat)
        return weighted.detach().cpu() if weighted.is_cuda else weighted

    raise ValueError(f"Unsupported TF-IDF feature_representation={repr_name!r}")


def _build_dataloader_tf_idf(df: pd.DataFrame, text_col: str, bundle: dict):
    art = bundle["artifacts"]
    from experiment_setup.tf_idf_runner import pre_process

    keep_numbers = art.get("keep_numbers", False)
    lemma_cfg = art.get("lemma_config", {})
    proc = pre_process(df, text_col, keep_numbers=keep_numbers, lemma_config=lemma_cfg)

    feats = _tfidf_feature_matrix(proc, text_col, art)
    labels = _empty_labels(len(proc))
    batch_size = int(art.get("batch_size", 32))

    return proc, build_tfidf_dataloader(feats, labels, batch_size=batch_size, shuffle=False)


def _build_dataloader_bigru(df: pd.DataFrame, text_col: str, bundle: dict):
    art = bundle["artifacts"]
    from experiment_setup.bi_gru_runner import pre_process

    keep_numbers = art.get("keep_numbers", False)
    lemma_cfg = art.get("lemma_config", {})
    embedding_type = str(art.get("embedding_type", "none")).lower()

    proc = pre_process(df, text_col, keep_numbers=keep_numbers, lemma_config=lemma_cfg)

    if embedding_type == "contextual":
        texts = proc[text_col].astype(str).tolist()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embs = get_contextual_embeddings(
            texts,
            model_name=str(art.get("embedding_model_name", "bert-base-uncased")),
            device=device,
        )
        length_tensor = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
        batch_size = int(art.get("batch_size", 32))
        n_rows = padded.shape[0]
        zeros = torch.zeros(n_rows, dtype=torch.long)
        dl = build_bigru_dataloader(
            padded, length_tensor, zeros, zeros, batch_size=batch_size, shuffle=False
        )
        return proc, dl

    lemmas = bool(lemma_cfg)
    tokens_col = f"{text_col}_tokens_lemma" if lemmas else f"{text_col}_tokens"
    docs = proc[tokens_col].tolist()

    seq_enc = art["seq_enc"]
    max_len = int(art["max_len"])

    seq_tensor = torch.tensor(seq_enc.encode_sequences(docs), dtype=torch.long)
    lengths = torch.tensor([min(len(doc), max_len) for doc in docs], dtype=torch.long)
    batch_size = int(art.get("batch_size", 32))

    zeros = torch.zeros(seq_tensor.shape[0], dtype=torch.long)
    dl = build_bigru_dataloader(
        seq_tensor, lengths, zeros, zeros, batch_size=batch_size, shuffle=False
    )
    return proc, dl


def _build_dataloader_bert(df: pd.DataFrame, text_col: str, bundle: dict):
    art = bundle["artifacts"]
    working = df[[text_col]].copy()
    working["_infer_stub"] = 0

    bert_cfg = BertEmbeddingConfig(
        model_name=str(art.get("model_name", "bert-base-uncased")),
        max_length=int(art["max_length"]),
        dropout=0.1,
        pooling=str(art.get("pooling", "mean")),
        fine_tune=bool(art.get("fine_tune", False)),
    )

    tokenizer = BertTokenizerWrapper(bert_cfg)
    bs = int(art.get("batch_size", 8))
    return df, df_to_bert_dataloader(working, text_col, "_infer_stub", tokenizer, batch_size=bs, shuffle=False)


def _build_dataloader_looped(df: pd.DataFrame, text_col: str, bundle: dict):
    art = bundle["artifacts"]
    working = df[[text_col]].copy()
    working["_infer_stub"] = 0

    bert_cfg = BertEmbeddingConfig(
        model_name=str(art["model_name"]),
        max_length=int(art["max_length"]),
        dropout=0.1,
        pooling="mean",
        fine_tune=False,
    )

    tokenizer = BertTokenizerWrapper(bert_cfg)
    bs = int(art.get("batch_size", 32))
    return df, df_to_bert_dataloader(
        working, text_col, "_infer_stub", tokenizer, batch_size=bs, shuffle=False
    )


def _dispatcher(model_type_lower: str):
    if model_type_lower == "tf_idf":
        return _build_dataloader_tf_idf
    if model_type_lower.startswith("bigru"):
        return _build_dataloader_bigru
    if model_type_lower == "bert":
        return _build_dataloader_bert
    if model_type_lower == "looped_transformer":
        return _build_dataloader_looped
    raise ValueError(f"No inference dataloader mapper for model_type={model_type_lower!r}")


def _run_bundle_on_df(df: pd.DataFrame, text_col: str, bundle: dict):
    mapper = _dispatcher(bundle["model_type"].lower())
    subset_df, dataloader = mapper(df, text_col, bundle)

    cfg_infer = _infer_run_config(bundle)
    cfg_infer["energy_model"] = bundle["artifacts"].get(
        "energy_model", bundle["energy_model"]
    )

    outs = run_inference(cfg_infer, dataloader=dataloader)
    preds_cpu = outs["all_preds"]

    probs = outs["all_probs"]
    scores = probs.max(dim=1).values

    decoded = bundle["label_enc"].decode_many(preds_cpu.tolist())

    out_index = getattr(subset_df, "index", df.index)

    return out_index, decoded, scores.tolist()


def infer(
    dataset_path: str,
    energy_model_dir: str | None = None,
    damage_model_dir: str | None = None,
    output_path: str | None = None,
    text_col: str = "description",
) -> pd.DataFrame:
    """Attach model predictions plus confidence/action columns to ``dataset_path``.

    Args:
        dataset_path: Incident CSV compatible with preprocessing utilities.
        energy_model_dir: Optional folder produced by ``train(..., model_type="energy")``.
        damage_model_dir: Optional folder produced by ``train(..., model_type="damage")``.
        output_path: Optional CSV sink (no row index column).
        text_col: Text column consumed by preprocessing/tokenizers.

    Returns:
        A copy of the input frame with appended prediction/action columns depending on
        whether one or both model directories were supplied.

    Raises:
        ValueError: When neither optional directory path is supplied.
    """
    if energy_model_dir is None and damage_model_dir is None:
        raise ValueError("At least one of energy_model_dir or damage_model_dir must be set.")

    high_thr, med_thr = _load_confidence_thresholds()
    frame = _rename_columns(pd.read_csv(dataset_path))
    enriched = frame.copy()

    bundles: list[tuple[str, dict]] = []
    if energy_model_dir is not None:
        bundles.append(("energy", load_model(energy_model_dir)))
    if damage_model_dir is not None:
        bundles.append(("damage", load_model(damage_model_dir)))

    for bucket, bundle in bundles:
        aligned_idx, preds, confidences = _run_bundle_on_df(enriched, text_col, bundle)
        tiers = [_confidence_tier(float(score), high_thr, med_thr) for score in confidences]

        if bucket == "energy":
            enriched["predicted_energy_type"] = pd.NA
            enriched.loc[aligned_idx, "predicted_energy_type"] = pd.Series(preds, index=aligned_idx)
            enriched["energy_confidence"] = pd.NA
            enriched.loc[aligned_idx, "energy_confidence"] = pd.Series(confidences, index=aligned_idx)
            enriched["energy_score"] = pd.NA
            enriched.loc[aligned_idx, "energy_score"] = pd.Series(tiers, index=aligned_idx)
            acts = [_default_action(score_label) for score_label in tiers]
            enriched["energy_action_required"] = pd.NA
            enriched.loc[aligned_idx, "energy_action_required"] = pd.Series(acts, index=aligned_idx)

        elif bucket == "damage":
            enriched["predicted_damage_potential"] = pd.NA
            enriched.loc[aligned_idx, "predicted_damage_potential"] = pd.Series(
                preds, index=aligned_idx
            )
            enriched["damage_confidence"] = pd.NA
            enriched.loc[aligned_idx, "damage_confidence"] = pd.Series(
                confidences, index=aligned_idx
            )
            enriched["damage_score"] = pd.NA
            enriched.loc[aligned_idx, "damage_score"] = pd.Series(tiers, index=aligned_idx)
            fatal_vals = ["YES" if lbl in _FATAL_LABELS else "NO" for lbl in preds]
            enriched["fatal_flag"] = pd.NA
            enriched.loc[aligned_idx, "fatal_flag"] = pd.Series(fatal_vals, index=aligned_idx)
            acts = [_damage_action(lbl, score_label) for lbl, score_label in zip(preds, tiers)]
            enriched["damage_action_required"] = pd.NA
            enriched.loc[aligned_idx, "damage_action_required"] = pd.Series(acts, index=aligned_idx)

    if output_path is not None:
        enriched.to_csv(output_path, index=False)

    return enriched
