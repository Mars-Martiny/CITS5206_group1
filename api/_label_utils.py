"""Helpers for rebuilding label encoders from saved checkpoints."""

from __future__ import annotations

from typing import Any

from modules.encoding import LabelEncoder


def label_encoder_from_class_dict(class_dict: dict[Any, Any]) -> LabelEncoder:
    """Reconstruct a fitted :class:`~modules.encoding.LabelEncoder` from saved metadata.

    Args:
        class_dict: Mapping ``class_index -> label`` as stored under training ``config["class_dict"]``.

    Returns:
        A label encoder usable for decoding integer predictions.
    """
    id_to_label: dict[int, str] = {}
    for k, v in class_dict.items():
        id_to_label[int(k)] = str(v)
    le = LabelEncoder()
    le.id_to_label = id_to_label
    le.label_to_id = {lbl: idx for idx, lbl in id_to_label.items()}
    le._fitted = True
    return le
