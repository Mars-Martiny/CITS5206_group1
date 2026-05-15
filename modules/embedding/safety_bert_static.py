"""Utilities to build static (non-contextual) word embeddings from a BERT model.

This module is intended for TF-IDF-style pipelines that want a single vector per
word/token in a project vocabulary.

It uses the *input embedding table* of a Hugging Face BERT-like model
(`model.get_input_embeddings()`), and maps a word to one or more WordPiece ids,
then averages the corresponding subword vectors.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except (TypeError, OSError):
        # Some BERT-based checkpoints (e.g. adanish91/safetybert) don't ship a
        # vocab.txt, causing resolved_vocab_files["vocab_file"] == None.
        # Fall back to bert-base-uncased which shares the same vocabulary.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def get_safety_bert_embedding_matrix(
    vocab: dict[str, int],
    *,
    model_name: str = "bert-base-uncased",
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Build an embedding matrix aligned to a {word: index} vocabulary.

    Args:
        vocab: Vocabulary mapping words to row indices.
        model_name: Hugging Face model name to load. This defaults to
            ``bert-base-uncased`` as a safe fallback; set it to your SafetyBERT
            checkpoint name if you have one.
        device: Optional device for the returned matrix (and for the model while
            extracting the embedding table). If None, keeps everything on CPU.
        verbose: Whether to print coverage / OOV diagnostics.

    Returns:
        A tensor of shape ``(max_index+1, hidden_dim)`` aligned to the provided
        indices. If indices are contiguous 0..len(vocab)-1, this is equivalent to
        ``(len(vocab), hidden_dim)``.
    """
    if not vocab:
        raise ValueError("vocab must be non-empty")

    model, tokenizer = _load_model_and_tokenizer(model_name)
    if device is not None:
        model = model.to(device)

    # Static table: (model_vocab_size, hidden_dim)
    static_embeddings = model.get_input_embeddings().weight.detach()

    hidden_dim = int(static_embeddings.shape[1])
    size = max(vocab.values()) + 1
    matrix = torch.zeros(size, hidden_dim, device=static_embeddings.device)

    found = 0
    oov: list[str] = []

    for word, idx in vocab.items():
        if idx < 0 or idx >= size:
            # Should not happen given size=max_idx+1, but keep it safe.
            continue

        # WordPiece: word may split into multiple subword tokens
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        if subword_ids:
            matrix[idx] = static_embeddings[subword_ids].mean(dim=0)
            found += 1
        else:
            oov.append(word)

    if verbose:
        denom = len(vocab)
        print(f"Coverage: {found}/{denom} words ({100 * found / denom:.1f}%)")
        if oov:
            print(f"OOV sample: {oov[:10]}")

    # Ensure the returned tensor is on the requested device (if any).
    if device is not None:
        matrix = matrix.to(device)
    return matrix


def get_contextual_embeddings(
    texts: list[str],
    model_name: str = "bert-base-uncased",
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Encode incident descriptions into contextual per-token embeddings via BERT.

    Args:
        texts:      List of raw description strings.
        model_name: Hugging Face checkpoint name (same one used for static embeddings).
        batch_size: Documents to encode per forward pass.
        max_length: BERT max tokens — 128 is sufficient for incident reports.
        device:     'cuda' or 'cpu'.

    Returns:
        List of float tensors, each shape ``(seq_len, hidden_dim)`` — one per
        document.  ``seq_len`` varies (padding and [CLS]/[SEP] tokens stripped).
    """
    model, tokenizer = _load_model_and_tokenizer(model_name)
    model.to(device)
    all_embeddings: list[torch.Tensor] = []

    texts = list(texts)  # accept pandas Series or any iterable, not just list[str]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        for j, length in enumerate(inputs["attention_mask"].sum(dim=1)):
            # Strip [CLS] (pos 0) and [SEP] (pos length-1) — BiGRU doesn't need them.
            all_embeddings.append(hidden[j, 1 : length - 1, :].cpu())

    return all_embeddings

