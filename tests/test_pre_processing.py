"""
Unit tests for modules/pre_processing.

Run with (from the project root): pytest tests/test_pre_processing.py -v
"""

import pandas as pd
import pytest

from modules.pre_processing import OneTextPreProcessor
from modules.pre_processing.contraction_handler import (
    handle_contraction,
    apply_contraction_handler,
)


# ── Contraction handler ──────────────────────────────────────────────────────

class TestContractionHandler:
    def test_expands_simple_contraction(self):
        result = handle_contraction("I'm fine")
        assert "I am" in result

    def test_expands_multiple_contractions(self):
        result = handle_contraction("He's going and they've arrived")
        assert "he is" in result.lower() or "they have" in result.lower()

    def test_non_string_passthrough(self):
        assert handle_contraction(None) is None
        assert handle_contraction(42) == 42

    def test_apply_contraction_handler_modifies_column(self):
        df = pd.DataFrame({"text": ["I'm here", "they've gone"]})
        result = apply_contraction_handler(df.copy(), "text")
        assert "I am" in result["text"].iloc[0]
        assert "they have" in result["text"].iloc[1]

    def test_string_without_contraction_unchanged(self):
        text = "the worker fell"
        assert handle_contraction(text) == text


# ── _basic_text_cleanup ──────────────────────────────────────────────────────

class TestBasicTextCleanup:
    @pytest.fixture
    def proc(self):
        return OneTextPreProcessor(lemmatize=False, domain_terms={})

    def test_lowercases(self, proc):
        assert proc._basic_text_cleanup("Hello World") == "hello world"

    def test_collapses_whitespace(self, proc):
        assert proc._basic_text_cleanup("too   many   spaces") == "too many spaces"

    def test_strips_edges(self, proc):
        assert proc._basic_text_cleanup("  trimmed  ") == "trimmed"


# ── pre_process_df — no lemmatize ────────────────────────────────────────────

class TestPreProcessDfNoLemmatize:

    @pytest.fixture
    def proc(self):
        return OneTextPreProcessor(lemmatize=False, domain_terms={})

    def test_output_columns_present(self, proc):
        df = pd.DataFrame({"description": ["worker fell down"]})
        result = proc.pre_process_df(df, "description")
        assert "description_clean" in result.columns

    def test_null_rows_dropped_by_default(self, proc):
        df = pd.DataFrame({"description": ["valid text", None, "another"]})
        result = proc.pre_process_df(df, "description")
        assert len(result) == 2

    def test_drop_null_false_with_nulls_keeps_rows(self):
        """
        When drop_null=False, null rows should NOT be dropped and no error raised.
        """
        df = pd.DataFrame({
            "description": ["Test", None],
        })

        processor = OneTextPreProcessor(
            drop_null=False,
            lemmatize=False,
            domain_terms={}
        )

        result = processor.pre_process_df(df, "description")

        assert len(result) == 2


# ── LemmaHandler via pipeline ───────────────────────────────────────────────

class TestLemmaHandlerViaPipeline:

    @pytest.fixture(scope="class")
    def proc(self):
        return OneTextPreProcessor(
            lemmatize=True,
            domain_terms={},
            lemma_config={"spacy_model": "en_core_web_sm", "use_ner": False},
        )

    def test_tokens_exist(self, proc):
        df = pd.DataFrame({"text": ["workers were running"]})
        result = proc.pre_process_df(df, "text")
        assert "text_tokens" in result.columns

    def test_lemmatization(self, proc):
        df = pd.DataFrame({"text": ["workers were running"]})
        result = proc.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        assert "worker" in tokens
        assert "run" in tokens