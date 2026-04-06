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

    def test_curly_double_quotes_replaced(self, proc):
        assert proc._basic_text_cleanup("\u201chello\u201d") == '"hello"'

    def test_curly_single_quotes_replaced(self, proc):
        assert proc._basic_text_cleanup("\u2018hi\u2019") == "'hi'"

    def test_em_dash_replaced(self, proc):
        assert proc._basic_text_cleanup("one\u2014two") == "one-two"

    def test_en_dash_replaced(self, proc):
        assert proc._basic_text_cleanup("one\u2013two") == "one-two"

    def test_nfkc_normalisation_fullwidth(self, proc):
        # fullwidth latin capital A (U+FF21) → A → lowercased to a
        assert proc._basic_text_cleanup("\uff21") == "a"


# ── _expand_domain_terms ─────────────────────────────────────────────────────

class TestExpandDomainTerms:
    @pytest.fixture
    def proc(self):
        # use default domain terms
        return OneTextPreProcessor(lemmatize=False)

    def test_ip_expanded(self, proc):
        result = proc._expand_domain_terms("The IP was injured")
        assert "injured_person" in result
        assert "IP" not in result

    def test_dto_expanded(self, proc):
        result = proc._expand_domain_terms("DTO reversed the vehicle")
        assert "dump_truck_operator" in result
        assert "DTO" not in result

    def test_equipment_abbreviation_with_number(self, proc):
        result = proc._expand_domain_terms("DT1 was reversing")
        assert "dump truck 1" in result

    def test_equipment_abbreviation_with_spaced_number(self, proc):
        result = proc._expand_domain_terms("EX 03 stopped")
        assert "excavator 03" in result

    def test_equipment_abbreviation_case_insensitive(self, proc):
        result = proc._expand_domain_terms("dt2 moved")
        assert "dump truck 2" in result

    def test_bare_equipment_abbreviation_not_expanded(self, proc):
        # A bare abbreviation without a trailing number must NOT be expanded
        result = proc._expand_domain_terms("the dt was there")
        assert "dt" in result  # unchanged

        result2 = proc._expand_domain_terms("the ex worker")
        assert "ex" in result2  # unchanged

    def test_empty_domain_terms_no_expansion(self):
        proc = OneTextPreProcessor(lemmatize=False, domain_terms={})
        text = "IP and DT1 are here"
        assert proc._expand_domain_terms(text) == text


# ── pre_process_df — no lemmatize ────────────────────────────────────────────

class TestPreProcessDfNoLemmatize:
    """Tests that do not require spaCy."""

    @pytest.fixture
    def proc(self):
        return OneTextPreProcessor(lemmatize=False, domain_terms={})

    def test_output_columns_present(self, proc):
        df = pd.DataFrame({"description": ["worker fell down"]})
        result = proc.pre_process_df(df, "description")
        assert "description_clean" in result.columns
        assert "description_tokens_raw" in result.columns
        assert "description_tokens" in result.columns

    def test_no_lemma_column_when_disabled(self, proc):
        df = pd.DataFrame({"description": ["worker fell down"]})
        result = proc.pre_process_df(df, "description")
        assert "description_tokens_lemma" not in result.columns

    def test_clean_column_is_lowercase(self, proc):
        df = pd.DataFrame({"description": ["Worker Fell Down"]})
        result = proc.pre_process_df(df, "description")
        assert result["description_clean"].iloc[0] == "worker fell down"

    def test_null_rows_dropped_by_default(self, proc):
        df = pd.DataFrame({"description": ["valid text", None, "another"]})
        result = proc.pre_process_df(df, "description")
        assert len(result) == 2

    def test_drop_null_false_with_nulls_raises(self):
        # Known limitation: drop_null=False does not guard against NaN in
        # _basic_text_cleanup, so passing None values currently raises TypeError.
        # If this test starts failing it means the bug has been fixed — update
        # the test to assert correct behaviour (len(result) == 2) instead.
        proc = OneTextPreProcessor(lemmatize=False, drop_null=False, domain_terms={})
        df = pd.DataFrame({"description": ["valid", None]})
        with pytest.raises(TypeError):
            proc.pre_process_df(df, "description")

    def test_column_rename(self):
        proc = OneTextPreProcessor(
            lemmatize=False, column_map={"raw": "description"}, domain_terms={}
        )
        df = pd.DataFrame({"raw": ["some text"]})
        result = proc.pre_process_df(df, "description")
        assert "description_clean" in result.columns

    def test_keep_numbers_true_preserves_numeric_tokens(self):
        proc = OneTextPreProcessor(lemmatize=False, keep_numbers=True, domain_terms={})
        df = pd.DataFrame({"text": ["worker fell 3 times"]})
        result = proc.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "3" in tokens
        assert "<num>" not in tokens

    def test_keep_numbers_false_replaces_numbers(self):
        proc = OneTextPreProcessor(lemmatize=False, keep_numbers=False, domain_terms={})
        df = pd.DataFrame({"text": ["worker fell 3 times"]})
        result = proc.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "<num>" in tokens
        assert "3" not in tokens

    def test_keep_numbers_false_replaces_dates(self):
        proc = OneTextPreProcessor(lemmatize=False, keep_numbers=False, domain_terms={})
        df = pd.DataFrame({"text": ["incident on 12/03/2024"]})
        result = proc.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "<date>" in tokens

    def test_contraction_expanded_in_pipeline(self, proc):
        df = pd.DataFrame({"text": ["the worker didn't stop"]})
        result = proc.pre_process_df(df, "text")
        # contraction expanded before clean — "did not" or "did not" expected
        assert "not" in result["text_tokens"].iloc[0]


# ── LemmaHandler via OneTextPreProcessor (requires spaCy en_core_web_sm) ─────

class TestLemmaHandlerViaPipeline:
    """
    These tests require the spaCy 'en_core_web_sm' model.

    Key expectation (by design):
      Even when keep_numbers=True, enabling use_ner=True means the lemmatizer
      replaces dates, times, and cardinal/ordinal numbers with placeholder
      tokens (<date>, <time>, <num>, <ord>) via spaCy NER — this is expected
      behaviour and not a bug.
    """

    @pytest.fixture(scope="class")
    def proc_ner(self):
        return OneTextPreProcessor(
            lemmatize=True,
            domain_terms={},
            lemma_config={"spacy_model": "en_core_web_sm", "use_ner": True},
        )

    @pytest.fixture(scope="class")
    def proc_no_ner(self):
        return OneTextPreProcessor(
            lemmatize=True,
            domain_terms={},
            lemma_config={"spacy_model": "en_core_web_sm", "use_ner": False},
        )

    @pytest.fixture(scope="class")
    def proc_stop_words(self):
        return OneTextPreProcessor(
            lemmatize=True,
            domain_terms={},
            lemma_config={
                "spacy_model": "en_core_web_sm",
                "use_ner": False,
                "filter_stop_words": True,
            },
        )

    @pytest.fixture(scope="class")
    def proc_short_token(self):
        return OneTextPreProcessor(
            lemmatize=True,
            domain_terms={},
            lemma_config={
                "spacy_model": "en_core_web_sm",
                "use_ner": False,
                "short_tokens_threshold": 4,
            },
        )

    def test_tokens_column_produced(self, proc_ner):
        df = pd.DataFrame({"text": ["workers were running"]})
        result = proc_ner.pre_process_df(df, "text")
        assert "text_tokens" in result.columns
        assert isinstance(result["text_tokens"].iloc[0], list)

    def test_basic_lemmatization(self, proc_no_ner):
        df = pd.DataFrame({"text": ["workers were running"]})
        result = proc_no_ner.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        # "workers" → "worker", "running" → "run"
        assert "worker" in tokens
        assert "run" in tokens

    def test_punctuation_not_in_tokens(self, proc_no_ner):
        df = pd.DataFrame({"text": ["fell, hit the ground."]})
        result = proc_no_ner.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "," not in tokens
        assert "." not in tokens

    # ── NER placeholder tests ────────────────────────────────────────────────

    def test_ner_replaces_cardinal_number(self, proc_ner):
        df = pd.DataFrame({"text": ["injured 3 workers"]})
        result = proc_ner.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "<num>" in tokens
        assert "3" not in tokens

    def test_ner_replaces_date(self, proc_ner):
        df = pd.DataFrame({"text": ["incident on january 5th 2023"]})
        result = proc_ner.pre_process_df(df, "text")
        tokens = result["text_tokens"].iloc[0]
        assert "<date>" in tokens

    def test_keep_numbers_true_but_ner_still_replaces(self, proc_ner):
        """
        Even with keep_numbers=True, use_ner=True causes the lemmatizer to
        replace numbers/dates with NER placeholders. This is expected behaviour:
        the lemmatize path overwrites text_col_tokens and does not respect
        keep_numbers.
        """
        proc = OneTextPreProcessor(
            keep_numbers=True,  # would preserve numbers in the non-lemma path
            lemmatize=True,
            domain_terms={},
            lemma_config={"spacy_model": "en_core_web_sm", "use_ner": True},
        )
        df = pd.DataFrame({"text": ["there were 3 workers injured"]})
        result = proc.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        # NER overrides keep_numbers — placeholder expected
        assert "<num>" in tokens

    def test_no_ner_keeps_numbers_as_lemma(self, proc_no_ner):
        """With use_ner=False, numeric tokens are not replaced by NER."""
        df = pd.DataFrame({"text": ["there were 3 workers"]})
        result = proc_no_ner.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        assert "<num>" not in tokens

    # ── filter_stop_words ────────────────────────────────────────────────────

    def test_stop_words_filtered(self, proc_stop_words):
        df = pd.DataFrame({"text": ["the worker was injured"]})
        result = proc_stop_words.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        # "the", "was" are stop words
        assert "the" not in tokens
        assert "was" not in tokens

    def test_stop_words_not_filtered_by_default(self, proc_no_ner):
        df = pd.DataFrame({"text": ["the worker was injured"]})
        result = proc_no_ner.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        assert "the" in tokens

    # ── short_tokens_threshold ───────────────────────────────────────────────

    def test_short_tokens_filtered(self, proc_short_token):
        # threshold=4: tokens shorter than 4 chars are dropped
        df = pd.DataFrame({"text": ["a big cat sat"]})
        result = proc_short_token.pre_process_df(df, "text")
        tokens = result["text_tokens_lemma"].iloc[0]
        for tok in tokens:
            assert len(tok) >= 4
