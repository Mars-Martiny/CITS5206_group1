import unicodedata
import re

from .contraction_handler import apply_contraction_handler
from .lemma_handler import LemmaHandler
from modules.tokenization import handle_tokenization, normalize_tokens
import pandas as pd

_DEFAULT_LEMMA_CONFIG: dict = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": False,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

_DEFAULT_DOMAIN_TERMS: dict = {
    "case_sensitive": {
        r"\bIP\b": "injured_person",
        r"\bDTO\b": "dump_truck_operator",
    },
    "equipment": {
        "dt": "dump truck",
        "dz": "dozer",
        "ex": "excavator",
    },
}


class OneTextPreProcessor:
    """Pre-processor for text columns in a pandas DataFrame.

    Applies a standard NLP pre-processing pipeline: unicode normalisation,
    contraction expansion, tokenisation, token normalisation, and optional
    lemmatization via spaCy.

    :param keep_numbers: When ``True``, numeric tokens are retained during
        normalisation. Defaults to ``False``.
    :type keep_numbers: bool
    :param column_map: Optional mapping of ``{original_name: new_name}`` used
        to rename DataFrame columns before processing. Defaults to ``{}``.
    :type column_map: dict[str, str]
    :param drop_null: When ``True``, rows with null values in the target text
        column are dropped before processing. Defaults to ``True``.
    :type drop_null: bool
    :param lemmatize: When ``True``, a ``<text_col>_tokens_lemma`` column is
        added containing spaCy-lemmatized tokens. Defaults to ``True``.
    :type lemmatize: bool
    :param lemma_config: Configuration passed to :class:`.LemmaHandler`.
        Supported keys: ``spacy_model``, ``filter_stop_words``,
        ``short_tokens_threshold``, ``use_ner``. Missing keys fall back to
        module-level defaults. Defaults to ``{}``.
    :type lemma_config: dict
    :param domain_terms: Domain-specific abbreviation map applied before any
        other processing step (before lowercasing). Supports two keys:

        - ``case_sensitive`` — ``{regex_pattern: replacement}`` applied with
          ``re.sub``; useful for uppercase sigils like ``IP``.
        - ``equipment`` — ``{abbreviation: full_name}`` matched
          case-insensitively, with an optional trailing numeric id preserved
          (e.g. ``DT1`` → ``dump truck 1``).

        Missing keys fall back to module-level defaults. Pass ``{}`` to
        disable domain expansion entirely. Defaults to ``None`` (use defaults).
    :type domain_terms: dict | None
    """

    def __init__(
        self,
        keep_numbers: bool = False,
        column_map: dict[str, str] = {},
        drop_null: bool = True,
        lemmatize: bool = True,
        lemma_config: dict = {},
        domain_terms: dict | None = None,
    ):
        self.keep_numbers = keep_numbers
        self.column_maps = column_map
        self.drop_null = drop_null
        self.lemmatize = lemmatize
        if lemmatize:
            merged_config = {**_DEFAULT_LEMMA_CONFIG, **lemma_config}
            self._lemma_handler = LemmaHandler(merged_config)
        if domain_terms is None:
            self._domain_terms = _DEFAULT_DOMAIN_TERMS
        else:
            self._domain_terms = domain_terms
    
    def _expand_domain_terms(self, text: str) -> str:
        """Expand domain-specific abbreviations before any other processing.

        Must run before :meth:`_basic_text_cleanup` so that case signals (e.g.
        uppercase ``IP``) are still intact.

        Two expansion passes are applied in order:

        1. **Case-sensitive** patterns from ``domain_terms["case_sensitive"]``
           — each key is a regex pattern, value is the replacement string.
        2. **Equipment prefixes** from ``domain_terms["equipment"]`` —
           matched case-insensitively and only when followed by a numeric id
           (with optional surrounding whitespace), e.g. ``DT1`` → ``dump truck 1``,
           ``EX 03`` → ``excavator 03``. Bare abbreviations without a number
           are intentionally left unchanged to avoid false positives.

        :param text: Raw input string.
        :type text: str
        :returns: String with domain abbreviations expanded.
        :rtype: str
        """
        for pattern, replacement in self._domain_terms.get("case_sensitive", {}).items():
            text = re.sub(pattern, replacement, text)

        for abbrev, full_name in self._domain_terms.get("equipment", {}).items():
            text = re.sub(
                rf"\b{re.escape(abbrev)}\s*(\d+)\b",
                lambda m, fn=full_name: f"{fn} {m.group(1)}",
                text,
                flags=re.IGNORECASE,
            )

        return text

    def _basic_text_cleanup(self, text: str) -> str:
        """Perform basic unicode and whitespace normalisation on a text string.

        Steps applied in order:

        1. NFKC unicode normalisation (fractions, ligatures, fullwidth chars).
        2. Curly quotes and em/en-dashes replaced with ASCII equivalents.
        3. Consecutive whitespace collapsed to a single space and stripped.
        4. Text lowercased.

        :param text: Raw input string to clean.
        :type text: str
        :returns: Cleaned, lowercased string.
        :rtype: str
        """
        # 1. NFKC first, normalize unicode fractions, ligatures, fullwidth
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Normalize quotes, dashes, and whitespace without changing token meaning.
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("–", "-").replace("—", "-")
        
        # 3. Whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        text = text.lower()

        return text
    
    def _rename_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns according to ``self.column_maps``.

        Returns an unmodified copy when no column map was provided.

        :param data_frame: Input DataFrame whose columns may be renamed.
        :type data_frame: pandas.DataFrame
        :returns: Copy of *data_frame* with columns renamed as specified.
        :rtype: pandas.DataFrame
        """
        df = data_frame.copy()
        if self.column_maps:
            df.rename(columns=self.column_maps, inplace=True)
        return df

    def pre_process_df(self, data_frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Run the full pre-processing pipeline on a DataFrame text column.

        The following columns are added to the returned DataFrame:

        - ``<text_col>_clean`` — lowercased, unicode-normalised text.
        - ``<text_col>_tokens_raw`` — tokenised list before normalisation.
        - ``<text_col>`` (in-place) — contractions expanded.
        - ``<text_col>_tokens`` — final normalised token list.
        - ``<text_col>_tokens_lemma`` — spaCy-lemmatized token list, only
          present when ``lemmatize=True``.

        :param data_frame: Input DataFrame containing the text column.
        :type data_frame: pandas.DataFrame
        :param text_col: Name of the column that holds the raw text to process.
        :type text_col: str
        :returns: Copy of *data_frame* with the additional processed columns.
        :rtype: pandas.DataFrame
        """
        df = data_frame.copy()
        df = self._rename_columns(df)
        if (self.drop_null):
            df = df.dropna(subset=[text_col])

        if self._domain_terms:
            df[text_col] = df[text_col].astype(str).apply(self._expand_domain_terms)

        apply_contraction_handler(df, text_col)

        df[f"{text_col}_clean"] = df[text_col].astype(str).apply(
            lambda s: self._basic_text_cleanup(s)
        )

        df[f"{text_col}_tokens_raw"] = df[f"{text_col}_clean"].apply(
            lambda s: handle_tokenization(s)
        )

        df[f"{text_col}_tokens"] = df[f"{text_col}_tokens_raw"].apply(
            lambda toks: normalize_tokens(toks, keep_numbers=self.keep_numbers)
        )

        if self.lemmatize:
            df[f"{text_col}_tokens_lemma"] = df[f"{text_col}_clean"].apply(
                lambda s: self._lemma_handler.lemmatize(s)
            )

        return df