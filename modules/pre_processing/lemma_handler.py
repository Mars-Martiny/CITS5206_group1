import spacy

spacy.prefer_gpu()

# Map overlapping spaCy NER labels to the regex pipeline's placeholder conventions
# so that text_col_tokens_lemma stays consistent with text_col_tokens.
_LABEL_MAP: dict[str, str] = {
    "CARDINAL": "<num>",
    "ORDINAL": "<ord>",
    "DATE": "<date>",
    "TIME": "<time>",
}


class LemmaHandler:
    """Lemmatizer for pre-processed text using a spaCy transformer model.

    Operates on cleaned text strings (output of ``_basic_text_cleanup``) and
    returns a list of lemmatized tokens.  Named-entity spans are optionally
    replaced with placeholder tokens that follow the same ``<label>``
    convention used by the regex tokenization pipeline.

    Overlapping spaCy NER labels are mapped to the pipeline's existing
    placeholders:

    - ``CARDINAL`` → ``<num>``
    - ``ORDINAL``  → ``<ord>``
    - ``DATE``     → ``<date>``
    - ``TIME``     → ``<time>``

    All other entity labels are emitted as ``<LABEL>`` (e.g. ``<GPE>``,
    ``<ORG>``).

    :param config: Configuration dictionary with the following keys:

        - ``spacy_model`` *(str)* — spaCy model name.
          Defaults to ``"en_core_web_trf"``.
        - ``filter_stop_words`` *(bool)* — when ``True``, stop words are
          removed.  Defaults to ``False``.
        - ``short_tokens_threshold`` *(int)* — lemmas shorter than this value
          are dropped.  ``0`` disables the filter.  Defaults to ``0``.
        - ``use_ner`` *(bool)* — when ``True``, named-entity spans are
          replaced with placeholder tokens and the ``ner`` pipeline component
          is kept enabled.  Defaults to ``True``.

    :type config: dict
    """

    def __init__(self, config: dict) -> None:
        model: str = config.get("spacy_model", "en_core_web_trf")
        self.use_ner: bool = config.get("use_ner", True)
        self.filter_stop_words: bool = config.get("filter_stop_words", False)
        self.short_tokens_threshold: int = config.get("short_tokens_threshold", 0)

        disabled = ["parser"]
        if not self.use_ner:
            disabled.append("ner")

        self.nlp = spacy.load(model, disable=disabled)

    def lemmatize(self, text: str) -> list[str]:
        """Lemmatize a pre-cleaned text string.

        Processing steps (in order):

        1. Run the spaCy pipeline to produce a ``Doc``.
        2. If ``use_ner`` is enabled, build a token-index map of entity spans.
        3. Iterate tokens:

           - Entity spans are replaced by a single placeholder token.
           - Punctuation and whitespace tokens are skipped.
           - Stop words are skipped when ``filter_stop_words`` is ``True``.
           - Lemmas shorter than ``short_tokens_threshold`` are skipped
             (NER placeholders are exempt from this filter).
           - Remaining tokens are appended as their lemma form.

        :param text: Pre-cleaned, lowercased input string.
        :type text: str
        :returns: Ordered list of lemmatized (and optionally filtered) tokens.
        :rtype: list[str]
        """
        doc = self.nlp(text)
        tokens: list[str] = []

        # Build entity span index: token-start-index → (token-end-index, label)
        ent_map: dict[int, tuple[int, str]] = {}
        if self.use_ner:
            for ent in doc.ents:
                ent_map[ent.start] = (ent.end, ent.label_)

        i = 0
        while i < len(doc):
            # --- Named-entity span ---
            if i in ent_map:
                end_idx, label = ent_map[i]
                tokens.append(_LABEL_MAP.get(label, f"<{label}>"))
                i = end_idx
                continue

            token = doc[i]

            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                i += 1
                continue

            # Stop-word filter
            if self.filter_stop_words and token.is_stop:
                i += 1
                continue

            lemma = token.lemma_

            # Short-token filter
            if self.short_tokens_threshold > 0 and len(lemma) < self.short_tokens_threshold:
                i += 1
                continue

            tokens.append(lemma)
            i += 1

        return tokens
