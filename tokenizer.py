from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "tokenizer.py requires spaCy. Install it with: pip install spacy"
    ) from exc


__all__ = [
    "TextTokenizer",
    "create_tokenizer",
    "ensure_booknlp_extensions",
    "iter_book_tokens",
    "tokens_to_dicts",
    "tokens_to_json",
    "tokens_to_tsv",
    "write_tsv",
    "token_by_id",
    "span_to_token_range",
]


class TextTokenizer:
    """
    Build the Stage-1 BookNLP-inspired base artifact as a native spaCy Doc.

    The important architectural change is that this class no longer converts
    spaCy tokens into a custom Token dataclass. The returned object is the
    spaCy Doc itself. Project-specific BookNLP-style metadata is attached to
    tokens through spaCy extension attributes:

        token._.paragraph_id
        token._.sentence_id
        token._.token_id_in_sentence
        token._.token_id
        token._.byte_start
        token._.byte_end
        token._.syntactic_head_id
        token._.in_quote
        token._.event

    Downstream artifacts should point to doc token offsets or spaCy Spans.
    This keeps the BookNLP principle of a central token substrate, while making
    the substrate spaCy-native.
    """

    def __init__(
        self,
        nlp: Language,
        *,
        max_length: int = 10_000_000,
        verbose: bool = False,
        progress_every: int = 5_000,
    ) -> None:
        ensure_booknlp_extensions()
        self._nlp = nlp
        self._nlp.max_length = max_length
        self.verbose = bool(verbose)
        self.progress_every = int(progress_every) if progress_every else 0
        self._ensure_sentence_boundaries()

        self._log(
            "[tokenizer:init] TextTokenizer ready: "
            f"pipeline={self._nlp.pipe_names}, "
            f"disabled={list(self._nlp.disabled)}, "
            f"max_length={self._nlp.max_length}"
        )

    @property
    def nlp(self) -> Language:
        """Return the wrapped spaCy pipeline for advanced customization."""
        return self._nlp

    def tokenize(self, text: str) -> Doc:
        """Process raw text and return a BookNLP-annotated spaCy Doc."""
        self._validate_text(text)

        total_t0 = time.perf_counter()
        self._log(
            "[tokenizer:tokenize] Starting raw-text tokenization: "
            f"characters={len(text)}, approx_words={len(text.split())}"
        )

        nlp_t0 = time.perf_counter()
        doc = self._process_text_with_optional_diagnostics(text)
        nlp_t1 = time.perf_counter()

        self._log(
            "[tokenizer:tokenize] spaCy processing finished: "
            f"seconds={nlp_t1 - nlp_t0:.2f}, tokens={len(doc)}, "
            f"has_SENT_START={doc.has_annotation('SENT_START')}, "
            f"has_TAG={doc.has_annotation('TAG')}, "
            f"has_POS={doc.has_annotation('POS')}, "
            f"has_DEP={doc.has_annotation('DEP')}, "
            f"has_LEMMA={doc.has_annotation('LEMMA')}"
        )

        annotate_t0 = time.perf_counter()
        self._annotate_doc(doc)
        annotate_t1 = time.perf_counter()

        self._log(
            "[tokenizer:tokenize] BookNLP-style annotation finished: "
            f"seconds={annotate_t1 - annotate_t0:.2f}, "
            f"booknlp_annotated={doc._.booknlp_annotated}"
        )
        self._log(
            "[tokenizer:tokenize] DONE: "
            f"total_seconds={time.perf_counter() - total_t0:.2f}"
        )
        return doc

    def pipe(self, texts: Iterable[str], *, batch_size: int = 1000) -> Iterator[Doc]:
        """Process many texts and yield BookNLP-annotated spaCy Docs."""
        materialized = list(texts)
        for text in materialized:
            self._validate_text(text)

        self._log(
            "[tokenizer:pipe] Starting pipe: "
            f"documents={len(materialized)}, batch_size={batch_size}"
        )
        for doc_i, doc in enumerate(self._nlp.pipe(materialized, batch_size=batch_size), start=1):
            self._log(f"[tokenizer:pipe] Annotating document {doc_i}/{len(materialized)}")
            self._annotate_doc(doc)
            yield doc
        self._log("[tokenizer:pipe] DONE")

    def tag_pretokenized(
        self,
        words: Sequence[str],
        *,
        spaces: Sequence[bool] | None = None,
        sentence_starts: Sequence[bool] | None = None,
    ) -> Doc:
        """
        Process already-tokenized input and return a native spaCy Doc.

        Args:
            words: Token strings.
            spaces: Whether each token is followed by a space. If omitted,
                every token except the last receives a trailing space.
            sentence_starts: Optional sentence-start flags, one per token.
        """
        self._validate_words(words)
        resolved_spaces = self._resolve_spaces(words, spaces)
        doc = Doc(self._nlp.vocab, words=list(words), spaces=resolved_spaces)

        if sentence_starts is not None:
            self._set_sentence_starts(doc, sentence_starts)

        self._log(
            "[tokenizer:tag_pretokenized] Starting pretokenized pipeline: "
            f"tokens={len(doc)}, pipeline={self._nlp.pipe_names}"
        )
        doc = self._run_pipeline_on_doc(doc)
        self._annotate_doc(doc)
        self._log("[tokenizer:tag_pretokenized] DONE")
        return doc

    def _process_text_with_optional_diagnostics(self, text: str) -> Doc:
        """Run spaCy on raw text, optionally printing per-component timings."""
        if not self.verbose:
            return self._nlp(text)

        self._log("[tokenizer:spacy] make_doc started")
        t0 = time.perf_counter()
        doc = self._nlp.make_doc(text)
        self._log(
            "[tokenizer:spacy] make_doc finished: "
            f"seconds={time.perf_counter() - t0:.2f}, tokens={len(doc)}"
        )
        return self._run_pipeline_on_doc(doc)

    def _run_pipeline_on_doc(self, doc: Doc) -> Doc:
        """Run active spaCy pipeline components with optional timing diagnostics."""
        for name, component in self._nlp.pipeline:
            self._log(f"[tokenizer:spacy] component '{name}' started")
            t0 = time.perf_counter()
            doc = component(doc)
            self._log(
                f"[tokenizer:spacy] component '{name}' finished: "
                f"seconds={time.perf_counter() - t0:.2f}"
            )
        return doc

    def _log(self, message: str) -> None:
        """Print diagnostics only when verbose=True."""
        if not self.verbose:
            return
        print(message)
        sys.stdout.flush()

    def _ensure_sentence_boundaries(self) -> None:
        """Add a sentencizer only when the pipeline cannot create sentences."""
        if self._has_sentence_component():
            return
        self._nlp.add_pipe("sentencizer")

    def _has_sentence_component(self) -> bool:
        sentence_components = {"parser", "senter", "sentencizer"}
        return any(name in self._nlp.pipe_names for name in sentence_components)

    def _annotate_doc(self, doc: Doc) -> None:
        """Attach BookNLP-style token metadata to a spaCy Doc in place."""
        self._log(
            "[tokenizer:annotate] Starting BookNLP-style annotation: "
            f"spacy_tokens={len(doc)}, characters={len(doc.text)}"
        )

        t0 = time.perf_counter()
        token_id_by_spacy_index = _map_non_space_token_ids(doc)
        self._log(
            "[tokenizer:annotate] Non-space token id map built: "
            f"book_tokens={len(token_id_by_spacy_index)}, seconds={time.perf_counter() - t0:.2f}"
        )

        t0 = time.perf_counter()
        _reset_booknlp_token_attributes(doc)
        self._log(
            "[tokenizer:annotate] Existing extension attributes reset: "
            f"seconds={time.perf_counter() - t0:.2f}"
        )

        # IMPORTANT PERFORMANCE FIX:
        # The previous implementation called _char_to_byte_offset(doc.text, offset)
        # twice per token. That function sliced and encoded doc.text[:offset], which
        # makes byte-offset annotation O(num_tokens * text_length). On a full book,
        # this can look like the tokenizer is stuck. We now build a char->byte lookup
        # once in O(text_length) and then do O(1) offset reads per token.
        t0 = time.perf_counter()
        char_to_byte = _build_char_to_byte_offsets(doc.text)
        self._log(
            "[tokenizer:annotate] Char->byte offset lookup built: "
            f"entries={len(char_to_byte)}, seconds={time.perf_counter() - t0:.2f}"
        )

        paragraph_tracker = _ParagraphTracker(doc.text)
        sentence_id = 0
        tokens_done = 0
        total_book_tokens = len(token_id_by_spacy_index)
        annotation_loop_t0 = time.perf_counter()

        for sentence in doc.sents:
            sentence_tokens = [token for token in sentence if not token.is_space]
            if not sentence_tokens:
                continue

            for token_id_in_sentence, token in enumerate(sentence_tokens):
                token_id = token_id_by_spacy_index[token.i]
                token._.token_id = token_id
                token._.sentence_id = sentence_id
                token._.token_id_in_sentence = token_id_in_sentence
                token._.paragraph_id = paragraph_tracker.paragraph_for(token)
                token._.byte_start = char_to_byte[token.idx]
                token._.byte_end = char_to_byte[token.idx + len(token.text)]
                token._.syntactic_head_id = token_id_by_spacy_index.get(
                    token.head.i, token_id
                )
                token._.in_quote = False
                token._.event = "O"

                tokens_done += 1
                if (
                    self.progress_every
                    and tokens_done % self.progress_every == 0
                ):
                    elapsed = time.perf_counter() - annotation_loop_t0
                    rate = tokens_done / elapsed if elapsed > 0 else 0.0
                    self._log(
                        "[tokenizer:annotate] Progress: "
                        f"tokens={tokens_done}/{total_book_tokens}, "
                        f"sentences={sentence_id + 1}, "
                        f"elapsed={elapsed:.2f}s, rate={rate:.1f} tok/s"
                    )

            sentence_id += 1

        doc._.booknlp_annotated = True
        elapsed = time.perf_counter() - annotation_loop_t0
        self._log(
            "[tokenizer:annotate] Annotation loop finished: "
            f"sentences={sentence_id}, tokens={tokens_done}, seconds={elapsed:.2f}"
        )

    def _set_sentence_starts(
        self, doc: Doc, sentence_starts: Sequence[bool]
    ) -> None:
        """Attach external sentence-start information to a pretokenized Doc."""
        if len(sentence_starts) != len(doc):
            raise ValueError("sentence_starts must have the same length as words.")
        for token, is_start in zip(doc, sentence_starts, strict=True):
            token.is_sent_start = bool(is_start)

    def _resolve_spaces(
        self, words: Sequence[str], spaces: Sequence[bool] | None
    ) -> list[bool]:
        """Return validated token-spacing flags."""
        if spaces is None:
            return [True] * (len(words) - 1) + [False]
        if len(spaces) != len(words):
            raise ValueError("spaces must have the same length as words.")
        return list(spaces)

    def _validate_text(self, text: str) -> None:
        """Reject invalid raw-text inputs early."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text:
            raise ValueError("text must not be empty.")

    def _validate_words(self, words: Sequence[str]) -> None:
        """Reject invalid pretokenized inputs early."""
        if not words:
            raise ValueError("words must not be empty.")
        if any(not isinstance(word, str) for word in words):
            raise TypeError("every item in words must be a string.")
        if any(word == "" for word in words):
            raise ValueError("words must not contain empty strings.")


class _ParagraphTracker:
    """Tracks paragraph ids from whitespace between non-space tokens."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._paragraph_id = 0
        self._previous_token_end = 0

    def paragraph_for(self, token: Token) -> int:
        """Return the paragraph id for a token and update internal state."""
        whitespace_before_token = self._text[self._previous_token_end : token.idx]
        if _contains_paragraph_break(whitespace_before_token):
            self._paragraph_id += 1
        self._previous_token_end = token.idx + len(token.text)
        return self._paragraph_id


def create_tokenizer(
    model_name: str = "en_core_web_sm",
    *,
    disable: Sequence[str] = ("ner",),
    max_length: int = 10_000_000,
    verbose: bool = False,
    progress_every: int = 5_000,
) -> TextTokenizer:
    """
    Create a spaCy-native BookNLP-inspired tokenizer.

    NER is disabled by default because Stage 1 should create the central Doc
    artifact. Entity/mention mining remains a later pipeline stage.
    Pass disable=() if you explicitly want spaCy's NER annotations in the Doc.
    Set verbose=True to print component-level timing and annotation progress.
    """
    nlp = _load_spacy_model(model_name, disable=disable)
    return TextTokenizer(
        nlp,
        max_length=max_length,
        verbose=verbose,
        progress_every=progress_every,
    )


def ensure_booknlp_extensions(*, force: bool = False) -> None:
    """Register all spaCy extension attributes used by this module."""
    _set_token_extension("paragraph_id", default=None, force=force)
    _set_token_extension("sentence_id", default=None, force=force)
    _set_token_extension("token_id_in_sentence", default=None, force=force)
    _set_token_extension("token_id", default=None, force=force)
    _set_token_extension("byte_start", default=None, force=force)
    _set_token_extension("byte_end", default=None, force=force)
    _set_token_extension("syntactic_head_id", default=None, force=force)
    _set_token_extension("in_quote", default=False, force=force)
    _set_token_extension("event", default="O", force=force)

    _set_doc_extension("booknlp_annotated", default=False, force=force)
    _set_doc_extension("book_tokens", getter=lambda doc: list(iter_book_tokens(doc)), force=force)
    _set_doc_extension("token_by_id", method=lambda doc, token_id: token_by_id(doc, token_id), force=force)

    _set_span_extension(
        "token_range",
        method=lambda span, inclusive=False: span_to_token_range(
            span, inclusive=inclusive
        ),
        force=force,
    )


def iter_book_tokens(doc: Doc, *, include_space: bool = False) -> Iterator[Token]:
    """Yield tokens belonging to the compact BookNLP-style token view."""
    _require_annotated_doc(doc)
    for token in doc:
        if token.is_space and not include_space:
            continue
        if token._.token_id is None and not include_space:
            continue
        yield token


def token_by_id(doc: Doc, token_id: int) -> Token:
    """Return the spaCy token with the given compact document-level token id."""
    _require_annotated_doc(doc)
    for token in doc:
        if token._.token_id == token_id:
            return token
    raise IndexError(f"No token with token_id={token_id} exists in this Doc.")


def span_to_token_range(span: Span, *, inclusive: bool = False) -> tuple[int, int]:
    """
    Convert a spaCy Span to compact BookNLP-style token offsets.

    By default, the returned range is half-open: (start_token, end_token),
    matching spaCy's own span convention. With inclusive=True, the second value
    is the last token id covered by the span, matching BookNLP's TSV convention.
    """
    _require_annotated_doc(span.doc)
    tokens = [token for token in span if not token.is_space]
    if not tokens:
        raise ValueError("Cannot get a token range for an empty/space-only span.")

    start = tokens[0]._.token_id
    last = tokens[-1]._.token_id
    if start is None or last is None:
        raise ValueError("Span contains tokens without BookNLP token ids.")

    if inclusive:
        return start, last
    return start, last + 1


def tokens_to_dicts(doc: Doc) -> list[dict[str, object]]:
    """Export the BookNLP-style token view as JSON-serializable dicts."""
    return [_token_to_dict(token) for token in iter_book_tokens(doc)]


def tokens_to_json(doc: Doc, *, indent: int = 2) -> str:
    """Serialize the BookNLP-style token view as JSON."""
    return json.dumps(tokens_to_dicts(doc), ensure_ascii=False, indent=indent)


def tokens_to_tsv(doc: Doc) -> str:
    """Serialize the BookNLP-style token view as a TSV string."""
    rows = [_tsv_header()]
    rows.extend(_token_to_tsv_row(token) for token in iter_book_tokens(doc))
    return "\n".join("\t".join(map(str, row)) for row in rows) + "\n"


def write_tsv(doc: Doc, path: str | Path) -> None:
    """Write the BookNLP-style token view to a TSV file."""
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(_tsv_header())
        for token in iter_book_tokens(doc):
            writer.writerow(_token_to_tsv_row(token))


def _token_to_dict(token: Token) -> dict[str, object]:
    """Convert one spaCy token plus extension metadata to a stable dict."""
    return {
        "paragraph_id": token._.paragraph_id,
        "sentence_id": token._.sentence_id,
        "token_id_in_sentence": token._.token_id_in_sentence,
        "token_id": token._.token_id,
        "text": token.text,
        "lemma": token.lemma_ or token.text,
        "char_start": token.idx,
        "char_end": token.idx + len(token.text),
        "pos": token.pos_ or "",
        "fine_pos": token.tag_ or "",
        "dependency_relation": token.dep_ or "",
        "syntactic_head_id": token._.syntactic_head_id,
        "ent_iob": _ent_iob(token),
        "ent_type": _ent_type(token),
        "in_quote": token._.in_quote,
        "event": token._.event,
    }


def _token_to_tsv_row(token: Token) -> list[object]:
    """Convert one spaCy token plus extension metadata to one TSV row."""
    return [
        token._.paragraph_id,
        token._.sentence_id,
        token._.token_id_in_sentence,
        token._.token_id,
        token.text,
        token.lemma_ or token.text,
        token.idx,
        token.idx + len(token.text),
        token._.byte_start,
        token._.byte_end,
        token.pos_ or "",
        token.tag_ or "",
        token.dep_ or "",
        token._.syntactic_head_id,
        _ent_iob(token),
        _ent_type(token),
        int(bool(token._.in_quote)),
        token._.event,
    ]


def _tsv_header() -> list[str]:
    """Return the stable TSV schema."""
    return [
        "paragraph_ID",
        "sentence_ID",
        "token_ID_within_sentence",
        "token_ID_within_document",
        "word",
        "lemma",
        "char_onset",
        "char_offset",
        "byte_onset",
        "byte_offset",
        "POS_tag",
        "fine_POS_tag",
        "dependency_relation",
        "syntactic_head_ID",
        "ent_iob",
        "ent_type",
        "in_quote",
        "event",
    ]


def _load_spacy_model(model_name: str, *, disable: Sequence[str]) -> Language:
    """Load spaCy while keeping dependency errors readable."""
    try:
        return spacy.load(model_name, disable=list(disable))
    except OSError as exc:  # pragma: no cover - depends on local models
        raise OSError(
            f"Could not load spaCy model {model_name!r}. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def _map_non_space_token_ids(doc: Doc) -> dict[int, int]:
    """Map spaCy token indexes to compact document token ids."""
    mapping: dict[int, int] = {}
    next_id = 0
    for token in doc:
        if token.is_space:
            continue
        mapping[token.i] = next_id
        next_id += 1
    return mapping


def _reset_booknlp_token_attributes(doc: Doc) -> None:
    """Clear extension values before annotating/re-annotating a Doc."""
    for token in doc:
        token._.paragraph_id = None
        token._.sentence_id = None
        token._.token_id_in_sentence = None
        token._.token_id = None
        token._.byte_start = None
        token._.byte_end = None
        token._.syntactic_head_id = None
        token._.in_quote = False
        token._.event = "O"


def _ent_iob(token: Token) -> str:
    """Return entity IOB only if the Doc actually contains NER annotation."""
    if not token.doc.has_annotation("ENT_IOB"):
        return ""
    return token.ent_iob_ or "O"


def _ent_type(token: Token) -> str:
    """Return entity type only if the Doc actually contains NER annotation."""
    if not token.doc.has_annotation("ENT_IOB"):
        return ""
    return token.ent_type_ or ""


def _require_annotated_doc(doc: Doc) -> None:
    """Fail early when a raw spaCy Doc is passed where a Stage-1 Doc is needed."""
    if not Doc.has_extension("booknlp_annotated") or not doc._.booknlp_annotated:
        raise ValueError(
            "This Doc has not been annotated by TextTokenizer. "
            "Create it with create_tokenizer(...).tokenize(text)."
        )


def _contains_paragraph_break(whitespace: str) -> bool:
    """Return True when a whitespace stretch contains a blank line."""
    return re.search(r"\n\s*\n", whitespace) is not None


def _build_char_to_byte_offsets(text: str) -> list[int]:
    """Return a lookup table mapping every char offset to its UTF-8 byte offset.

    The returned list has length len(text) + 1, so offsets can be read for both
    token starts and token ends. Building this table once is linear in the text
    length and avoids repeated prefix encoding for every token.
    """
    offsets = [0] * (len(text) + 1)
    byte_offset = 0
    for char_i, char in enumerate(text):
        offsets[char_i] = byte_offset
        byte_offset += len(char.encode("utf-8"))
    offsets[len(text)] = byte_offset
    return offsets


def _char_to_byte_offset(text: str, char_offset: int) -> int:
    """Convert a Python character offset to a UTF-8 byte offset.

    This helper is kept for backward compatibility. For many offsets from the
    same text, prefer _build_char_to_byte_offsets(text).
    """
    return len(text[:char_offset].encode("utf-8"))


def _set_token_extension(name: str, *, default: object, force: bool) -> None:
    if Token.has_extension(name):
        if force:
            Token.set_extension(name, default=default, force=True)
        return
    Token.set_extension(name, default=default)


def _set_doc_extension(
    name: str,
    *,
    default: object | None = None,
    getter: object | None = None,
    method: object | None = None,
    force: bool,
) -> None:
    kwargs = _extension_kwargs(default=default, getter=getter, method=method)
    if Doc.has_extension(name):
        if force:
            Doc.set_extension(name, **kwargs, force=True)
        return
    Doc.set_extension(name, **kwargs)


def _set_span_extension(
    name: str,
    *,
    default: object | None = None,
    getter: object | None = None,
    method: object | None = None,
    force: bool,
) -> None:
    kwargs = _extension_kwargs(default=default, getter=getter, method=method)
    if Span.has_extension(name):
        if force:
            Span.set_extension(name, **kwargs, force=True)
        return
    Span.set_extension(name, **kwargs)


def _extension_kwargs(
    *,
    default: object | None = None,
    getter: object | None = None,
    method: object | None = None,
) -> dict[str, object]:
    provided = [
        ("default", default is not None),
        ("getter", getter is not None),
        ("method", method is not None),
    ]
    if sum(is_set for _, is_set in provided) != 1:
        names = [name for name, is_set in provided if is_set]
        raise ValueError(
            "Exactly one of default, getter, or method must be provided. "
            f"Got: {names}"
        )
    if getter is not None:
        return {"getter": getter}
    if method is not None:
        return {"method": method}
    return {"default": default}


def _parse_disable(value: str) -> tuple[str, ...]:
    if value.strip() == "":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a BookNLP-style token TSV from a spaCy-native Doc."
    )
    parser.add_argument("input", type=Path, help="Input plain-text file.")
    parser.add_argument("output", type=Path, help="Output TSV file.")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name.")
    parser.add_argument(
        "--disable",
        default="ner",
        help="Comma-separated spaCy components to disable. Use '' to disable none.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic timing information while tokenizing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help="When --verbose is used, print annotation progress every N tokens.",
    )
    args = parser.parse_args(argv)

    tokenizer = create_tokenizer(
        args.model,
        disable=_parse_disable(args.disable),
        verbose=args.verbose,
        progress_every=args.progress_every,
    )
    text = args.input.read_text(encoding="utf-8")
    doc = tokenizer.tokenize(text)
    write_tsv(doc, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
