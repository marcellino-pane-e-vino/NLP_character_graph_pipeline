from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from spacy.tokens import Doc, Span, Token
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "coreference_cluster_extractor.py requires spaCy. Install it with: pip install spacy"
    ) from exc

__all__ = [
    "ExtractedMention",
    "ExtractedCluster",
    "ExtractedChunkClusters",
    "MaverickCoreferenceClusterExtractor",
    "create_coreference_cluster_extractor",
]

DEFAULT_MODEL_NAME = "sapienzanlp/maverick-mes-litbank"
BAD_CANONICAL_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "we", "us", "our", "ours", "ourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
}


@dataclass(slots=True)
class ExtractedMention:
    """One mention produced by Maverick inside a local chunk.

    Offsets are local spaCy token offsets over chunk_doc and use half-open spans.
    local_mention_id is stable inside its local_cluster_id.
    """

    local_mention_id: int
    local_start: int
    local_end: int
    text: str
    head_local_i: int | None = None


@dataclass(slots=True)
class ExtractedCluster:
    """One local Maverick cluster inside a chunk."""

    local_cluster_id: int
    canonical_name: str | None
    mentions: list[ExtractedMention] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedChunkClusters:
    """All local clusters produced for one chunk."""

    chunk_id: str
    clusters: list[ExtractedCluster] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedMaverickInput:
    """Sentence-split tokenized input plus alignment back to local spaCy token indexes."""

    sentences: list[list[str]]
    flat_token_to_spacy_i: list[int]
    flat_token_to_sentence_i: list[int]
    flat_token_to_sentence_token_i: list[int]

    @property
    def flat_tokens(self) -> list[str]:
        return [token for sent in self.sentences for token in sent]

    def same_sentence(self, flat_start: int, flat_end_inclusive: int) -> bool:
        if flat_start < 0 or flat_end_inclusive < flat_start:
            return False
        if flat_end_inclusive >= len(self.flat_token_to_sentence_i):
            return False
        return self.flat_token_to_sentence_i[flat_start] == self.flat_token_to_sentence_i[flat_end_inclusive]


class MaverickCoreferenceClusterExtractor:
    """Run Maverick coreference on one DocChunk and return local cluster records.

    Contract:
        - Maverick owns mention discovery and local clustering.
        - This adapter does not create CorefLayer objects.
        - This adapter never writes to doc._ or token._.
        - Extracted mention offsets are local spaCy token offsets over chunk_doc.
        - Global offsets are computed later by local_clusters.py using DocChunk maps.
        - The adapter does not keep per-chunk predictions after extraction.
    """

    def __init__(
        self,
        *,
        hf_name_or_path: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        singletons: bool = True,
        require_booknlp_doc: bool = True,
        verbose: bool = True,
        maverick_predict_kwargs: dict[str, Any] | None = None,
        validate_alignment: bool = True,
        clean_mentions: bool = True,
        max_mention_tokens: int | None = 20,
        drop_cross_sentence_mentions: bool = True,
        debug_raw_mentions: int = 0,
        cpu_load_first: bool = True,
        precision_policy: str = "auto",
        p100_fallback_to_float32: bool = True,
    ) -> None:
        self.hf_name_or_path = hf_name_or_path
        self.device = device or _default_device()
        self.singletons = singletons
        self.require_booknlp_doc = require_booknlp_doc
        self.verbose = verbose
        self.maverick_predict_kwargs = dict(maverick_predict_kwargs or {})
        self.validate_alignment = validate_alignment
        self.clean_mentions = clean_mentions
        self.max_mention_tokens = max_mention_tokens
        self.drop_cross_sentence_mentions = drop_cross_sentence_mentions
        self.debug_raw_mentions = debug_raw_mentions
        self.cpu_load_first = bool(cpu_load_first)
        self.precision_policy = str(precision_policy or "auto").lower()
        self.p100_fallback_to_float32 = bool(p100_fallback_to_float32)
        self._model: Any | None = None
        self._active_precision: str | None = None
        self._log(
            "[coref] Extractor build: streaming-safe "
            f"device={self.device!r}, precision_policy={self.precision_policy!r}, "
            f"cpu_load_first={self.cpu_load_first}."
        )

    def extract(self, chunk_or_doc: Any) -> ExtractedChunkClusters:
        """Extract local Maverick clusters from a DocChunk or plain spaCy Doc."""
        chunk_id = getattr(chunk_or_doc, "chunk_id", "chunk_000")
        doc = getattr(chunk_or_doc, "chunk_doc", chunk_or_doc)

        self._validate_doc(doc)
        prepared = self._prepare_maverick_input(doc)

        if not prepared.flat_token_to_spacy_i:
            self._log(f"[coref] {chunk_id}: empty chunk; extracted 0 clusters.")
            return ExtractedChunkClusters(chunk_id=chunk_id, clusters=[])

        try:
            prediction = self._predict(prepared)
        except Exception as exc:
            if self._should_retry_float32(exc):
                self._log(
                    "[coref][fallback] Prediction failed under float16 on P100-like runtime. "
                    "Reloading model in float32 and retrying once."
                )
                self._reload_model_with_precision("float32")
                prediction = self._predict(prepared)
            else:
                if _is_cuda_oom(exc):
                    raise RuntimeError(
                        "Maverick failed with CUDA out-of-memory. Float32 fallback would normally use more GPU "
                        "memory, so the safe correction is to reduce CHUNK_SIZE / overlap_sentences."
                    ) from exc
                raise

        if self.validate_alignment:
            self._validate_prediction_alignment(prediction, prepared)

        extracted = self._prediction_to_extracted_clusters(
            chunk_id=chunk_id,
            doc=doc,
            prediction=prediction,
            prepared=prepared,
        )
        self._log(
            f"[coref] {chunk_id}: extracted "
            f"{len(extracted.clusters)} local clusters, "
            f"{sum(len(c.mentions) for c in extracted.clusters)} local mentions."
        )
        return extracted

    def __call__(self, chunk_or_doc: Any) -> ExtractedChunkClusters:
        return self.extract(chunk_or_doc)

    def clear_runtime_state(self) -> None:
        """Compatibility no-op for old notebooks; predictions are not retained."""
        gc.collect()
        _empty_cuda_cache_if_available()

    def _predict(self, prepared: _PreparedMaverickInput) -> dict[str, Any]:
        model = self._load_model()
        with self._prediction_precision_context():
            return model.predict(
                prepared.sentences,
                singletons=self.singletons,
                **self.maverick_predict_kwargs,
            )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._preflight_runtime_report()
        _patch_transformers_embedding_resize_mean_resizing(verbose=self.verbose)

        try:
            from maverick import Maverick
        except ImportError as exc:  # pragma: no cover - depends on local installation
            raise ImportError(
                "Missing dependency 'maverick-coref'. Install it with:\n"
                "    pip install maverick-coref\n"
                "Then re-run the notebook cell that creates the extractor."
            ) from exc

        requested_device = self.device
        load_device = "cpu" if self.cpu_load_first and _is_cuda_device(requested_device) else requested_device

        self._log(
            f"[coref] Loading Maverick model {self.hf_name_or_path!r}. "
            f"load_device={load_device!r}, requested_device={requested_device!r}."
        )
        self._log_memory("[coref][memory] before Maverick(...)")

        self._model = Maverick(hf_name_or_path=self.hf_name_or_path, device=load_device)
        self._log_memory("[coref][memory] after Maverick(...) on load_device")
        self._normalize_maverick_model_for_inference(
            target_device=requested_device,
            precision=self._resolve_initial_precision(),
        )
        return self._model

    def _reload_model_with_precision(self, precision: str) -> None:
        self._unload_model()
        self._active_precision = precision
        # Force precision for the next load.
        old_policy = self.precision_policy
        try:
            self.precision_policy = precision
            self._load_model()
        finally:
            self.precision_policy = old_policy

    def _unload_model(self) -> None:
        self._model = None
        gc.collect()
        _empty_cuda_cache_if_available()

    def _resolve_initial_precision(self) -> str:
        if self.precision_policy in {"float16", "fp16", "half"}:
            return "float16"
        if self.precision_policy in {"float32", "fp32", "full"}:
            return "float32"
        if _is_cuda_device(self.device):
            return "float16"
        return "float32"

    def _normalize_maverick_model_for_inference(self, *, target_device: str | None = None, precision: str = "float16") -> None:
        if self._model is None:
            return

        try:
            import torch
        except Exception:
            self._log("[coref][warning] torch unavailable; skipping Maverick dtype normalization.")
            return

        target_device = target_device or self.device
        inner_model = getattr(self._model, "model", None)
        if inner_model is None or not hasattr(inner_model, "parameters"):
            self._log("[coref][warning] Maverick inner torch model not found; skipping dtype normalization.")
            return

        inner_model.eval()
        use_cuda = _is_cuda_device(target_device) and torch.cuda.is_available()
        resolved_precision = "float32" if not use_cuda else precision

        if resolved_precision == "float16" and use_cuda:
            inner_model.half()
            inner_model.to(target_device)
            dtype_policy = "cuda-float16"
        elif use_cuda:
            inner_model.float()
            inner_model.to(target_device)
            dtype_policy = "cuda-float32"
        else:
            inner_model.float()
            inner_model.to("cpu")
            target_device = "cpu"
            dtype_policy = "cpu-float32"

        if hasattr(self._model, "device"):
            self._model.device = target_device
        self.device = target_device
        self._active_precision = resolved_precision

        param_dtypes = sorted({str(param.dtype) for param in inner_model.parameters()})
        buffer_dtypes = sorted({str(buffer.dtype) for buffer in inner_model.buffers() if buffer.is_floating_point()})
        param_devices = sorted({str(param.device) for param in inner_model.parameters()})

        self._log(
            "[coref] Maverick model normalized for inference: "
            f"policy={dtype_policy}, "
            f"parameter_dtypes={param_dtypes}, "
            f"floating_buffer_dtypes={buffer_dtypes}, "
            f"parameter_devices={param_devices}."
        )
        _empty_cuda_cache_if_available()
        self._log_memory("[coref][memory] after dtype/device normalization")

    def _prediction_precision_context(self):
        from contextlib import nullcontext

        try:
            import torch
        except Exception:
            return nullcontext()

        if not _is_cuda_device(self.device) or not torch.cuda.is_available():
            return nullcontext()

        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", enabled=False)
        try:
            return torch.cuda.amp.autocast(enabled=False)
        except Exception:
            return nullcontext()

    def _should_retry_float32(self, exc: BaseException) -> bool:
        if not self.p100_fallback_to_float32:
            return False
        if self._active_precision != "float16":
            return False
        if not _is_p100_runtime():
            return False
        if _is_cuda_oom(exc):
            return False
        return _looks_like_dtype_error(exc)

    def _preflight_runtime_report(self) -> None:
        try:
            import torch
        except Exception:
            self._log("[coref][runtime] torch unavailable.")
            return

        self._log(f"[coref][runtime] torch={getattr(torch, '__version__', 'unknown')}")
        self._log(f"[coref][runtime] torch_cuda={getattr(torch.version, 'cuda', None)}")
        self._log(f"[coref][runtime] cuda_available={torch.cuda.is_available()}")

        if torch.cuda.is_available():
            try:
                self._log(f"[coref][runtime] gpu={torch.cuda.get_device_name(0)!r}")
                self._log(f"[coref][runtime] capability={torch.cuda.get_device_capability(0)}")
                self._log(f"[coref][runtime] compiled_arches={torch.cuda.get_arch_list()}")
            except Exception as exc:
                self._log(f"[coref][runtime][warning] CUDA diagnostics failed: {exc!r}")

    def _log_memory(self, label: str) -> None:
        try:
            import resource
            max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            self._log(f"{label}: max_rss={max_rss_mb:.1f} MiB")
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                self._log(f"{label}: cuda_allocated={allocated:.1f} MiB, cuda_reserved={reserved:.1f} MiB")
        except Exception:
            pass

    def _validate_doc(self, doc: Doc) -> None:
        if not isinstance(doc, Doc):
            raise TypeError("extract(...) expects a DocChunk or a spaCy Doc.")

        if self.require_booknlp_doc:
            has_marker = Doc.has_extension("booknlp_annotated")
            is_annotated = has_marker and bool(getattr(doc._, "booknlp_annotated"))
            if not is_annotated:
                raise ValueError(
                    "This Doc does not look like it was produced by tokenizer.py. "
                    "Create it with: tokenizer = create_tokenizer(); doc = tokenizer.tokenize(text)."
                )

        if not doc.has_annotation("SENT_START"):
            raise ValueError("The Doc has no sentence boundaries. Maverick integration expects doc.sents.")

    def _prepare_maverick_input(self, doc: Doc) -> _PreparedMaverickInput:
        sentences: list[list[str]] = []
        flat_token_to_spacy_i: list[int] = []
        flat_token_to_sentence_i: list[int] = []
        flat_token_to_sentence_token_i: list[int] = []

        for sentence_i, sent in enumerate(doc.sents):
            sent_tokens = [token for token in sent if self._is_model_token(token)]
            if not sent_tokens:
                continue

            sentences.append([token.text for token in sent_tokens])
            for sentence_token_i, token in enumerate(sent_tokens):
                flat_token_to_spacy_i.append(token.i)
                flat_token_to_sentence_i.append(sentence_i)
                flat_token_to_sentence_token_i.append(sentence_token_i)

        self._log(
            "[coref] Prepared Maverick input: "
            f"{len(sentences)} sentences, {len(flat_token_to_spacy_i)} tokens."
        )
        return _PreparedMaverickInput(
            sentences=sentences,
            flat_token_to_spacy_i=flat_token_to_spacy_i,
            flat_token_to_sentence_i=flat_token_to_sentence_i,
            flat_token_to_sentence_token_i=flat_token_to_sentence_token_i,
        )

    def _validate_prediction_alignment(self, prediction: dict[str, Any], prepared: _PreparedMaverickInput) -> None:
        predicted_tokens = prediction.get("tokens") if isinstance(prediction, dict) else None
        if predicted_tokens is None:
            self._log("[coref][warning] Maverick prediction has no 'tokens' field; cannot validate alignment.")
            return

        expected_tokens = prepared.flat_tokens
        if list(predicted_tokens) == expected_tokens:
            self._log("[coref] Alignment check OK: Maverick tokens match spaCy tokens.")
            return

        self._log("[coref][warning] Alignment mismatch: Maverick tokens differ from spaCy tokens.")
        self._log(f"[coref][warning] expected={len(expected_tokens)} predicted={len(predicted_tokens)}")
        for i, (expected, predicted) in enumerate(zip(expected_tokens, predicted_tokens)):
            if expected != predicted:
                self._log(f"[coref][warning] first mismatch at flat token {i}: spaCy={expected!r}, Maverick={predicted!r}")
                return
        if len(expected_tokens) != len(predicted_tokens):
            self._log("[coref][warning] token sequences share a prefix but have different lengths.")

    def _prediction_to_extracted_clusters(
        self,
        *,
        chunk_id: str,
        doc: Doc,
        prediction: dict[str, Any],
        prepared: _PreparedMaverickInput,
    ) -> ExtractedChunkClusters:
        raw_clusters = _extract_clusters_token_offsets(prediction)
        text_clusters = _extract_clusters_text_mentions(prediction)

        extracted_clusters: list[ExtractedCluster] = []
        dropped_mentions = 0
        debug_printed = 0

        for raw_cluster_i, raw_cluster in enumerate(raw_clusters):
            mentions: list[ExtractedMention] = []
            text_cluster = text_clusters[raw_cluster_i] if raw_cluster_i < len(text_clusters) else []

            for raw_mention_i, (flat_start, flat_end) in enumerate(raw_cluster):
                expected_text = str(text_cluster[raw_mention_i]) if raw_mention_i < len(text_cluster) else None
                span = self._flat_offsets_to_spacy_span(
                    doc=doc,
                    prepared=prepared,
                    flat_start=flat_start,
                    flat_end=flat_end,
                    expected_text=expected_text,
                )
                if span is None:
                    dropped_mentions += 1
                    continue
                if self.clean_mentions and self._should_drop_span(span):
                    dropped_mentions += 1
                    continue
                if self.debug_raw_mentions and debug_printed < self.debug_raw_mentions:
                    self._log(
                        "[coref][debug] kept mention "
                        f"raw=({flat_start}, {flat_end}) span=({span.start}, {span.end}) "
                        f"text={span.text!r} expected={expected_text!r}"
                    )
                    debug_printed += 1

                mentions.append(
                    ExtractedMention(
                        local_mention_id=raw_mention_i,
                        local_start=span.start,
                        local_end=span.end,
                        text=span.text,
                        head_local_i=_safe_span_head_i(span),
                    )
                )

            if mentions:
                extracted_clusters.append(
                    ExtractedCluster(
                        local_cluster_id=raw_cluster_i,
                        canonical_name=_choose_canonical_name_from_extracted(mentions),
                        mentions=mentions,
                    )
                )

        if dropped_mentions:
            self._log(f"[coref] {chunk_id}: dropped {dropped_mentions} suspicious/pathological mentions.")
        return ExtractedChunkClusters(chunk_id=chunk_id, clusters=extracted_clusters)

    def _flat_offsets_to_spacy_span(
        self,
        *,
        doc: Doc,
        prepared: _PreparedMaverickInput,
        flat_start: int,
        flat_end: int,
        expected_text: str | None = None,
    ) -> Span | None:
        if flat_start < 0 or flat_end < flat_start:
            self._log(f"[coref][warning] Skipping invalid Maverick mention offset ({flat_start}, {flat_end}).")
            return None

        candidates: list[tuple[str, int, int, Span]] = []
        inclusive = self._candidate_span_from_inclusive_offsets(doc=doc, prepared=prepared, flat_start=flat_start, flat_end_inclusive=flat_end)
        if inclusive is not None:
            candidates.append(("inclusive", flat_start, flat_end, inclusive))
        if flat_end > flat_start:
            exclusive = self._candidate_span_from_inclusive_offsets(doc=doc, prepared=prepared, flat_start=flat_start, flat_end_inclusive=flat_end - 1)
            if exclusive is not None:
                candidates.append(("exclusive", flat_start, flat_end - 1, exclusive))

        if not candidates:
            self._log(
                "[coref][warning] Skipping out-of-range Maverick mention offset "
                f"({flat_start}, {flat_end}); available flat tokens: {len(prepared.flat_token_to_spacy_i)}."
            )
            return None

        if expected_text is not None:
            for _, _, _, span in candidates:
                if _texts_match_loosely(span.text, expected_text):
                    return span

        for _, cand_start, cand_end_inclusive, span in candidates:
            if prepared.same_sentence(cand_start, cand_end_inclusive):
                return span

        convention, _, _, span = candidates[0]
        if self.drop_cross_sentence_mentions:
            self._log(f"[coref][drop] cross-sentence mention: raw=({flat_start}, {flat_end}) convention={convention} text={span.text!r}")
            return None
        return span

    def _candidate_span_from_inclusive_offsets(
        self,
        *,
        doc: Doc,
        prepared: _PreparedMaverickInput,
        flat_start: int,
        flat_end_inclusive: int,
    ) -> Span | None:
        if flat_end_inclusive >= len(prepared.flat_token_to_spacy_i):
            return None
        if flat_start < 0 or flat_end_inclusive < flat_start:
            return None
        if self.drop_cross_sentence_mentions and not prepared.same_sentence(flat_start, flat_end_inclusive):
            return None
        spacy_start = prepared.flat_token_to_spacy_i[flat_start]
        spacy_end = prepared.flat_token_to_spacy_i[flat_end_inclusive] + 1
        if spacy_start >= spacy_end:
            return None
        return doc[spacy_start:spacy_end]

    def _should_drop_span(self, span: Span) -> bool:
        tokens = [token for token in span if not token.is_space]
        if not tokens:
            return True
        if "\n" in span.text:
            self._log(f"[coref][drop] newline-containing mention: {span.text!r}")
            return True
        if self.max_mention_tokens is not None and len(tokens) > self.max_mention_tokens:
            self._log(f"[coref][drop] too-long mention ({len(tokens)} tokens): {span.text!r}")
            return True
        return False

    def _is_model_token(self, token: Token) -> bool:
        if token.is_space:
            return False
        if self.require_booknlp_doc and hasattr(token._, "token_id"):
            return token._.token_id is not None
        return True

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


def create_coreference_cluster_extractor(
    *,
    hf_name_or_path: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    singletons: bool = True,
    require_booknlp_doc: bool = True,
    verbose: bool = True,
    maverick_predict_kwargs: dict[str, Any] | None = None,
    validate_alignment: bool = True,
    clean_mentions: bool = True,
    max_mention_tokens: int | None = 20,
    drop_cross_sentence_mentions: bool = True,
    debug_raw_mentions: int = 0,
    cpu_load_first: bool = True,
    precision_policy: str = "auto",
    p100_fallback_to_float32: bool = True,
) -> MaverickCoreferenceClusterExtractor:
    return MaverickCoreferenceClusterExtractor(
        hf_name_or_path=hf_name_or_path,
        device=device,
        singletons=singletons,
        require_booknlp_doc=require_booknlp_doc,
        verbose=verbose,
        maverick_predict_kwargs=maverick_predict_kwargs,
        validate_alignment=validate_alignment,
        clean_mentions=clean_mentions,
        max_mention_tokens=max_mention_tokens,
        drop_cross_sentence_mentions=drop_cross_sentence_mentions,
        debug_raw_mentions=debug_raw_mentions,
        cpu_load_first=cpu_load_first,
        precision_policy=precision_policy,
        p100_fallback_to_float32=p100_fallback_to_float32,
    )


def _extract_clusters_token_offsets(prediction: dict[str, Any]) -> list[list[tuple[int, int]]]:
    if not isinstance(prediction, dict):
        raise TypeError(f"Maverick predict(...) was expected to return a dict, got {type(prediction).__name__}.")
    raw_clusters = prediction.get("clusters_token_offsets")
    if raw_clusters is None:
        available_keys = ", ".join(sorted(map(str, prediction.keys())))
        raise KeyError(f"Maverick prediction does not contain 'clusters_token_offsets'. Available keys: {available_keys}")

    normalized_clusters: list[list[tuple[int, int]]] = []
    for raw_cluster in raw_clusters:
        raw_mentions = [raw_cluster] if _is_offset_pair(raw_cluster) else list(raw_cluster)
        normalized_mentions: list[tuple[int, int]] = []
        for raw_mention in raw_mentions:
            if isinstance(raw_mention, dict):
                start = raw_mention.get("start", raw_mention.get("start_token"))
                end = raw_mention.get("end", raw_mention.get("end_token"))
                if start is None or end is None:
                    raise ValueError(f"Invalid Maverick mention offset dict: {raw_mention!r}")
                normalized_mentions.append((int(start), int(end)))
                continue
            if not _is_offset_pair(raw_mention):
                raise ValueError(f"Invalid Maverick mention offset. Expected a pair of ints, got: {raw_mention!r}")
            normalized_mentions.append((int(raw_mention[0]), int(raw_mention[1])))
        normalized_clusters.append(normalized_mentions)
    return normalized_clusters


def _extract_clusters_text_mentions(prediction: dict[str, Any]) -> list[list[str]]:
    if not isinstance(prediction, dict):
        return []
    for key in ("clusters_text_mentions", "clusters_token_text", "clusters_char_text"):
        value = prediction.get(key)
        if value is not None:
            return [list(map(str, cluster)) for cluster in value]
    return []


def _is_offset_pair(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(item, int) for item in value)


def _safe_span_head_i(span: Span) -> int | None:
    if len(span) == 0:
        return None
    try:
        return span.root.i
    except Exception:
        return span[-1].i


def _choose_canonical_name_from_extracted(mentions: Iterable[ExtractedMention]) -> str | None:
    materialized = list(mentions)
    if not materialized:
        return None
    counts = Counter(_canonical_key(m.text) for m in materialized)

    def score(mention: ExtractedMention) -> tuple[int, int, int, int, int, str]:
        text = mention.text.strip()
        key = _canonical_key(text)
        clean_score = 100 if _is_clean_canonical_text(text) else 0
        non_pronoun_score = 50 if key not in BAD_CANONICAL_PRONOUNS else -200
        freq_score = counts[key] * 20
        token_len = max(1, mention.local_end - mention.local_start)
        compact_score = -abs(token_len - 2) * 3
        char_penalty = -max(0, len(text) - 40)
        return (clean_score, non_pronoun_score, freq_score, compact_score, char_penalty, text)

    best = max(materialized, key=score)
    return best.text


def _canonical_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _is_clean_canonical_text(text: str) -> bool:
    if not text or "\n" in text or '"' in text:
        return False
    if any(mark in text for mark in [",", ";", ":", "?", "!"]):
        return False
    if len(text.split()) > 8:
        return False
    return True


def _texts_match_loosely(left: str, right: str) -> bool:
    return _canonical_key(left) == _canonical_key(right)


def _is_cuda_device(device: str | None) -> bool:
    return str(device or "").startswith("cuda")


def _default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _empty_cuda_cache_if_available() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_p100_runtime() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and "P100" in torch.cuda.get_device_name(0).upper()
    except Exception:
        return False


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _looks_like_dtype_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    patterns = (
        "expected scalar type half",
        "expected scalar type float",
        "found half",
        "found float",
        "mat1 and mat2 must have the same dtype",
        "input type",
        "weight type",
        "dtype",
        "half",
        "float",
    )
    return any(pattern in message for pattern in patterns)


def _patch_transformers_embedding_resize_mean_resizing(*, verbose: bool = False) -> None:
    """Force Transformers embedding resize to avoid covariance-based mean resizing."""
    try:
        import inspect
        from transformers.modeling_utils import PreTrainedModel
    except Exception as exc:
        if verbose:
            print(f"[coref][warning] Could not patch Transformers embedding resize: {exc!r}")
        return

    original = PreTrainedModel.resize_token_embeddings
    if getattr(original, "_maverick_no_mean_resizing_patch", False):
        if verbose:
            print("[coref] Transformers resize_token_embeddings patch already active.")
        return

    try:
        signature = inspect.signature(original)
    except Exception:
        signature = None

    if signature is not None and "mean_resizing" not in signature.parameters:
        if verbose:
            print("[coref] Transformers resize_token_embeddings has no mean_resizing parameter; no patch needed.")
        return

    def resize_token_embeddings_no_mean_resizing(self, *args, **kwargs):
        kwargs["mean_resizing"] = False
        return original(self, *args, **kwargs)

    resize_token_embeddings_no_mean_resizing._maverick_no_mean_resizing_patch = True
    PreTrainedModel.resize_token_embeddings = resize_token_embeddings_no_mean_resizing

    if verbose:
        print("[coref] Patched Transformers resize_token_embeddings(mean_resizing=False).")
