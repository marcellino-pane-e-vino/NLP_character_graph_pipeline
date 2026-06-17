from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover
    raise ImportError("doc_chunker.py requires spaCy. Install it with: pip install spacy") from exc

__all__ = [
    "ChunkSpec",
    "ChunkPlan",
    "DocChunk",
    "DocChunker",
    "create_chunker",
]


@dataclass(slots=True, frozen=True)
class ChunkSpec:
    """Lightweight sentence-aligned chunk boundary record.

    This object is safe to keep for the whole document. It contains no spaCy Doc,
    no local/global maps, and no Maverick output.

    All offsets are global spaCy token offsets over the original document and use
    half-open intervals. ``global_start``/``global_end`` describe the expanded
    chunk that is sent to Maverick. Therefore ``n_tokens`` is the actual expanded
    chunk size, including overlap.
    """

    chunk_id: str
    chunk_index: int
    global_start: int
    global_end: int
    core_start: int
    core_end: int
    left_overlap_start: int
    left_overlap_end: int
    right_overlap_start: int
    right_overlap_end: int
    sentence_start: int
    sentence_end: int

    @property
    def n_tokens(self) -> int:
        return self.global_end - self.global_start


@dataclass(slots=True, frozen=True)
class ChunkPlan:
    """Cheap complete chunk plan for one document.

    ``chunk_size`` is the maximum expanded chunk size. If the user says 2000
    tokens, each materialized chunk must contain at most 2000 spaCy tokens total,
    including overlap.
    """

    chunk_size: int
    overlap_sentences: int
    max_expanded_chunk_tokens: int
    specs: tuple[ChunkSpec, ...]

    def __iter__(self) -> Iterator[ChunkSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


@dataclass(slots=True)
class DocChunk:
    """Runtime-only materialized view of one ChunkSpec.

    This is the expensive object: it contains the local spaCy Doc passed to
    Maverick plus local/global token maps. It should be created, processed, and
    deleted one chunk at a time in memory-constrained runs.
    """

    chunk_id: str
    chunk_index: int
    chunk_doc: Doc
    global_start: int
    global_end: int
    core_start: int
    core_end: int
    left_overlap_start: int
    left_overlap_end: int
    right_overlap_start: int
    right_overlap_end: int
    local_to_global: dict[int, int]
    global_to_local: dict[int, int]
    sentence_start: int
    sentence_end: int

    @property
    def n_tokens(self) -> int:
        return self.global_end - self.global_start


class DocChunker:
    """Create sentence-aligned chunks with sentence-based overlap.

    Contract:
        - chunk_size is the maximum expanded chunk size, including overlap.
        - overlap_sentences is expressed in sentences.
        - chunks are planned as cheap ChunkSpec objects first.
        - only materialize one DocChunk at a time in the main pipeline.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 8_000,
        overlap_sentences: int = 80,
        max_expanded_chunk_tokens: int | None = None,
        id_prefix: str = "chunk",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0.")
        if max_expanded_chunk_tokens is not None and max_expanded_chunk_tokens <= 0:
            raise ValueError("max_expanded_chunk_tokens must be > 0 when provided.")
        self.chunk_size = int(chunk_size)
        self.overlap_sentences = int(overlap_sentences)
        self.max_expanded_chunk_tokens = int(max_expanded_chunk_tokens or chunk_size)
        if self.max_expanded_chunk_tokens < self.chunk_size:
            raise ValueError("max_expanded_chunk_tokens cannot be smaller than chunk_size.")
        self.id_prefix = str(id_prefix)

    def plan(self, doc: Doc) -> ChunkPlan:
        """Return a lightweight chunk plan without materializing chunk Docs."""
        self._validate_doc(doc)
        sentences = list(doc.sents)
        if not sentences:
            raise ValueError("Cannot chunk a Doc without sentence boundaries.")

        specs: list[ChunkSpec] = []
        core_sent_start = 0
        chunk_index = 0
        n_sentences = len(sentences)

        while core_sent_start < n_sentences:
            best: ChunkSpec | None = None
            core_sent_end = core_sent_start + 1

            while core_sent_end <= n_sentences:
                spec = self._make_spec_from_core_sentence_range(
                    sentences=sentences,
                    chunk_index=chunk_index,
                    core_sent_start=core_sent_start,
                    core_sent_end=core_sent_end,
                )

                if spec.n_tokens <= self.max_expanded_chunk_tokens:
                    best = spec
                    core_sent_end += 1
                    continue

                # The first candidate already violates the hard limit. Because
                # the user explicitly requested hard safety limits, fail early
                # instead of silently reducing overlap.
                if best is None:
                    raise ValueError(
                        "Cannot create a sentence-aligned chunk under the expanded-token limit. "
                        f"chunk_index={chunk_index}, core_sentence={core_sent_start}, "
                        f"candidate_expanded_tokens={spec.n_tokens}, "
                        f"max_expanded_chunk_tokens={self.max_expanded_chunk_tokens}. "
                        "Reduce overlap_sentences, increase chunk_size, or inspect unusually long sentences."
                    )
                break

            if best is None:  # defensive; practically covered above
                raise RuntimeError(f"Failed to create chunk spec at sentence {core_sent_start}.")

            specs.append(best)
            core_sent_start = best.sentence_end
            chunk_index += 1

        plan = ChunkPlan(
            chunk_size=self.chunk_size,
            overlap_sentences=self.overlap_sentences,
            max_expanded_chunk_tokens=self.max_expanded_chunk_tokens,
            specs=tuple(specs),
        )
        self._validate_plan(doc, plan)
        return plan

    def materialize(self, doc: Doc, spec: ChunkSpec) -> DocChunk:
        """Create one memory-heavy runtime DocChunk from a lightweight ChunkSpec."""
        self._validate_doc(doc)
        if spec.n_tokens > self.max_expanded_chunk_tokens:
            raise ValueError(
                f"{spec.chunk_id} has {spec.n_tokens} expanded tokens, exceeding "
                f"max_expanded_chunk_tokens={self.max_expanded_chunk_tokens}."
            )

        chunk_doc = self._make_chunk_doc(doc, spec.global_start, spec.global_end)
        local_to_global = {
            local_i: global_i
            for local_i, global_i in enumerate(range(spec.global_start, spec.global_end))
        }
        global_to_local = {global_i: local_i for local_i, global_i in local_to_global.items()}

        chunk = DocChunk(
            chunk_id=spec.chunk_id,
            chunk_index=spec.chunk_index,
            chunk_doc=chunk_doc,
            global_start=spec.global_start,
            global_end=spec.global_end,
            core_start=spec.core_start,
            core_end=spec.core_end,
            left_overlap_start=spec.left_overlap_start,
            left_overlap_end=spec.left_overlap_end,
            right_overlap_start=spec.right_overlap_start,
            right_overlap_end=spec.right_overlap_end,
            local_to_global=local_to_global,
            global_to_local=global_to_local,
            sentence_start=spec.sentence_start,
            sentence_end=spec.sentence_end,
        )
        self._validate_materialized_chunk(doc, chunk)
        return chunk

    def iter_chunks(self, doc: Doc, plan: ChunkPlan | None = None) -> Iterator[DocChunk]:
        """Yield one materialized DocChunk at a time."""
        resolved_plan = plan or self.plan(doc)
        for spec in resolved_plan.specs:
            yield self.materialize(doc, spec)

    def chunk(self, doc: Doc) -> list[DocChunk]:
        """Compatibility/debug helper that materializes all chunks.

        Do not use this in memory-constrained or long-running full-document runs.
        Use plan(...)+materialize(...) or iter_chunks(...) instead.
        """
        plan = self.plan(doc)
        return [self.materialize(doc, spec) for spec in plan.specs]

    def _make_spec_from_core_sentence_range(
        self,
        *,
        sentences: Sequence,
        chunk_index: int,
        core_sent_start: int,
        core_sent_end: int,
    ) -> ChunkSpec:
        n_sentences = len(sentences)
        expanded_sent_start = max(0, core_sent_start - self.overlap_sentences)
        expanded_sent_end = min(n_sentences, core_sent_end + self.overlap_sentences)

        global_start = sentences[expanded_sent_start].start
        global_end = sentences[expanded_sent_end - 1].end
        core_start = sentences[core_sent_start].start
        core_end = sentences[core_sent_end - 1].end

        return ChunkSpec(
            chunk_id=f"{self.id_prefix}_{chunk_index:03d}",
            chunk_index=chunk_index,
            global_start=global_start,
            global_end=global_end,
            core_start=core_start,
            core_end=core_end,
            left_overlap_start=global_start,
            left_overlap_end=core_start,
            right_overlap_start=core_end,
            right_overlap_end=global_end,
            sentence_start=core_sent_start,
            sentence_end=core_sent_end,
        )

    def _make_chunk_doc(self, doc: Doc, start: int, end: int) -> Doc:
        span = doc[start:end]
        try:
            chunk_doc = span.as_doc(copy_user_data=True)
        except TypeError:  # older spaCy fallback
            chunk_doc = span.as_doc()

        if hasattr(doc._, "booknlp_annotated"):
            try:
                chunk_doc._.booknlp_annotated = bool(doc._.booknlp_annotated)
            except Exception:
                pass
        return chunk_doc

    def _validate_doc(self, doc: Doc) -> None:
        if not isinstance(doc, Doc):
            raise TypeError("DocChunker expects a spaCy Doc.")
        if not doc.has_annotation("SENT_START"):
            raise ValueError("Doc has no sentence boundaries. Tokenize it with tokenizer.py first.")

    def _validate_plan(self, doc: Doc, plan: ChunkPlan) -> None:
        if not plan.specs:
            raise ValueError("Chunking produced no chunk specs.")
        if plan.specs[0].core_start != 0:
            raise ValueError("First chunk core does not start at token 0.")
        if plan.specs[-1].core_end != len(doc):
            raise ValueError("Last chunk core does not end at len(doc).")

        previous_core_end = None
        for spec in plan.specs:
            if spec.n_tokens > plan.max_expanded_chunk_tokens:
                raise ValueError(
                    f"{spec.chunk_id} has {spec.n_tokens} expanded tokens, exceeding "
                    f"max_expanded_chunk_tokens={plan.max_expanded_chunk_tokens}."
                )
            if not (0 <= spec.global_start <= spec.core_start <= spec.core_end <= spec.global_end <= len(doc)):
                raise ValueError(f"Invalid boundaries for {spec.chunk_id}.")
            if not (spec.left_overlap_start == spec.global_start and spec.left_overlap_end == spec.core_start):
                raise ValueError(f"Invalid left overlap boundaries for {spec.chunk_id}.")
            if not (spec.right_overlap_start == spec.core_end and spec.right_overlap_end == spec.global_end):
                raise ValueError(f"Invalid right overlap boundaries for {spec.chunk_id}.")
            if previous_core_end is not None and previous_core_end != spec.core_start:
                raise ValueError(
                    f"Core chunk gap/overlap before {spec.chunk_id}: "
                    f"previous_core_end={previous_core_end}, core_start={spec.core_start}."
                )
            previous_core_end = spec.core_end

    def _validate_materialized_chunk(self, doc: Doc, chunk: DocChunk) -> None:
        if chunk.n_tokens > self.max_expanded_chunk_tokens:
            raise ValueError(f"{chunk.chunk_id} exceeds the expanded-token hard limit.")
        if len(chunk.chunk_doc) != chunk.n_tokens:
            raise ValueError(f"local/global token count mismatch for {chunk.chunk_id}.")
        if len(chunk.local_to_global) != len(chunk.chunk_doc):
            raise ValueError(f"local_to_global incomplete for {chunk.chunk_id}.")
        if chunk.global_start < 0 or chunk.global_end > len(doc):
            raise ValueError(f"{chunk.chunk_id} global boundaries are outside doc.")


def create_chunker(
    *,
    chunk_size: int = 8_000,
    overlap_sentences: int = 80,
    max_expanded_chunk_tokens: int | None = None,
    id_prefix: str = "chunk",
) -> DocChunker:
    """Factory used by the notebook to keep setup compact."""
    return DocChunker(
        chunk_size=chunk_size,
        overlap_sentences=overlap_sentences,
        max_expanded_chunk_tokens=max_expanded_chunk_tokens,
        id_prefix=id_prefix,
    )
