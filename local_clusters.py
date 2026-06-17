from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover
    raise ImportError("local_clusters.py requires spaCy. Install it with: pip install spacy") from exc

__all__ = [
    "CHUNKS_COLUMNS",
    "CLUSTERS_COLUMNS",
    "MENTIONS_COLUMNS",
    "LocalClusterChunkRows",
    "LocalClustersTables",
    "create_local_clusters_metadata",
    "create_local_clusters_tables",
    "create_local_clusters_tables_from_plan",
    "chunk_specs_to_rows",
    "extracted_chunk_to_rows",
    "add_extracted_chunk_clusters",
    "validate_chunk_rows",
    "validate_local_clusters_tables",
    "refresh_metadata_counts",
    "make_local_cluster_uid",
    "make_local_mention_uid",
    "make_overlap_exact_span_key",
    "classify_mention_zone",
]

ValidationMode = Literal["basic", "debug"]

CHUNKS_COLUMNS = [
    "chunk_id",
    "chunk_index",
    "global_start",
    "global_end",
    "core_start",
    "core_end",
    "left_overlap_start",
    "left_overlap_end",
    "right_overlap_start",
    "right_overlap_end",
    "n_tokens",
    "sentence_start",
    "sentence_end",
]

CLUSTERS_COLUMNS = [
    "local_cluster_uid",
    "chunk_id",
    "chunk_index",
    "local_cluster_id",
    "canonical_name",
    "n_mentions",
]

MENTIONS_COLUMNS = [
    "local_mention_uid",
    "local_cluster_uid",
    "chunk_id",
    "chunk_index",
    "local_cluster_id",
    "local_mention_id",
    "local_start",
    "local_end",
    "global_start",
    "global_end",
    "text",
    "head_local_i",
    "head_global_i",
    "zone",
    "overlap_exact_span_key",
]


@dataclass(slots=True)
class LocalClusterChunkRows:
    """Rows produced from one extracted chunk before streaming to CSV."""

    chunk_id: str
    clusters_rows: list[dict[str, Any]] = field(default_factory=list)
    mentions_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class LocalClustersTables:
    """In-memory tabular artifact representation.

    This remains useful for local/debug runs and for loading the final zip. The
    marathon-safe path should commit per-chunk shards through
    artifact_io.LocalClusterMarathonStore.
    """

    metadata: dict[str, Any]
    chunks_rows: list[dict[str, Any]] = field(default_factory=list)
    clusters_rows: list[dict[str, Any]] = field(default_factory=list)
    mentions_rows: list[dict[str, Any]] = field(default_factory=list)


def create_local_clusters_metadata(
    *,
    doc: Doc,
    chunk_count: int,
    chunk_size: int | None,
    overlap_sentences: int | None,
    max_expanded_chunk_tokens: int | None,
    maverick_config: dict[str, Any],
    document_id: str | None = None,
) -> dict[str, Any]:
    """Create metadata for the local Maverick artifact."""
    if document_id is None:
        document_id = "document"

    return {
        "artifact_type": "local_clusters",
        "schema_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document": {
            "document_id": document_id,
            "text_hash": _sha256_text(doc.text),
            "token_count": len(doc),
            "sentence_count": _safe_sentence_count(doc),
        },
        "offsets": {
            "unit": "spacy_token_index",
            "global_span_convention": "half_open",
            "local_span_convention": "half_open",
            "includes_space_tokens": True,
        },
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_size_semantics": "expanded_chunk_tokens_including_overlap",
            "max_expanded_chunk_tokens": max_expanded_chunk_tokens,
            "overlap_sentences": overlap_sentences,
            "chunk_unit": "sentence_aligned_spacy_tokens",
            "sentence_aligned": True,
            "n_chunks": chunk_count,
        },
        "maverick": dict(maverick_config),
        "matching_contract": {
            "local_cluster_identity": ["chunk_id", "local_cluster_id"],
            "local_mention_identity": ["chunk_id", "local_cluster_id", "local_mention_id"],
            "allow_same_chunk_merge": False,
            "local_clusters_are_temporary": True,
            "local_mentions_are_preserved": True,
            "distant_chunk_matching_required": True,
        },
        "files": {},
        "stats": {},
    }


def create_local_clusters_tables(
    *,
    doc: Doc,
    chunks: Sequence[Any],
    maverick_config: dict[str, Any],
    document_id: str | None = None,
) -> LocalClustersTables:
    """Create in-memory local-cluster tables from already materialized chunks/specs."""
    chunk_size = maverick_config.get("chunk_size")
    overlap_sentences = maverick_config.get("overlap_sentences")
    max_expanded = maverick_config.get("max_expanded_chunk_tokens", chunk_size)
    metadata = create_local_clusters_metadata(
        doc=doc,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        overlap_sentences=overlap_sentences,
        max_expanded_chunk_tokens=max_expanded,
        maverick_config=maverick_config,
        document_id=document_id,
    )
    tables = LocalClustersTables(metadata=metadata, chunks_rows=chunk_specs_to_rows(chunks))
    refresh_metadata_counts(tables)
    return tables


def create_local_clusters_tables_from_plan(
    *,
    doc: Doc,
    chunk_plan: Any,
    maverick_config: dict[str, Any],
    document_id: str | None = None,
) -> LocalClustersTables:
    """Create in-memory tables from a lightweight ChunkPlan."""
    return create_local_clusters_tables(
        doc=doc,
        chunks=list(getattr(chunk_plan, "specs", chunk_plan)),
        maverick_config={
            **dict(maverick_config),
            "chunk_size": getattr(chunk_plan, "chunk_size", maverick_config.get("chunk_size")),
            "overlap_sentences": getattr(chunk_plan, "overlap_sentences", maverick_config.get("overlap_sentences")),
            "max_expanded_chunk_tokens": getattr(
                chunk_plan,
                "max_expanded_chunk_tokens",
                maverick_config.get("max_expanded_chunk_tokens", maverick_config.get("chunk_size")),
            ),
        },
        document_id=document_id,
    )


def chunk_specs_to_rows(chunks_or_specs: Sequence[Any]) -> list[dict[str, Any]]:
    return [_chunk_to_row(chunk) for chunk in chunks_or_specs]


def extracted_chunk_to_rows(*, chunk: Any, extracted: Any) -> LocalClusterChunkRows:
    """Convert one ExtractedChunkClusters result into CSV-ready rows."""
    temp_tables = LocalClustersTables(metadata={})
    add_extracted_chunk_clusters(tables=temp_tables, chunk=chunk, extracted=extracted)
    return LocalClusterChunkRows(
        chunk_id=chunk.chunk_id,
        clusters_rows=temp_tables.clusters_rows,
        mentions_rows=temp_tables.mentions_rows,
    )


def add_extracted_chunk_clusters(
    *,
    tables: LocalClustersTables,
    chunk: Any,
    extracted: Any,
) -> None:
    """Add one ExtractedChunkClusters result to normalized local tables."""
    rows = extracted_chunk_to_rows_without_recursion(chunk=chunk, extracted=extracted)
    tables.clusters_rows.extend(rows.clusters_rows)
    tables.mentions_rows.extend(rows.mentions_rows)
    if tables.metadata:
        refresh_metadata_counts(tables)


def extracted_chunk_to_rows_without_recursion(*, chunk: Any, extracted: Any) -> LocalClusterChunkRows:
    extracted_chunk_id = getattr(extracted, "chunk_id", None)
    if extracted_chunk_id is not None and extracted_chunk_id != chunk.chunk_id:
        raise ValueError(
            f"Extracted chunk id mismatch: extracted={extracted_chunk_id!r}, chunk={chunk.chunk_id!r}"
        )

    clusters_rows: list[dict[str, Any]] = []
    mentions_rows: list[dict[str, Any]] = []

    for cluster in getattr(extracted, "clusters", []):
        local_cluster_id = int(getattr(cluster, "local_cluster_id"))
        local_cluster_uid = make_local_cluster_uid(chunk.chunk_id, local_cluster_id)
        mention_uids: list[str] = []

        for mention in getattr(cluster, "mentions", []):
            local_mention_id = int(getattr(mention, "local_mention_id"))
            local_start = int(getattr(mention, "local_start"))
            local_end = int(getattr(mention, "local_end"))
            if local_start < 0 or local_end <= local_start:
                continue
            if local_start not in chunk.local_to_global or (local_end - 1) not in chunk.local_to_global:
                raise ValueError(
                    f"Mention local span [{local_start}, {local_end}) is outside {chunk.chunk_id}."
                )

            global_start = int(chunk.local_to_global[local_start])
            global_end = int(chunk.local_to_global[local_end - 1]) + 1
            head_local_i = getattr(mention, "head_local_i", None)
            head_global_i = None
            if head_local_i is not None and int(head_local_i) in chunk.local_to_global:
                head_global_i = int(chunk.local_to_global[int(head_local_i)])

            text = str(getattr(mention, "text", ""))
            zone = classify_mention_zone(chunk, global_start, global_end)
            overlap_exact_span_key = make_overlap_exact_span_key(
                global_start=global_start,
                global_end=global_end,
                text=text,
                zone=zone,
            )

            local_mention_uid = make_local_mention_uid(
                chunk.chunk_id,
                local_cluster_id,
                local_mention_id,
            )
            mention_uids.append(local_mention_uid)

            mentions_rows.append(
                {
                    "local_mention_uid": local_mention_uid,
                    "local_cluster_uid": local_cluster_uid,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "local_cluster_id": local_cluster_id,
                    "local_mention_id": local_mention_id,
                    "local_start": local_start,
                    "local_end": local_end,
                    "global_start": global_start,
                    "global_end": global_end,
                    "text": text,
                    "head_local_i": _empty_if_none(head_local_i),
                    "head_global_i": _empty_if_none(head_global_i),
                    "zone": zone,
                    "overlap_exact_span_key": overlap_exact_span_key,
                }
            )

        if not mention_uids:
            continue

        clusters_rows.append(
            {
                "local_cluster_uid": local_cluster_uid,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "local_cluster_id": local_cluster_id,
                "canonical_name": _empty_if_none(getattr(cluster, "canonical_name", None)),
                "n_mentions": len(mention_uids),
            }
        )

    return LocalClusterChunkRows(
        chunk_id=chunk.chunk_id,
        clusters_rows=clusters_rows,
        mentions_rows=mentions_rows,
    )


def make_local_cluster_uid(chunk_id: str, local_cluster_id: int) -> str:
    return f"{chunk_id}::cluster_{int(local_cluster_id):03d}"


def make_local_mention_uid(chunk_id: str, local_cluster_id: int, local_mention_id: int) -> str:
    return f"{make_local_cluster_uid(chunk_id, local_cluster_id)}::mention_{int(local_mention_id):03d}"


def classify_mention_zone(chunk: Any, global_start: int, global_end: int) -> str:
    """Classify a mention span against a chunk's core/overlap boundaries."""
    if global_start >= chunk.core_start and global_end <= chunk.core_end:
        return "core"
    if (
        chunk.left_overlap_start < chunk.left_overlap_end
        and global_start >= chunk.left_overlap_start
        and global_end <= chunk.left_overlap_end
    ):
        return "left_overlap"
    if (
        chunk.right_overlap_start < chunk.right_overlap_end
        and global_start >= chunk.right_overlap_start
        and global_end <= chunk.right_overlap_end
    ):
        return "right_overlap"
    return "cross_zone"


def make_overlap_exact_span_key(
    *,
    global_start: int,
    global_end: int,
    text: str,
    zone: str,
) -> str:
    if zone == "core":
        return ""
    return f"{int(global_start)}:{int(global_end)}:{_normalize_text(text)}"


def validate_chunk_rows(
    *,
    doc: Doc,
    chunk: Any,
    rows: LocalClusterChunkRows,
    mode: ValidationMode = "basic",
) -> dict[str, Any]:
    """Validate one chunk's rows before streaming them to CSV."""
    tables = LocalClustersTables(
        metadata={},
        chunks_rows=[_chunk_to_row(chunk)],
        clusters_rows=list(rows.clusters_rows),
        mentions_rows=list(rows.mentions_rows),
    )
    report = validate_local_clusters_tables(doc=doc, tables=tables, mode=mode)
    report["chunk_id"] = chunk.chunk_id
    return report


def validate_local_clusters_tables(
    *,
    doc: Doc,
    tables: LocalClustersTables,
    mode: ValidationMode = "basic",
) -> dict[str, Any]:
    """Validate normalized local-cluster tables.

    mode='basic' checks schema, IDs, foreign keys, and numeric offset ranges.
    mode='debug' additionally validates text spans, zone consistency, and overlap keys.
    Raises ValueError on validation failure and returns diagnostics otherwise.
    """
    if mode not in {"basic", "debug"}:
        raise ValueError("validation mode must be either 'basic' or 'debug'.")

    errors: list[str] = []
    warnings: list[str] = []

    _check_required_columns("chunks", tables.chunks_rows, CHUNKS_COLUMNS, errors)
    _check_required_columns("clusters", tables.clusters_rows, CLUSTERS_COLUMNS, errors)
    _check_required_columns("mentions", tables.mentions_rows, MENTIONS_COLUMNS, errors)

    chunk_ids = _unique_values(tables.chunks_rows, "chunk_id", errors, "chunks")
    cluster_uids = _unique_values(tables.clusters_rows, "local_cluster_uid", errors, "clusters")
    _unique_values(tables.mentions_rows, "local_mention_uid", errors, "mentions")

    for row in tables.clusters_rows:
        if row.get("chunk_id") not in chunk_ids:
            errors.append(f"Cluster {row.get('local_cluster_uid')} references missing chunk_id={row.get('chunk_id')!r}.")

    for row in tables.mentions_rows:
        if row.get("local_cluster_uid") not in cluster_uids:
            errors.append(
                f"Mention {row.get('local_mention_uid')} references missing local_cluster_uid="
                f"{row.get('local_cluster_uid')!r}."
            )
        if row.get("chunk_id") not in chunk_ids:
            errors.append(f"Mention {row.get('local_mention_uid')} references missing chunk_id={row.get('chunk_id')!r}.")
        _validate_span(row, "local", errors)
        _validate_span(row, "global", errors, max_end=len(doc))

    if mode == "debug":
        chunks_by_id = {row["chunk_id"]: row for row in tables.chunks_rows}
        for row in tables.mentions_rows:
            try:
                global_start = int(row["global_start"])
                global_end = int(row["global_end"])
            except Exception:
                continue
            span_text = doc[global_start:global_end].text
            if _normalize_text(span_text) != _normalize_text(str(row.get("text", ""))):
                warnings.append(
                    f"Text mismatch for {row.get('local_mention_uid')}: "
                    f"doc_span={span_text!r}, mention_text={row.get('text')!r}."
                )

            chunk_row = chunks_by_id.get(row.get("chunk_id"))
            if chunk_row is not None:
                expected_zone = _classify_zone_from_chunk_row(chunk_row, global_start, global_end)
                if row.get("zone") != expected_zone:
                    errors.append(
                        f"Zone mismatch for {row.get('local_mention_uid')}: "
                        f"stored={row.get('zone')!r}, expected={expected_zone!r}."
                    )

            expected_key = make_overlap_exact_span_key(
                global_start=global_start,
                global_end=global_end,
                text=str(row.get("text", "")),
                zone=str(row.get("zone", "")),
            )
            if row.get("overlap_exact_span_key", "") != expected_key:
                errors.append(f"overlap_exact_span_key mismatch for {row.get('local_mention_uid')}.")

    if tables.metadata:
        refresh_metadata_counts(tables)

    if errors:
        message = "Local cluster table validation failed:\n" + "\n".join(f"- {e}" for e in errors[:50])
        if len(errors) > 50:
            message += f"\n... and {len(errors) - 50} more errors."
        raise ValueError(message)

    return {
        "mode": mode,
        "errors": errors,
        "warnings": warnings,
        "n_chunks": len(tables.chunks_rows),
        "n_clusters": len(tables.clusters_rows),
        "n_mentions": len(tables.mentions_rows),
    }


def refresh_metadata_counts(tables: LocalClustersTables) -> None:
    """Refresh metadata row counts/statistics from current in-memory tables."""
    n_overlap_mentions = sum(1 for row in tables.mentions_rows if row.get("zone") != "core")
    tables.metadata.setdefault("chunking", {})["n_chunks"] = len(tables.chunks_rows)
    tables.metadata["files"] = {
        "chunks.csv": {"rows": len(tables.chunks_rows)},
        "clusters.csv": {"rows": len(tables.clusters_rows)},
        "mentions.csv": {"rows": len(tables.mentions_rows)},
    }
    tables.metadata["stats"] = {
        "n_chunks": len(tables.chunks_rows),
        "n_local_clusters": len(tables.clusters_rows),
        "n_local_mentions": len(tables.mentions_rows),
        "n_overlap_mentions": n_overlap_mentions,
    }


def _chunk_to_row(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "global_start": chunk.global_start,
        "global_end": chunk.global_end,
        "core_start": chunk.core_start,
        "core_end": chunk.core_end,
        "left_overlap_start": chunk.left_overlap_start,
        "left_overlap_end": chunk.left_overlap_end,
        "right_overlap_start": chunk.right_overlap_start,
        "right_overlap_end": chunk.right_overlap_end,
        "n_tokens": chunk.n_tokens,
        "sentence_start": chunk.sentence_start,
        "sentence_end": chunk.sentence_end,
    }


def _classify_zone_from_chunk_row(chunk: dict[str, Any], global_start: int, global_end: int) -> str:
    core_start = int(chunk["core_start"])
    core_end = int(chunk["core_end"])
    lo_start = int(chunk["left_overlap_start"])
    lo_end = int(chunk["left_overlap_end"])
    ro_start = int(chunk["right_overlap_start"])
    ro_end = int(chunk["right_overlap_end"])
    if global_start >= core_start and global_end <= core_end:
        return "core"
    if lo_start < lo_end and global_start >= lo_start and global_end <= lo_end:
        return "left_overlap"
    if ro_start < ro_end and global_start >= ro_start and global_end <= ro_end:
        return "right_overlap"
    return "cross_zone"


def _check_required_columns(name: str, rows: list[dict[str, Any]], required: Sequence[str], errors: list[str]) -> None:
    if not rows:
        return
    available = set(rows[0].keys())
    missing = [col for col in required if col not in available]
    if missing:
        errors.append(f"{name} rows missing required columns: {missing}")


def _unique_values(rows: list[dict[str, Any]], key: str, errors: list[str], table_name: str) -> set[Any]:
    values: list[Any] = [row.get(key) for row in rows]
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        preview = sorted(map(str, duplicates))[:10]
        errors.append(f"{table_name}.{key} contains duplicates: {preview}")
    return seen


def _validate_span(row: dict[str, Any], prefix: str, errors: list[str], max_end: int | None = None) -> None:
    try:
        start = int(row[f"{prefix}_start"])
        end = int(row[f"{prefix}_end"])
    except Exception:
        errors.append(f"Invalid {prefix} span values in row {row!r}.")
        return
    if start < 0 or end <= start:
        errors.append(f"Invalid {prefix} span [{start}, {end}) in {row.get('local_mention_uid')}.")
    if max_end is not None and end > max_end:
        errors.append(f"{prefix} span [{start}, {end}) exceeds max_end={max_end} in {row.get('local_mention_uid')}")


def _safe_sentence_count(doc: Doc) -> int:
    try:
        return sum(1 for _ in doc.sents)
    except Exception:
        return 0


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value
