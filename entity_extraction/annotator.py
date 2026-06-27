from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "entity_extraction.annotator requires spaCy. Install it with: pip install spacy"
    ) from exc

from annotation_layer.entities import EntityClusterRecord, EntityMentionRecord, EntitySubLayer
from annotation_layer.spacy_extension import ensure_annotation_layer


__all__ = [
    "EntityAnnotationError",
    "annotate_doc_with_global_entities",
]


GLOBAL_CLUSTER_REQUIRED_COLUMNS = {
    "global_cluster_uid",
    "canonical_name",
}

GLOBAL_MENTION_REQUIRED_COLUMNS = {
    "global_cluster_uid",
    "global_start",
    "global_end",
    "text",
    "head_global_i",
}

_GLOBAL_CLUSTER_UID_RE = re.compile(r"^global_cluster_(\d+)$")


class EntityAnnotationError(ValueError):
    """Raised when global entity artifacts cannot annotate the given Doc."""


def annotate_doc_with_global_entities(
    *,
    doc: Doc,
    clusters_csv: str | Path,
    mentions_csv: str | Path,
    document_id: str,
    verbose: bool = False,
    overwrite: bool = False,
) -> Doc:
    """
    Attach the final entity sub-layer to a BookNLP-annotated spaCy Doc.

    The function returns the same Doc object, mutated in place:
        doc._.annotation_layer.entities = EntitySubLayer(...)

    Policy:
    - cluster_id is parsed from global_cluster_uid, e.g. global_cluster_000008 -> 8.
    - mention_id is assigned sequentially after deduplication.
    - mentions are deduplicated by (cluster_id, global_start, global_end).
    - a mention row survives only if row['text'] == doc[start:end].text.
    - same exact span across different clusters is kept, with a verbose warning.
    - clusters with zero final mentions are dropped.
    """
    ann = ensure_annotation_layer(doc, document_id=document_id)
    if ann.entities is not None and not overwrite:
        raise EntityAnnotationError(
            "Doc already has entity annotations. Pass overwrite=True to replace them."
        )

    _require_booknlp_entity_doc(doc)

    clusters_df = pd.read_csv(clusters_csv)
    mentions_df = pd.read_csv(mentions_csv)

    _validate_required_columns(
        clusters_df=clusters_df,
        mentions_df=mentions_df,
    )

    clusters_df = _prepare_clusters_dataframe(clusters_df)
    mentions_df = _prepare_mentions_dataframe(mentions_df)

    _validate_cluster_references(
        cluster_ids=set(clusters_df["cluster_id"].tolist()),
        mentions_df=mentions_df,
    )
    _validate_mention_offsets(doc=doc, mentions_df=mentions_df)

    candidate_rows = _deduplicate_mentions_against_doc(
        doc=doc,
        mentions_df=mentions_df,
        verbose=verbose,
    )

    mentions = _build_mentions(candidate_rows)
    clusters = _build_clusters(
        clusters_df=clusters_df,
        mentions=mentions,
        verbose=verbose,
    )

    mentions = {
        mention_id: mention
        for mention_id, mention in mentions.items()
        if mention.cluster_id in clusters
    }

    _warn_same_span_across_clusters(mentions=mentions, verbose=verbose)

    layer = EntitySubLayer.from_data(
        mentions=mentions,
        clusters=clusters,
    )

    ann.attach_entities(layer, overwrite=overwrite)

    if verbose:
        _log(f"[entity-annotator] summary: {layer.summary()}")

    return doc


def _require_booknlp_entity_doc(doc: Doc) -> None:
    if not Doc.has_extension("booknlp_annotated") or not doc._.booknlp_annotated:
        raise EntityAnnotationError(
            "Doc is not BookNLP-annotated. global_start/global_end may not "
            "match this tokenization."
        )


def _validate_required_columns(
    *,
    clusters_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
) -> None:
    missing_cluster_cols = sorted(GLOBAL_CLUSTER_REQUIRED_COLUMNS - set(clusters_df.columns))
    if missing_cluster_cols:
        raise EntityAnnotationError(
            f"global_clusters.csv is missing required columns: {missing_cluster_cols}"
        )

    missing_mention_cols = sorted(GLOBAL_MENTION_REQUIRED_COLUMNS - set(mentions_df.columns))
    if missing_mention_cols:
        raise EntityAnnotationError(
            f"global_mentions.csv is missing required columns: {missing_mention_cols}"
        )


def _prepare_clusters_dataframe(clusters_df: pd.DataFrame) -> pd.DataFrame:
    df = clusters_df.copy()
    df["cluster_id"] = df["global_cluster_uid"].map(parse_cluster_id)

    empty_name_mask = df["canonical_name"].isna() | (
        df["canonical_name"].astype(str).str.strip() == ""
    )
    if empty_name_mask.any():
        bad_ids = df.loc[empty_name_mask, "global_cluster_uid"].head(10).tolist()
        raise EntityAnnotationError(
            "global_clusters.csv contains empty canonical_name values. "
            f"Examples: {bad_ids}"
        )

    duplicate_ids = df.loc[df["cluster_id"].duplicated(), "cluster_id"].unique().tolist()
    if duplicate_ids:
        raise EntityAnnotationError(
            f"global_clusters.csv contains duplicate cluster IDs: {duplicate_ids[:10]}"
        )

    return df


def _prepare_mentions_dataframe(mentions_df: pd.DataFrame) -> pd.DataFrame:
    df = mentions_df.copy()
    df["cluster_id"] = df["global_cluster_uid"].map(parse_cluster_id)
    df["global_start"] = df["global_start"].map(_require_int)
    df["global_end"] = df["global_end"].map(_require_int)
    df["head_token_i"] = df["head_global_i"].map(_optional_int)
    df["text"] = df["text"].astype(str)
    return df


def parse_cluster_id(global_cluster_uid: Any) -> int:
    match = _GLOBAL_CLUSTER_UID_RE.match(str(global_cluster_uid))
    if not match:
        raise EntityAnnotationError(f"Invalid global_cluster_uid: {global_cluster_uid!r}")
    return int(match.group(1))


def _require_int(value: Any) -> int:
    if pd.isna(value):
        raise EntityAnnotationError("Required integer value is missing.")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise EntityAnnotationError(f"Expected integer-compatible value, got {value!r}") from exc
    if not as_float.is_integer():
        raise EntityAnnotationError(f"Expected integer-compatible value, got {value!r}")
    return int(as_float)


def _optional_int(value: Any) -> int | None:
    if pd.isna(value) or value == "":
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not as_float.is_integer():
        return None
    return int(as_float)


def _validate_cluster_references(
    *,
    cluster_ids: set[int],
    mentions_df: pd.DataFrame,
) -> None:
    referenced_ids = set(mentions_df["cluster_id"].tolist())
    missing_ids = sorted(referenced_ids - cluster_ids)
    if missing_ids:
        raise EntityAnnotationError(
            "global_mentions.csv references cluster IDs missing from "
            f"global_clusters.csv: {missing_ids[:20]}"
        )


def _validate_mention_offsets(*, doc: Doc, mentions_df: pd.DataFrame) -> None:
    invalid_mask = (
        (mentions_df["global_start"] < 0)
        | (mentions_df["global_end"] <= mentions_df["global_start"])
        | (mentions_df["global_end"] > len(doc))
    )
    if invalid_mask.any():
        examples = mentions_df.loc[
            invalid_mask,
            ["global_cluster_uid", "global_start", "global_end", "text"],
        ].head(10)
        raise EntityAnnotationError(
            "global_mentions.csv contains offsets incompatible with the Doc. "
            f"Doc length={len(doc)}. Examples:\n{examples.to_string(index=False)}"
        )


def _deduplicate_mentions_against_doc(
    *,
    doc: Doc,
    mentions_df: pd.DataFrame,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Return final mention candidates before sequential mention_id assignment."""
    rows: list[dict[str, Any]] = []
    dropped_text_mismatch_groups = 0
    duplicate_groups_collapsed = 0
    invalid_heads = 0

    group_cols = ["cluster_id", "global_start", "global_end"]
    sorted_df = mentions_df.sort_values(
        ["global_start", "global_end", "cluster_id", "text"],
        kind="mergesort",
    )

    for (cluster_id, start, end), group in sorted_df.groupby(group_cols, sort=False):
        doc_text = doc[int(start) : int(end)].text
        matching = group[group["text"] == doc_text]

        if matching.empty:
            dropped_text_mismatch_groups += 1
            if verbose and dropped_text_mismatch_groups <= 10:
                csv_texts = sorted(set(group["text"].astype(str).tolist()))
                _log(
                    "[entity-annotator] warning: dropped mention group because "
                    "no CSV text matched doc span: "
                    f"cluster_id={cluster_id}, span=({start}, {end}), "
                    f"doc_text={doc_text!r}, csv_texts={csv_texts[:5]!r}"
                )
            continue

        if len(group) > 1:
            duplicate_groups_collapsed += 1

        head_token_i = _select_valid_head_token_i(
            start=int(start),
            end=int(end),
            candidate_heads=matching["head_token_i"].tolist(),
        )
        if head_token_i is None and any(pd.notna(x) for x in matching["head_token_i"].tolist()):
            invalid_heads += 1

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "start": int(start),
                "end": int(end),
                "text": doc_text,
                "head_token_i": head_token_i,
            }
        )

    if verbose:
        if dropped_text_mismatch_groups:
            _log(
                "[entity-annotator] warning: dropped "
                f"{dropped_text_mismatch_groups} mention group(s) because CSV text "
                "did not match doc[start:end].text"
            )
        if duplicate_groups_collapsed:
            _log(
                "[entity-annotator] collapsed "
                f"{duplicate_groups_collapsed} duplicate mention group(s) by "
                "(cluster_id, global_start, global_end)"
            )
        if invalid_heads:
            _log(
                "[entity-annotator] warning: set head_token_i=None for "
                f"{invalid_heads} mention group(s) with no valid in-span head"
            )

    return rows


def _select_valid_head_token_i(
    *,
    start: int,
    end: int,
    candidate_heads: list[Any],
) -> int | None:
    for raw_head in candidate_heads:
        head = _optional_int(raw_head)
        if head is None:
            continue
        if start <= head < end:
            return head
    return None


def _build_mentions(candidate_rows: list[dict[str, Any]]) -> dict[int, EntityMentionRecord]:
    sorted_rows = sorted(
        candidate_rows,
        key=lambda row: (row["start"], row["end"], row["cluster_id"], row["text"]),
    )

    mentions: dict[int, EntityMentionRecord] = {}
    for mention_id, row in enumerate(sorted_rows):
        mentions[mention_id] = EntityMentionRecord(
            mention_id=mention_id,
            cluster_id=row["cluster_id"],
            start=row["start"],
            end=row["end"],
            text=row["text"],
            head_token_i=row["head_token_i"],
        )

    return mentions


def _build_clusters(
    *,
    clusters_df: pd.DataFrame,
    mentions: dict[int, EntityMentionRecord],
    verbose: bool,
) -> dict[int, EntityClusterRecord]:
    mention_ids_by_cluster: dict[int, list[int]] = {}
    for mention in mentions.values():
        mention_ids_by_cluster.setdefault(mention.cluster_id, []).append(mention.mention_id)

    clusters: dict[int, EntityClusterRecord] = {}
    dropped_zero_mention_clusters = 0

    for row in clusters_df.sort_values("cluster_id", kind="mergesort").itertuples(index=False):
        cluster_id = int(row.cluster_id)
        mention_ids = mention_ids_by_cluster.get(cluster_id, [])
        if not mention_ids:
            dropped_zero_mention_clusters += 1
            continue

        mention_ids = sorted(
            mention_ids,
            key=lambda mention_id: (
                mentions[mention_id].start,
                mentions[mention_id].end,
                mention_id,
            ),
        )

        clusters[cluster_id] = EntityClusterRecord(
            cluster_id=cluster_id,
            mention_ids=tuple(mention_ids),
            canonical_name=str(row.canonical_name),
        )

    if verbose and dropped_zero_mention_clusters:
        _log(
            "[entity-annotator] dropped "
            f"{dropped_zero_mention_clusters} cluster(s) with zero final mentions"
        )

    return clusters


def _warn_same_span_across_clusters(
    *,
    mentions: dict[int, EntityMentionRecord],
    verbose: bool,
) -> None:
    if not verbose:
        return

    span_to_cluster_ids: dict[tuple[int, int], set[int]] = {}
    for mention in mentions.values():
        span_to_cluster_ids.setdefault((mention.start, mention.end), set()).add(mention.cluster_id)

    conflicts = [
        (span, sorted(cluster_ids))
        for span, cluster_ids in span_to_cluster_ids.items()
        if len(cluster_ids) > 1
    ]
    if not conflicts:
        return

    _log(
        "[entity-annotator] warning: "
        f"{len(conflicts)} exact span(s) are assigned to multiple clusters; kept all"
    )
    for span, cluster_ids in conflicts[:10]:
        _log(
            "[entity-annotator] warning: same span across clusters: "
            f"span={span}, clusters={cluster_ids}"
        )


def _log(message: str) -> None:
    print(message)
    sys.stdout.flush()
