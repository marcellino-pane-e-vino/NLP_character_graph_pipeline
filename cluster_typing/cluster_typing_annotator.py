"""Cluster-typing JSONL annotator.

This module reads compact schema-v2 JSONL artifacts produced by
``cluster_typing_probability_scoring`` and creates a fresh
``doc._.cluster_typing_layer``.

The final document layer stores only the canonical ontology class IRI selected
for each cluster. Labels and other display forms are derived through
``OntologyClassGraph``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coreference.coref_schema import require_coref_layer
from ontology.class_graph import OntologyClassGraph

from .graph_contract import (
    STAY,
    VIRTUAL_ROOT,
    validate_selected_path_edge,
)
from .cluster_typing_artifacts import read_jsonl
from .cluster_typing_schema import (
    ClusterTypingAnnotation,
    ClusterTypingLayer,
    register_spacy_cluster_typing_extension,
)


__all__ = [
    "ClusterTypingAnnotationConfig",
    "ClusterTypingAnnotationError",
    "collapse_mention_weight",
    "annotate_doc_with_cluster_typing_folder",
]


SCHEMA_VERSION = 2


class ClusterTypingAnnotationError(ValueError):
    """Raised when cluster-typing JSONL artifacts cannot annotate the given Doc."""


@dataclass(frozen=True)
class ClusterTypingAnnotationConfig:
    """Configuration for rebuilding ``doc._.cluster_typing_layer`` from JSONL artifacts."""

    use_mention_weight: bool = True
    aggregation_method: str = "top_down_weighted_edge"
    rounding_digits: int = 6


def collapse_mention_weight(
    *,
    high: float,
    medium: float,
    low: float,
    rounding_digits: int = 6,
) -> float:
    """Collapse high/medium/low mention-weight probabilities into a 0-1 weight."""

    return round(1.0 * high + 0.5 * medium + 0.0 * low, rounding_digits)


def _require_single_cluster_id(records: list[dict[str, Any]], *, jsonl_path: Path) -> int:
    cluster_ids = sorted({int(record["cluster_id"]) for record in records})
    if len(cluster_ids) != 1:
        raise ClusterTypingAnnotationError(
            f"Each cluster-typing JSONL must contain exactly one cluster_id. "
            f"Found {cluster_ids} in {jsonl_path}"
        )
    return int(cluster_ids[0])


def _validate_schema_version(
    *,
    record: dict[str, Any],
    jsonl_path: Path,
    row_index: int,
) -> None:
    raw_version = record.get("schema_version")
    try:
        version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ClusterTypingAnnotationError(
            f"Cluster-typing JSONL record must have integer schema_version={SCHEMA_VERSION}. "
            f"Got {raw_version!r}. JSONL: {jsonl_path}, row={row_index}"
        ) from exc

    if version != SCHEMA_VERSION:
        raise ClusterTypingAnnotationError(
            f"Cluster-typing JSONL record must have schema_version={SCHEMA_VERSION}. "
            f"Got {raw_version!r}. JSONL: {jsonl_path}, row={row_index}"
        )


def _validate_record_alignment(
    *,
    record: dict[str, Any],
    coref_layer: Any,
    jsonl_path: Path,
    row_index: int,
) -> tuple[int, int]:
    _validate_schema_version(record=record, jsonl_path=jsonl_path, row_index=row_index)

    try:
        cluster_id = int(record["cluster_id"])
        mention_id = int(record["mention_id"])
        mention_start = int(record["mention_start"])
        mention_end = int(record["mention_end"])
        mention_text = str(record["mention_text"])
    except KeyError as exc:
        raise ClusterTypingAnnotationError(
            f"Cluster-typing JSONL record missing required identity field {exc.args[0]!r}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        ) from exc

    if cluster_id not in coref_layer.clusters:
        raise ClusterTypingAnnotationError(
            f"JSONL references unknown cluster_id={cluster_id}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    if mention_id not in coref_layer.mentions:
        raise ClusterTypingAnnotationError(
            f"JSONL references unknown mention_id={mention_id}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    mention = coref_layer.mentions[mention_id]

    if int(mention.cluster_id) != cluster_id:
        raise ClusterTypingAnnotationError(
            f"JSONL/coref cluster mismatch for mention_id={mention_id}: "
            f"JSONL cluster_id={cluster_id}, coref cluster_id={mention.cluster_id}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    if mention_start != int(mention.start) or mention_end != int(mention.end):
        raise ClusterTypingAnnotationError(
            f"JSONL/coref span mismatch for mention_id={mention_id}: "
            f"JSONL span=({mention_start}, {mention_end}), "
            f"coref span=({mention.start}, {mention.end}). "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    if mention_text != str(mention.text):
        raise ClusterTypingAnnotationError(
            f"JSONL/coref text mismatch for mention_id={mention_id}: "
            f"JSONL text={mention_text!r}, coref text={str(mention.text)!r}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    return mention_id, cluster_id


def _validate_selected_path(
    *,
    class_graph: OntologyClassGraph,
    record: dict[str, Any],
    jsonl_path: Path,
    row_index: int,
) -> list[dict[str, Any]]:
    selected_path = record.get("selected_path")
    if not isinstance(selected_path, list):
        raise ClusterTypingAnnotationError(
            f"JSONL record must contain selected_path as a list. "
            f"JSONL: {jsonl_path}, row={row_index}"
        )

    for edge_index, edge in enumerate(selected_path):
        if not isinstance(edge, dict):
            raise ClusterTypingAnnotationError(
                f"selected_path[{edge_index}] must be an object. "
                f"JSONL: {jsonl_path}, row={row_index}"
            )

        try:
            parent_class_iri = str(edge["parent_class_iri"])
            edge_kind = str(edge["edge_kind"])
            child_raw = edge.get("child_class_iri")
            child_class_iri = None if child_raw is None else str(child_raw)
            edge_weight = float(edge["edge_weight"])
        except KeyError as exc:
            raise ClusterTypingAnnotationError(
                f"selected_path[{edge_index}] missing required field {exc.args[0]!r}. "
                f"JSONL: {jsonl_path}, row={row_index}"
            ) from exc

        if not 0.0 <= edge_weight <= 1.0:
            raise ClusterTypingAnnotationError(
                f"selected_path[{edge_index}].edge_weight must be in [0, 1], "
                f"got {edge_weight}. JSONL: {jsonl_path}, row={row_index}"
            )

        try:
            validate_selected_path_edge(
                class_graph.graph,
                parent_class_iri=parent_class_iri,
                child_class_iri=child_class_iri,
                edge_kind=edge_kind,
            )
        except Exception as exc:
            raise ClusterTypingAnnotationError(
                f"Invalid selected_path edge at index {edge_index}. "
                f"JSONL: {jsonl_path}, row={row_index}: {exc}"
            ) from exc

    return selected_path


def _mention_weight_from_record(
    record: dict[str, Any],
    *,
    config: ClusterTypingAnnotationConfig,
    jsonl_path: Path,
    row_index: int,
) -> float:
    if not config.use_mention_weight:
        return 1.0

    raw = record.get("mention_weight_raw")
    if not isinstance(raw, dict):
        raise ClusterTypingAnnotationError(
            f"JSONL record must contain mention_weight_raw object when "
            f"use_mention_weight=True. JSONL: {jsonl_path}, row={row_index}"
        )

    try:
        return collapse_mention_weight(
            high=float(raw["high"]),
            medium=float(raw["medium"]),
            low=float(raw["low"]),
            rounding_digits=config.rounding_digits,
        )
    except KeyError as exc:
        raise ClusterTypingAnnotationError(
            f"mention_weight_raw missing field {exc.args[0]!r}. "
            f"JSONL: {jsonl_path}, row={row_index}"
        ) from exc


def _aggregate_cluster_class_iri(
    *,
    class_graph: OntologyClassGraph,
    records: list[dict[str, Any]],
    config: ClusterTypingAnnotationConfig,
    jsonl_path: Path,
) -> str:
    if not records:
        raise ClusterTypingAnnotationError(f"Cannot aggregate empty JSONL: {jsonl_path}")

    edge_scores: dict[tuple[str, str], float] = defaultdict(float)

    for row_index, record in enumerate(records, start=0):
        selected_path = _validate_selected_path(
            class_graph=class_graph,
            record=record,
            jsonl_path=jsonl_path,
            row_index=row_index,
        )
        mention_weight = _mention_weight_from_record(
            record,
            config=config,
            jsonl_path=jsonl_path,
            row_index=row_index,
        )

        for edge in selected_path:
            parent_class_iri = str(edge["parent_class_iri"])
            edge_kind = str(edge["edge_kind"])
            edge_weight = float(edge["edge_weight"])
            contribution = mention_weight * edge_weight

            if edge_kind == "stay":
                edge_scores[(parent_class_iri, STAY)] += contribution
            else:
                child_class_iri = str(edge["child_class_iri"])
                edge_scores[(parent_class_iri, child_class_iri)] += contribution

    roots = list(class_graph.roots())
    if not roots:
        raise ClusterTypingAnnotationError("Ontology graph has no roots.")

    if len(roots) > 1:
        current_class_iri = VIRTUAL_ROOT
    else:
        current_class_iri = roots[0]

    if current_class_iri == VIRTUAL_ROOT:
        root_scores = {
            root_iri: edge_scores.get((VIRTUAL_ROOT, root_iri), 0.0)
            for root_iri in roots
        }
        best_root, best_score = max(root_scores.items(), key=lambda item: item[1])
        if best_score <= 0.0:
            raise ClusterTypingAnnotationError(
                "Could not choose a root class from selected path evidence."
            )
        current_class_iri = best_root

    while True:
        children = list(class_graph.children(current_class_iri))

        if not children:
            return current_class_iri

        child_scores = {
            child_iri: edge_scores.get((current_class_iri, child_iri), 0.0)
            for child_iri in children
        }
        best_child, best_child_score = max(child_scores.items(), key=lambda item: item[1])
        stay_score = edge_scores.get((current_class_iri, STAY), 0.0)

        if stay_score >= best_child_score and stay_score > 0.0:
            return current_class_iri

        if best_child_score <= 0.0:
            return current_class_iri

        current_class_iri = best_child


def _cluster_annotation_from_records(
    *,
    class_graph: OntologyClassGraph,
    cluster_id: int,
    records: list[dict[str, Any]],
    config: ClusterTypingAnnotationConfig,
    jsonl_path: Path,
) -> ClusterTypingAnnotation:
    final_class_iri = _aggregate_cluster_class_iri(
        class_graph=class_graph,
        records=records,
        config=config,
        jsonl_path=jsonl_path,
    )

    return ClusterTypingAnnotation(
        cluster_id=int(cluster_id),
        class_iri=final_class_iri,
    )


def annotate_doc_with_cluster_typing_folder(
    doc: Any,
    class_graph: OntologyClassGraph,
    folder_path: str | Path,
    *,
    config: ClusterTypingAnnotationConfig | None = None,
    pattern: str = "cluster_typing_evidence_cluster_*.jsonl",
) -> Any:
    """Annotate ``doc`` with a fresh ``doc._.cluster_typing_layer`` from JSONL files.

    The annotator processes every matching schema-v2 JSONL in the folder. It
    does not decide whether those JSONLs should exist and does not inspect
    semantic types beyond validating class IRIs against ``class_graph``.
    """

    config = config or ClusterTypingAnnotationConfig()
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ClusterTypingAnnotationError(
            f"Cluster-typing evidence folder does not exist or is not a directory: {folder}"
        )

    jsonl_paths = sorted(folder.glob(pattern))
    if not jsonl_paths:
        raise ClusterTypingAnnotationError(
            f"No cluster-typing JSONL files found in {folder} with pattern {pattern!r}"
        )

    coref_layer = require_coref_layer(doc)
    register_spacy_cluster_typing_extension()

    cluster_typing_layer = ClusterTypingLayer(class_graph=class_graph, source_folder=str(folder))
    seen_cluster_ids: set[int] = set()
    seen_mention_ids: set[int] = set()

    for jsonl_path in jsonl_paths:
        records = list(read_jsonl(jsonl_path))
        if not records:
            raise ClusterTypingAnnotationError(f"Cluster-typing JSONL is empty: {jsonl_path}")

        cluster_id = _require_single_cluster_id(records, jsonl_path=jsonl_path)
        if cluster_id in seen_cluster_ids:
            raise ClusterTypingAnnotationError(
                f"Duplicate cluster-typing JSONLs for cluster_id={cluster_id}. "
                f"Ambiguous cluster annotation in folder {folder}"
            )
        seen_cluster_ids.add(cluster_id)

        for row_index, record in enumerate(records, start=0):
            mention_id, _row_cluster_id = _validate_record_alignment(
                record=record,
                coref_layer=coref_layer,
                jsonl_path=jsonl_path,
                row_index=row_index,
            )

            if mention_id in seen_mention_ids:
                raise ClusterTypingAnnotationError(
                    f"Duplicate mention_id={mention_id} across cluster-typing JSONLs in {folder}."
                )
            seen_mention_ids.add(mention_id)

        cluster_typing_layer.clusters[cluster_id] = _cluster_annotation_from_records(
            class_graph=class_graph,
            cluster_id=cluster_id,
            records=records,
            config=config,
            jsonl_path=jsonl_path,
        )

    doc._.cluster_typing_layer = cluster_typing_layer
    return doc
