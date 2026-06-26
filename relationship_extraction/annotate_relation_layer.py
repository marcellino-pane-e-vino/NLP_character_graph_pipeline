from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "annotate_relation_layer.py requires spaCy. Install it with: pip install spacy"
    ) from exc

from coreference.coref_schema import require_coref_layer
from relationship_extraction.relation_schema import (
    ClusterAssertion,
    RelationAssignment,
    RelationLayer,
    RelationMention,
    register_spacy_relation_extension,
)


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assignment_from_record(
    *,
    record: dict[str, Any],
    coref: Any,
) -> tuple[str, RelationAssignment]:
    relation_mention_id = str(record["relation_mention_id"])
    source_mention_id = int(record["source_mention_id"])
    target_mention_id = int(record["target_mention_id"])

    if source_mention_id not in coref.mentions:
        raise KeyError(f"source_mention_id={source_mention_id} not found in doc._.coref_layer")

    if target_mention_id not in coref.mentions:
        raise KeyError(f"target_mention_id={target_mention_id} not found in doc._.coref_layer")

    relation_mention = RelationMention(
        relation_mention_id=relation_mention_id,
        source_mention=coref.mentions[source_mention_id],
        predicate_token_i=int(record["predicate_token_i"]),
        predicate_start=int(record["predicate_start"]),
        predicate_end=int(record["predicate_end"]),
        target_mention=coref.mentions[target_mention_id],
    )

    logits = json.loads(record["object_property_logits_json"])
    assignment = RelationAssignment(
        relation_mention=relation_mention,
        object_property_logits={str(k): float(v) for k, v in logits.items()},
        selection_method=str(record["selection_method"]),
    )

    return relation_mention_id, assignment


def _cluster_assertion_from_record(record: dict[str, Any]) -> tuple[str, ClusterAssertion]:
    assertion_id = str(record["cluster_assertion_id"])
    support_assignment_ids = tuple(json.loads(record["support_assignment_ids_json"]))

    assertion = ClusterAssertion(
        cluster_assertion_id=assertion_id,
        source_cluster_id=int(record["source_cluster_id"]),
        object_property_iri=str(record["object_property_iri"]),
        target_cluster_id=int(record["target_cluster_id"]),
        support_assignment_ids=support_assignment_ids,
        aggregation_method=str(record["aggregation_method"]),
    )

    return assertion_id, assertion


def build_relation_layer_from_files(
    *,
    doc: Doc,
    assignments_path: str | Path,
    cluster_assertions_path: str | Path,
) -> RelationLayer:
    """Build a RelationLayer from staging CSVs and doc._.coref_layer."""

    coref = require_coref_layer(doc)

    assignments = dict(
        _assignment_from_record(record=record, coref=coref)
        for record in _read_csv(assignments_path)
    )

    cluster_assertions = dict(
        _cluster_assertion_from_record(record)
        for record in _read_csv(cluster_assertions_path)
    )

    missing_support_ids = sorted(
        {
            assignment_id
            for assertion in cluster_assertions.values()
            for assignment_id in assertion.support_assignment_ids
            if assignment_id not in assignments
        }
    )
    if missing_support_ids:
        preview = ", ".join(missing_support_ids[:20])
        raise ValueError(
            "cluster_assertions reference assignment ids that are absent from assignments CSV: "
            f"{preview}"
        )

    return RelationLayer.from_data(
        assignments=assignments,
        cluster_assertions=cluster_assertions,
    )


def annotate_relation_layer_from_files(
    *,
    doc: Doc,
    assignments_path: str | Path,
    cluster_assertions_path: str | Path,
    force: bool = False,
) -> RelationLayer:
    """Create doc._.relation_layer from assignment/assertion staging files."""

    register_spacy_relation_extension(force=force)

    if doc._.relation_layer is not None and not force:
        raise ValueError(
            "doc._.relation_layer already exists. Pass force=True to replace it."
        )

    relation_layer = build_relation_layer_from_files(
        doc=doc,
        assignments_path=assignments_path,
        cluster_assertions_path=cluster_assertions_path,
    )
    doc._.relation_layer = relation_layer
    return relation_layer
