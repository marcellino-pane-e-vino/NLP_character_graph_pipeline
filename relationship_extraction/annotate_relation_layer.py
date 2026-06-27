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

from annotation_layer.relations import (
    RelationAssertionRecord,
    RelationAssignmentRecord,
    RelationInstanceRecord,
    RelationSubLayer,
)
from annotation_layer.spacy_extension import require_annotation_layer, require_entities


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assignment_and_instance_from_record(
    *,
    record: dict[str, Any],
    entities: Any,
) -> tuple[str, RelationInstanceRecord, RelationAssignmentRecord]:
    relation_id = str(record["relation_id"])
    source_mention_id = int(record["source_mention_id"])
    target_mention_id = int(record["target_mention_id"])

    if source_mention_id not in entities.mentions:
        raise KeyError(
            f"source_mention_id={source_mention_id} not found in doc._.annotation_layer.entities"
        )

    if target_mention_id not in entities.mentions:
        raise KeyError(
            f"target_mention_id={target_mention_id} not found in doc._.annotation_layer.entities"
        )

    instance = RelationInstanceRecord(
        relation_id=relation_id,
        source_mention_id=source_mention_id,
        predicate_token_i=int(record["predicate_token_i"]),
        predicate_start=int(record["predicate_start"]),
        predicate_end=int(record["predicate_end"]),
        target_mention_id=target_mention_id,
    )

    logits = json.loads(record["object_property_logits_json"])
    assignment = RelationAssignmentRecord(
        relation_id=relation_id,
        object_property_logits={str(k): float(v) for k, v in logits.items()},
        selection_method=str(record["selection_method"]),
    )

    return relation_id, instance, assignment


def _cluster_assertion_from_record(record: dict[str, Any]) -> tuple[str, RelationAssertionRecord]:
    assertion_id = str(record["cluster_assertion_id"])
    support_relation_ids = tuple(str(value) for value in json.loads(record["support_relation_ids_json"]))
    confidence_raw = record.get("aggregated_score") or record.get("confidence")
    confidence = None if confidence_raw in (None, "") else float(confidence_raw)

    assertion = RelationAssertionRecord(
        assertion_id=assertion_id,
        source_cluster_id=int(record["source_cluster_id"]),
        object_property_iri=str(record["object_property_iri"]),
        target_cluster_id=int(record["target_cluster_id"]),
        support_relation_ids=support_relation_ids,
        aggregation_method=str(record["aggregation_method"]),
        confidence=confidence,
    )

    return assertion_id, assertion


def build_relation_sublayer_from_files(
    *,
    doc: Doc,
    assignments_path: str | Path,
    cluster_assertions_path: str | Path,
) -> RelationSubLayer:
    """Build a RelationSubLayer from staging CSVs and entity annotations."""

    entities = require_entities(doc)

    instances: dict[str, RelationInstanceRecord] = {}
    assignments: dict[str, RelationAssignmentRecord] = {}

    for record in _read_csv(assignments_path):
        relation_id, instance, assignment = _assignment_and_instance_from_record(
            record=record,
            entities=entities,
        )
        instances[relation_id] = instance
        assignments[relation_id] = assignment

    assertions = dict(
        _cluster_assertion_from_record(record)
        for record in _read_csv(cluster_assertions_path)
    )

    missing_support_ids = sorted(
        {
            relation_id
            for assertion in assertions.values()
            for relation_id in assertion.support_relation_ids
            if relation_id not in assignments
        }
    )
    if missing_support_ids:
        preview = ", ".join(missing_support_ids[:20])
        raise ValueError(
            "cluster_assertions reference relation ids that are absent from assignments CSV: "
            f"{preview}"
        )

    return RelationSubLayer.from_data(
        instances=instances,
        assignments=assignments,
        assertions=assertions,
    )


def attach_relations_from_files(
    *,
    doc: Doc,
    assignments_path: str | Path,
    cluster_assertions_path: str | Path,
    overwrite: bool = False,
) -> RelationSubLayer:
    """Create doc._.annotation_layer.relations from assignment/assertion staging files."""

    ann = require_annotation_layer(doc)
    relation_layer = build_relation_sublayer_from_files(
        doc=doc,
        assignments_path=assignments_path,
        cluster_assertions_path=cluster_assertions_path,
    )
    ann.attach_relations(relation_layer, overwrite=overwrite)
    return relation_layer
