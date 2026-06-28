from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

from annotation_layer.relations import make_relation_assertion_id


@dataclass(frozen=True, slots=True)
class RelationAggregationConfig:
    """Configuration for V1 cluster-level relation aggregation."""

    aggregation_method: str = "sum_softmax_by_cluster_pair"
    min_support_count: int = 1
    min_score: float = 0.0


def _read_assignment_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assignment_probabilities(row: dict[str, Any]) -> dict[str, float]:
    probabilities = json.loads(row["object_property_probabilities_json"])
    return {
        str(object_property_iri): float(probability)
        for object_property_iri, probability in probabilities.items()
    }


def export_cluster_assertions_csv(
    *,
    assignments_path: str | Path,
    output_path: str | Path,
    aggregation_config: RelationAggregationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    config = aggregation_config or RelationAggregationConfig()
    assignments_path = Path(assignments_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True to replace it.")

    rows = _read_assignment_rows(assignments_path)

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["source_cluster_id"]), int(row["target_cluster_id"]))
        grouped.setdefault(key, []).append(row)

    fieldnames = [
        "cluster_assertion_id",
        "source_cluster_id",
        "object_property_iri",
        "target_cluster_id",
        "support_relation_ids_json",
        "aggregation_method",
        "aggregated_score",
    ]

    n_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (source_cluster_id, target_cluster_id), support_rows in sorted(grouped.items()):
            if len(support_rows) < config.min_support_count:
                continue

            property_scores: dict[str, float] = {}
            support_relation_ids: list[str] = []

            for row in support_rows:
                relation_id = str(row["relation_id"])
                support_relation_ids.append(relation_id)

                for object_property_iri, probability in _assignment_probabilities(row).items():
                    property_scores[object_property_iri] = (
                        property_scores.get(object_property_iri, 0.0) + float(probability)
                    )

            if not property_scores:
                continue

            best_property_iri = max(property_scores, key=property_scores.get)
            best_score = float(property_scores[best_property_iri])

            if best_score < config.min_score:
                continue

            assertion_id = make_relation_assertion_id(
                source_cluster_id=source_cluster_id,
                object_property_iri=best_property_iri,
                target_cluster_id=target_cluster_id,
            )

            writer.writerow(
                {
                    "cluster_assertion_id": assertion_id,
                    "source_cluster_id": source_cluster_id,
                    "object_property_iri": best_property_iri,
                    "target_cluster_id": target_cluster_id,
                    "support_relation_ids_json": json.dumps(tuple(support_relation_ids), ensure_ascii=False),
                    "aggregation_method": config.aggregation_method,
                    "aggregated_score": best_score,
                }
            )
            n_written += 1

    print(f"[relation aggregation] Wrote {n_written} cluster assertions to {output_path}")
    return output_path


# Short alias used by pipeline notebooks.
export_cluster_assertions = export_cluster_assertions_csv
