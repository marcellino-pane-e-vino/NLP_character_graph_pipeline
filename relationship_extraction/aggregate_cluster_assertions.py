from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

from annotation_layer.relations import make_relation_assertion_id


MULTI_LABEL_NOISY_OR_METHOD = "multi_label_noisy_or_by_cluster_pair"
LEGACY_SUM_SOFTMAX_METHOD = "sum_softmax_by_cluster_pair"
DEFAULT_NOISY_OR_MIN_SCORE = 0.60


@dataclass(frozen=True, slots=True)
class RelationAggregationConfig:
    """Configuration for cluster-level relation aggregation.

    The public contract is intentionally kept compatible with the previous
    aggregator: callers still pass assignments_path, output_path, this config,
    and overwrite to export_cluster_assertions_csv(). The output CSV schema is
    also unchanged.

    Policy:
        - group relation assignments by directed cluster pair;
        - compute one Noisy-OR bag score for each object property;
        - emit every property whose bag score passes min_score and whose local
          winning support count passes min_support_count.

    Notes:
        - min_score is the minimum Noisy-OR bag-level confidence.
        - min_support_count counts relation assignments where the property is
          the local winner for that assignment.
        - the legacy aggregation_method value is accepted as an alias so older
          notebooks do not need to change. When that legacy value is used with
          min_score <= 0.0, DEFAULT_NOISY_OR_MIN_SCORE is used to avoid emitting
          every positive softmax tail as a cluster assertion.
    """

    aggregation_method: str = MULTI_LABEL_NOISY_OR_METHOD
    min_support_count: int = 1
    min_score: float = DEFAULT_NOISY_OR_MIN_SCORE


def _read_assignment_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json_object(row: dict[str, Any], fieldname: str) -> dict[str, float]:
    raw_value = row.get(fieldname)
    if raw_value in (None, ""):
        return {}

    values = json.loads(raw_value)
    if not isinstance(values, dict):
        raise ValueError(f"{fieldname} must contain a JSON object.")

    return {
        str(object_property_iri): float(value)
        for object_property_iri, value in values.items()
    }


def _stable_softmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}

    max_value = max(values.values())
    exp_values = {
        key: math.exp(float(value) - max_value)
        for key, value in values.items()
    }
    denominator = sum(exp_values.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("Cannot softmax relation-assignment logits with invalid denominator.")

    return {
        key: float(exp_value / denominator)
        for key, exp_value in exp_values.items()
    }


def _validate_probability(value: float, *, fieldname: str) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{fieldname} contains a non-finite probability: {probability!r}")

    tolerance = 1e-12
    if probability < -tolerance or probability > 1.0 + tolerance:
        raise ValueError(
            f"{fieldname} contains a value outside [0, 1]: {probability!r}"
        )

    return min(1.0, max(0.0, probability))


def _assignment_probabilities(row: dict[str, Any]) -> dict[str, float]:
    """Return object-property probabilities for one relation assignment.

    Current assignment CSVs store object_property_probabilities_json. Older
    staging CSVs may store object_property_logits_json; those are converted with
    a stable softmax here so the aggregation contract stays localized to this
    module.
    """

    probabilities = _read_json_object(row, "object_property_probabilities_json")
    if not probabilities:
        logits = _read_json_object(row, "object_property_logits_json")
        probabilities = _stable_softmax(logits)

    if not probabilities:
        relation_id = row.get("relation_id", "<unknown>")
        raise ValueError(
            f"Relation assignment {relation_id!r} has neither probabilities nor logits."
        )

    return {
        object_property_iri: _validate_probability(
            probability,
            fieldname="object_property_probabilities_json",
        )
        for object_property_iri, probability in probabilities.items()
    }


def _winning_property_iri(probabilities: dict[str, float]) -> str:
    if not probabilities:
        raise ValueError("Cannot select a winning object property from an empty assignment.")
    return max(probabilities, key=probabilities.get)


def _noisy_or(probabilities: list[float]) -> float:
    """Compute 1 - product(1 - p_i) with stable log-space arithmetic."""

    if not probabilities:
        return 0.0

    log_no_evidence = 0.0
    for probability in probabilities:
        p = _validate_probability(probability, fieldname="Noisy-OR input")
        if p >= 1.0:
            return 1.0
        log_no_evidence += math.log1p(-p)

    return float(-math.expm1(log_no_evidence))


def _resolved_aggregation_method(config: RelationAggregationConfig) -> str:
    method = str(config.aggregation_method)
    if method in {MULTI_LABEL_NOISY_OR_METHOD, LEGACY_SUM_SOFTMAX_METHOD}:
        return MULTI_LABEL_NOISY_OR_METHOD

    raise ValueError(
        "Unsupported relation aggregation method: "
        f"{config.aggregation_method!r}. Supported values are "
        f"{MULTI_LABEL_NOISY_OR_METHOD!r} and legacy alias "
        f"{LEGACY_SUM_SOFTMAX_METHOD!r}."
    )


def _effective_min_score(config: RelationAggregationConfig) -> float:
    """Return the Noisy-OR threshold to apply.

    Old notebooks may still instantiate RelationAggregationConfig with the old
    method name and min_score=0.0. Under multi-label Noisy-OR, a zero threshold
    would emit every candidate property because softmax probabilities are
    positive. Treat that exact legacy configuration as a request for the safe
    default threshold.
    """

    if float(config.min_score) > 0.0:
        return float(config.min_score)

    if str(config.aggregation_method) == LEGACY_SUM_SOFTMAX_METHOD:
        return DEFAULT_NOISY_OR_MIN_SCORE

    return float(config.min_score)


def export_cluster_assertions_csv(
    *,
    assignments_path: str | Path,
    output_path: str | Path,
    aggregation_config: RelationAggregationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Aggregate relation assignments into cluster-level relation assertions.

    This function keeps the previous I/O contract but changes the aggregation
    semantics from one best assertion per cluster pair to multi-label Noisy-OR
    aggregation. A directed cluster pair may therefore produce multiple
    assertions, one for each sufficiently supported object property.
    """

    config = aggregation_config or RelationAggregationConfig()
    aggregation_method = _resolved_aggregation_method(config)
    min_score = _effective_min_score(config)

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
            property_probabilities: dict[str, list[float]] = {}
            winning_support_relation_ids: dict[str, list[str]] = {}

            for row in support_rows:
                relation_id = str(row["relation_id"])
                probabilities = _assignment_probabilities(row)
                winning_property_iri = _winning_property_iri(probabilities)
                winning_support_relation_ids.setdefault(winning_property_iri, []).append(relation_id)

                for object_property_iri, probability in probabilities.items():
                    property_probabilities.setdefault(object_property_iri, []).append(probability)

            scored_properties = [
                (object_property_iri, _noisy_or(probabilities))
                for object_property_iri, probabilities in property_probabilities.items()
            ]

            for object_property_iri, bag_score in sorted(
                scored_properties,
                key=lambda item: (-item[1], item[0]),
            ):
                support_relation_ids = winning_support_relation_ids.get(object_property_iri, [])
                if len(support_relation_ids) < config.min_support_count:
                    continue

                if bag_score < min_score:
                    continue

                assertion_id = make_relation_assertion_id(
                    source_cluster_id=source_cluster_id,
                    object_property_iri=object_property_iri,
                    target_cluster_id=target_cluster_id,
                )

                writer.writerow(
                    {
                        "cluster_assertion_id": assertion_id,
                        "source_cluster_id": source_cluster_id,
                        "object_property_iri": object_property_iri,
                        "target_cluster_id": target_cluster_id,
                        "support_relation_ids_json": json.dumps(
                            tuple(support_relation_ids),
                            ensure_ascii=False,
                        ),
                        "aggregation_method": aggregation_method,
                        "aggregated_score": bag_score,
                    }
                )
                n_written += 1

    print(f"[relation aggregation] Wrote {n_written} cluster assertions to {output_path}")
    return output_path


# Short alias used by pipeline notebooks.
export_cluster_assertions = export_cluster_assertions_csv
