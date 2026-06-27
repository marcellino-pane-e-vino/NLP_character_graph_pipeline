from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from neural_runtime.nli import (
    DEFAULT_NLI_MODEL_NAME,
    DirectNLIConfig as RelationDirectNLIConfig,
    direct_entailment_logits_for_pairs,
    softmax_values,
)


__all__ = [
    "RelationDirectNLIConfig",
    "RelationScoringConfig",
    "RelationAssignmentExportConfig",
    "relation_hypothesis_text",
    "export_relation_assignments_csv",
    "export_relation_assignments",
]


@dataclass(frozen=True, slots=True)
class RelationScoringConfig:
    """Domain-level configuration for relation-property hypothesis scoring."""

    model_name: str = DEFAULT_NLI_MODEL_NAME
    hypothesis_template: str = (
        'The source entity and target entity are connected by the relation '
        '"{label}", meaning: {description}'
    )


@dataclass(frozen=True, slots=True)
class RelationAssignmentExportConfig:
    """Configuration for relation assignment CSV export."""

    scoring_config: RelationScoringConfig = field(default_factory=RelationScoringConfig)
    nli_config: RelationDirectNLIConfig = field(default_factory=RelationDirectNLIConfig)

    overwrite_csv: bool = False
    resume_from_csv: bool = True
    print_progress: bool = True


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def _candidate_label(candidate_property: dict[str, Any]) -> str:
    return str(
        candidate_property.get("human_readable_label")
        or candidate_property.get("label")
        or candidate_property.get("local_name")
        or candidate_property.get("iri")
    )


def relation_hypothesis_text(
    candidate_property: dict[str, Any],
    *,
    scoring_config: RelationScoringConfig,
) -> str:
    label = _candidate_label(candidate_property)
    description = str(candidate_property.get("description") or "").strip()
    return scoring_config.hypothesis_template.format(label=label, description=description)


def _score_candidate_property_probabilities(
    *,
    premise_text: str,
    candidate_properties: list[dict[str, Any]],
    scoring_config: RelationScoringConfig,
    nli_config: RelationDirectNLIConfig,
) -> dict[str, float]:
    property_iris = [str(candidate["iri"]) for candidate in candidate_properties]

    if len(property_iris) == 1:
        return {property_iris[0]: 1.0}

    hypotheses = [
        relation_hypothesis_text(
            candidate_property,
            scoring_config=scoring_config,
        )
        for candidate_property in candidate_properties
    ]

    logits = direct_entailment_logits_for_pairs(
        [(premise_text, hypothesis) for hypothesis in hypotheses],
        model_name=scoring_config.model_name,
        nli_config=nli_config,
    )
    if len(logits) != len(property_iris):
        raise ValueError(
            "NLI scorer returned a different number of logits than input pairs: "
            f"{len(logits)} != {len(property_iris)}"
        )

    probabilities = softmax_values([float(value) for value in logits])
    return {
        property_iri: float(probability)
        for property_iri, probability in zip(property_iris, probabilities)
    }


def _existing_assignment_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())

        if "object_property_logits_json" in fieldnames:
            raise ValueError(
                f"{path} uses the old relation-assignment schema with "
                "object_property_logits_json. Delete it or rerun with "
                "overwrite_csv=True to regenerate relation assignments."
            )

        if "object_property_probabilities_json" not in fieldnames:
            raise ValueError(
                f"{path} is missing object_property_probabilities_json. "
                "Delete it or rerun with overwrite_csv=True."
            )

        return {
            str(row["relation_id"])
            for row in reader
            if row.get("relation_id")
        }


def export_relation_assignments_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    config: RelationAssignmentExportConfig | None = None,
) -> Path:
    """Score routed candidates and export flat assignment rows.

    Output rows contain primary data needed to reconstruct RelationAssignment
    and RelationInstance inside annotate_relation_layer.py.

    ``object_property_probabilities_json`` stores the per-relation grouped-softmax
    probability distribution over the candidate object properties routed for the
    source/target type pair.
    """

    config = config or RelationAssignmentExportConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and config.overwrite_csv:
        output_path.unlink()

    if output_path.exists() and not config.resume_from_csv:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Pass overwrite_csv=True or resume_from_csv=True."
        )

    existing_ids = _existing_assignment_ids(output_path) if config.resume_from_csv else set()
    write_header = not output_path.exists()

    fieldnames = [
        "relation_id",
        "source_mention_id",
        "predicate_token_i",
        "predicate_start",
        "predicate_end",
        "target_mention_id",
        "source_cluster_id",
        "target_cluster_id",
        "object_property_probabilities_json",
        "selection_method",
    ]

    n_written = 0
    n_skipped = 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in _read_jsonl(input_path):
            relation_id = str(row["relation_id"])
            if relation_id in existing_ids:
                n_skipped += 1
                continue

            candidate_properties = list(row.get("candidate_properties") or [])
            if not candidate_properties:
                raise ValueError(
                    f"Routed candidate {relation_id!r} has no candidate_properties."
                )

            probabilities = _score_candidate_property_probabilities(
                premise_text=str(row["premise_text"]),
                candidate_properties=candidate_properties,
                scoring_config=config.scoring_config,
                nli_config=config.nli_config,
            )

            selection_method = (
                "single_candidate_router"
                if len(candidate_properties) == 1
                else "nli_grouped_softmax"
            )

            writer.writerow(
                {
                    "relation_id": relation_id,
                    "source_mention_id": int(row["source_mention_id"]),
                    "predicate_token_i": int(row["predicate_token_i"]),
                    "predicate_start": int(row["predicate_start"]),
                    "predicate_end": int(row["predicate_end"]),
                    "target_mention_id": int(row["target_mention_id"]),
                    "source_cluster_id": int(row["source_cluster_id"]),
                    "target_cluster_id": int(row["target_cluster_id"]),
                    "object_property_probabilities_json": json.dumps(
                        probabilities,
                        ensure_ascii=False,
                    ),
                    "selection_method": selection_method,
                }
            )
            n_written += 1

            if config.print_progress and n_written % 100 == 0:
                print(f"[relation alignment] wrote {n_written} assignments...")

    if config.print_progress:
        print(
            f"[relation alignment] Wrote {n_written} assignments to {output_path} "
            f"({n_skipped} skipped by resume)."
        )

    return output_path


# Short alias used by pipeline notebooks.
export_relation_assignments = export_relation_assignments_csv
