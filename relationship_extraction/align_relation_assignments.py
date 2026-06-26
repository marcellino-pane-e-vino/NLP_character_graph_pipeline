from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True, slots=True)
class RelationNLIConfig:
    """Runtime configuration for neural relation-property scoring."""

    model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    hypothesis_template: str = (
        'The source entity and target entity are connected by the relation '
        '"{label}", meaning: {description}'
    )
    pair_batch_size: int = 16
    truncation: bool = True
    max_length: int | None = None
    device: str | None = None


class RelationPairScorer(Protocol):
    """Minimal protocol consumed by export_relation_assignments_csv."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        ...


class TransformersRelationNLISelector:
    """Small direct-NLI scorer for premise/hypothesis pairs.

    It returns the entailment logit for each pair. The heavy imports are lazy so
    the rest of the module remains usable without transformers/torch installed.
    """

    def __init__(self, config: RelationNLIConfig | None = None) -> None:
        self.config = config or RelationNLIConfig()

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TransformersRelationNLISelector requires torch and transformers. "
                "Install them or pass a custom selector implementing score_pairs()."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)

        if self.config.device is not None:
            self.device = self.config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()
        self.entailment_label_id = self._infer_entailment_label_id()

    def _infer_entailment_label_id(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        # Common fallback for many NLI heads. If wrong for a model, pass a custom selector.
        return max(int(i) for i in id2label) if id2label else 2

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        scores: list[float] = []
        batch_size = int(self.config.pair_batch_size)

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            premises = [premise for premise, _ in batch]
            hypotheses = [hypothesis for _, hypothesis in batch]

            encoded = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with self._torch.no_grad():
                logits = self.model(**encoded).logits

            entailment_logits = logits[:, self.entailment_label_id].detach().cpu().tolist()
            scores.extend(float(value) for value in entailment_logits)

        return scores


def load_relation_nli_selector(config: RelationNLIConfig | None = None) -> TransformersRelationNLISelector:
    return TransformersRelationNLISelector(config=config)


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
    config: RelationNLIConfig,
) -> str:
    label = _candidate_label(candidate_property)
    description = str(candidate_property.get("description") or "").strip()
    return config.hypothesis_template.format(label=label, description=description)


def _score_candidate_properties(
    *,
    premise_text: str,
    candidate_properties: list[dict[str, Any]],
    selector: RelationPairScorer,
    config: RelationNLIConfig,
) -> dict[str, float]:
    property_iris = [str(candidate["iri"]) for candidate in candidate_properties]

    if len(property_iris) == 1:
        return {property_iris[0]: 0.0}

    hypotheses = [
        relation_hypothesis_text(candidate, config=config)
        for candidate in candidate_properties
    ]
    pairs = [(premise_text, hypothesis) for hypothesis in hypotheses]

    logits = selector.score_pairs(pairs)
    if len(logits) != len(property_iris):
        raise ValueError(
            "selector.score_pairs returned a different number of logits than input pairs: "
            f"{len(logits)} != {len(property_iris)}"
        )

    return {
        property_iri: float(logit)
        for property_iri, logit in zip(property_iris, logits)
    }


def _existing_assignment_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {
            str(row["relation_mention_id"])
            for row in reader
            if row.get("relation_mention_id")
        }


def export_relation_assignments_csv(
    *,
    input_path: str | Path,
    output_path: str | Path,
    selector: RelationPairScorer,
    config: RelationNLIConfig | None = None,
    overwrite: bool = False,
    resume: bool = True,
    print_progress: bool = True,
) -> Path:
    """Score routed candidates and export flat assignment rows.

    Output rows contain only primary data needed to reconstruct RelationAssignment
    and RelationMention inside annotate_relation_layer.py.
    """

    config = config or RelationNLIConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and overwrite:
        output_path.unlink()

    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass overwrite=True or resume=True."
        )

    existing_ids = _existing_assignment_ids(output_path) if resume else set()
    write_header = not output_path.exists()

    fieldnames = [
        "relation_mention_id",
        "source_mention_id",
        "predicate_token_i",
        "predicate_start",
        "predicate_end",
        "target_mention_id",
        "source_cluster_id",
        "target_cluster_id",
        "object_property_logits_json",
        "selection_method",
    ]

    n_written = 0
    n_skipped = 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in _read_jsonl(input_path):
            relation_mention_id = str(row["relation_mention_id"])
            if relation_mention_id in existing_ids:
                n_skipped += 1
                continue

            candidate_properties = list(row.get("candidate_properties") or [])
            if not candidate_properties:
                raise ValueError(
                    f"Routed candidate {relation_mention_id!r} has no candidate_properties."
                )

            logits = _score_candidate_properties(
                premise_text=str(row["premise_text"]),
                candidate_properties=candidate_properties,
                selector=selector,
                config=config,
            )

            selection_method = (
                "single_candidate_router"
                if len(candidate_properties) == 1
                else "nli_grouped_softmax"
            )

            writer.writerow(
                {
                    "relation_mention_id": relation_mention_id,
                    "source_mention_id": int(row["source_mention_id"]),
                    "predicate_token_i": int(row["predicate_token_i"]),
                    "predicate_start": int(row["predicate_start"]),
                    "predicate_end": int(row["predicate_end"]),
                    "target_mention_id": int(row["target_mention_id"]),
                    "source_cluster_id": int(row["source_cluster_id"]),
                    "target_cluster_id": int(row["target_cluster_id"]),
                    "object_property_logits_json": json.dumps(logits, ensure_ascii=False),
                    "selection_method": selection_method,
                }
            )
            n_written += 1

            if print_progress and n_written % 100 == 0:
                print(f"[relation alignment] wrote {n_written} assignments...")

    if print_progress:
        print(
            f"[relation alignment] Wrote {n_written} assignments to {output_path} "
            f"({n_skipped} skipped by resume)."
        )

    return output_path


# Short alias used by pipeline notebooks.
export_relation_assignments = export_relation_assignments_csv
