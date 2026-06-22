"""JSONL-first ontology cluster typing evidence export.

This module is the model-inference/export stage of the ontology typing pipeline.
It consumes ``doc._.coref_layer`` and an external ``networkx.DiGraph`` ontology
DAG. Multi-cluster export intentionally labels every cluster present in
``doc._.coref_layer``.

It does not build the ontology graph, does not filter or semantically select
clusters, does not annotate the Doc, and never mutates ``doc._.coref_layer``.

Output contract:
    ``./outputs/ontology_typing/{n_mentions}/ontology_evidence_cluster_*.jsonl``

Each JSONL record stores only:
    - stable mention identity fields
    - the selected ontology path as weighted edges
    - raw high/medium/low mention-weight probabilities

No candidate distributions are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import gc
import time

import networkx as nx

from coreference.coref_schema import require_coref_layer

from ocean.ocean_probability_scoring import (
    ContextConfig,
    MentionRenderingConfig,
    DirectNLIConfig,
    canonical_name_for_cluster,
    mention_ids_for_cluster,
    mention_records_for_cluster,
    _cluster_random_seed,
    _device_name,
    _direct_entailment_logits_for_pairs,
)

from .graph_contract import (
    VIRTUAL_ROOT,
    class_label,
    class_prompt_label,
    is_ontology_leaf,
    ontology_children,
    ontology_roots,
    validate_ontology_graph,
)
from .ontology_artifacts import (
    append_jsonl,
    completed_mention_ids_from_jsonl,
    default_cluster_jsonl_path,
    ontology_typing_output_dir,
)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "ONTOLOGY_MENTION_WEIGHT_LABELS",
    "OntologyTraversalConfig",
    "OntologyScoringConfig",
    "OntologyMentionWeightConfig",
    "OntologyEvidenceExportConfig",
    "export_ontology_evidence_jsonl_for_cluster",
    "export_ontology_evidence_jsonls",
]


DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
SCHEMA_VERSION = 1


ONTOLOGY_MENTION_WEIGHT_LABELS: dict[str, str] = {
    "high": (
        "strong ontology typing evidence: the entity is named, described, "
        "or used in a way that clearly helps decide what kind of narrative "
        "entity it is"
    ),
    "medium": (
        "weak or ambiguous ontology typing evidence: the entity has some "
        "contextual clues about its entity type, but the evidence is indirect "
        "or incomplete"
    ),
    "low": (
        "no useful ontology typing evidence: the entity is mentioned in a "
        "context that does not help decide its ontology class"
    ),
}


@dataclass(frozen=True)
class OntologyTraversalConfig:
    """Controls greedy mention-level ontology traversal."""

    skip_single_root: bool = True
    include_stay_option: bool = True
    force_leaf: bool = False
    max_depth: int | None = None


@dataclass(frozen=True)
class OntologyScoringConfig:
    """Runtime configuration for direct-NLI ontology edge scoring."""

    model_name: str = DEFAULT_MODEL_NAME

    generic_child_hypothesis_template: str = "The entity is a {label} in this text."
    subject_child_hypothesis_template: str = "{subject} is a {label} in this text."

    generic_stay_hypothesis_template: str = (
        "The entity is best classified as {label}, not as a more specific subclass."
    )
    subject_stay_hypothesis_template: str = (
        "{subject} is best classified as {label}, not as a more specific subclass."
    )

    subject_aware: bool = True


@dataclass(frozen=True)
class OntologyMentionWeightConfig:
    """Configuration for raw ontology mention-weight probability scoring."""

    generic_hypothesis_template: str = "In this text, the entity provides {label}."
    subject_hypothesis_template: str = "In this text, {subject} provides {label}."


@dataclass(frozen=True)
class OntologyEvidenceExportConfig:
    """Configuration for all-cluster ontology evidence export.

    Cluster selection is deliberately not configurable here: multi-cluster
    export always processes every cluster in ``doc._.coref_layer``. Keep
    ``export_ontology_evidence_jsonl_for_cluster`` for focused debugging or
    one-off experiments on a single cluster.
    """

    n_mentions_per_cluster: int | None

    output_root: str | Path = "./outputs"
    random_seed: int | None = None
    sort_sample_by_cluster_order: bool = True

    overwrite_jsonl: bool = False
    resume_from_jsonl: bool = True

    chunk_size: int = 16

    context_config: ContextConfig = field(default_factory=ContextConfig)
    rendering_config: MentionRenderingConfig = field(default_factory=MentionRenderingConfig)
    traversal_config: OntologyTraversalConfig = field(default_factory=OntologyTraversalConfig)
    scoring_config: OntologyScoringConfig = field(default_factory=OntologyScoringConfig)
    mention_weight_config: OntologyMentionWeightConfig = field(default_factory=OntologyMentionWeightConfig)
    nli_config: DirectNLIConfig = field(default_factory=DirectNLIConfig)

    print_progress: bool = True


@dataclass(frozen=True)
class _Candidate:
    class_id: str | None
    edge_kind: str
    hypothesis: str


def _format_template(template: str, *, label: str, subject: str | None) -> str:
    text = template.replace("{label}", label)
    if "{subject}" in text:
        text = text.replace("{subject}", subject or "the entity")
    return text


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [pow(2.718281828459045, value - max_value) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def _score_grouped_probabilities(
    *,
    premise: str,
    hypotheses: list[str],
    model_name: str,
    nli_config: DirectNLIConfig,
) -> list[float]:
    logits = _direct_entailment_logits_for_pairs(
        [(premise, hypothesis) for hypothesis in hypotheses],
        model_name=model_name,
        nli_config=nli_config,
    )
    return _softmax([float(value) for value in logits])


def _candidate_hypothesis(
    *,
    graph: nx.DiGraph,
    candidate_class_id: str,
    edge_kind: str,
    subject: str | None,
    scoring_config: OntologyScoringConfig,
) -> str:
    label = class_prompt_label(graph, candidate_class_id)

    if edge_kind == "stay":
        template = (
            scoring_config.subject_stay_hypothesis_template
            if scoring_config.subject_aware and subject
            else scoring_config.generic_stay_hypothesis_template
        )
    else:
        template = (
            scoring_config.subject_child_hypothesis_template
            if scoring_config.subject_aware and subject
            else scoring_config.generic_child_hypothesis_template
        )

    return _format_template(template, label=label, subject=subject)


def _score_mention_weight_raw(
    *,
    context_text: str,
    subject: str | None,
    scoring_config: OntologyScoringConfig,
    mention_weight_config: OntologyMentionWeightConfig,
    nli_config: DirectNLIConfig,
) -> dict[str, float]:
    template = (
        mention_weight_config.subject_hypothesis_template
        if scoring_config.subject_aware and subject
        else mention_weight_config.generic_hypothesis_template
    )

    ordered_keys = ["high", "medium", "low"]
    hypotheses = [
        _format_template(
            template,
            label=ONTOLOGY_MENTION_WEIGHT_LABELS[key],
            subject=subject,
        )
        for key in ordered_keys
    ]
    probabilities = _score_grouped_probabilities(
        premise=context_text,
        hypotheses=hypotheses,
        model_name=scoring_config.model_name,
        nli_config=nli_config,
    )
    return {key: float(probability) for key, probability in zip(ordered_keys, probabilities)}


def _selected_path_for_mention(
    *,
    graph: nx.DiGraph,
    context_text: str,
    subject: str | None,
    traversal_config: OntologyTraversalConfig,
    scoring_config: OntologyScoringConfig,
    nli_config: DirectNLIConfig,
) -> list[dict[str, Any]]:
    roots = ontology_roots(graph)
    if not roots:
        raise ValueError("Ontology graph has no roots.")

    selected_path: list[dict[str, Any]] = []

    if len(roots) > 1:
        parent_id: str = VIRTUAL_ROOT
        candidates = [
            _Candidate(
                class_id=root,
                edge_kind="root",
                hypothesis=_candidate_hypothesis(
                    graph=graph,
                    candidate_class_id=root,
                    edge_kind="root",
                    subject=subject,
                    scoring_config=scoring_config,
                ),
            )
            for root in roots
        ]
        probabilities = _score_grouped_probabilities(
            premise=context_text,
            hypotheses=[candidate.hypothesis for candidate in candidates],
            model_name=scoring_config.model_name,
            nli_config=nli_config,
        )
        best_i = max(range(len(candidates)), key=lambda i: probabilities[i])
        chosen = candidates[best_i]
        current_id = str(chosen.class_id)
        selected_path.append(
            {
                "parent_id": parent_id,
                "child_id": current_id,
                "edge_kind": "root",
                "edge_weight": float(probabilities[best_i]),
            }
        )
    else:
        current_id = str(roots[0])

    depth = 0
    while True:
        if traversal_config.max_depth is not None and depth >= traversal_config.max_depth:
            break

        if is_ontology_leaf(graph, current_id):
            break

        child_ids = ontology_children(graph, current_id)
        candidates: list[_Candidate] = [
            _Candidate(
                class_id=child_id,
                edge_kind="child",
                hypothesis=_candidate_hypothesis(
                    graph=graph,
                    candidate_class_id=child_id,
                    edge_kind="child",
                    subject=subject,
                    scoring_config=scoring_config,
                ),
            )
            for child_id in child_ids
        ]

        if traversal_config.include_stay_option and not traversal_config.force_leaf:
            candidates.append(
                _Candidate(
                    class_id=current_id,
                    edge_kind="stay",
                    hypothesis=_candidate_hypothesis(
                        graph=graph,
                        candidate_class_id=current_id,
                        edge_kind="stay",
                        subject=subject,
                        scoring_config=scoring_config,
                    ),
                )
            )

        if not candidates:
            break

        probabilities = _score_grouped_probabilities(
            premise=context_text,
            hypotheses=[candidate.hypothesis for candidate in candidates],
            model_name=scoring_config.model_name,
            nli_config=nli_config,
        )
        best_i = max(range(len(candidates)), key=lambda i: probabilities[i])
        chosen = candidates[best_i]
        chosen_probability = float(probabilities[best_i])

        if chosen.edge_kind == "stay":
            selected_path.append(
                {
                    "parent_id": current_id,
                    "child_id": None,
                    "edge_kind": "stay",
                    "edge_weight": chosen_probability,
                }
            )
            break

        child_id = str(chosen.class_id)
        selected_path.append(
            {
                "parent_id": current_id,
                "child_id": child_id,
                "edge_kind": "child",
                "edge_weight": chosen_probability,
            }
        )
        current_id = child_id
        depth += 1

    return selected_path


def _evidence_record_for_mention(
    *,
    graph: nx.DiGraph,
    record: Any,
    traversal_config: OntologyTraversalConfig,
    scoring_config: OntologyScoringConfig,
    mention_weight_config: OntologyMentionWeightConfig,
    nli_config: DirectNLIConfig,
) -> dict[str, Any]:
    selected_path = _selected_path_for_mention(
        graph=graph,
        context_text=record.context_text,
        subject=record.subject,
        traversal_config=traversal_config,
        scoring_config=scoring_config,
        nli_config=nli_config,
    )
    mention_weight_raw = _score_mention_weight_raw(
        context_text=record.context_text,
        subject=record.subject,
        scoring_config=scoring_config,
        mention_weight_config=mention_weight_config,
        nli_config=nli_config,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_id": int(record.cluster_id),
        "subject": record.subject,
        "mention_index_in_cluster": int(record.mention_index_in_cluster),
        "mention_id": int(record.mention_id),
        "mention_text": str(record.mention_text),
        "mention_start": int(record.mention_start),
        "mention_end": int(record.mention_end),
        "sentence_index": int(record.sentence_index),
        "mention_render_rule": record.mention_render_rule,
        "mention_render_was_changed": bool(record.mention_render_was_changed),
        "context_text": record.context_text,
        "normalized_context_text": record.normalized_context_text,
        "original_context_text": record.original_context_text,
        "rendered_context_text": record.rendered_context_text,
        "selected_path": selected_path,
        "mention_weight_raw": mention_weight_raw,
    }


def export_ontology_evidence_jsonl_for_cluster(
    *,
    doc: Any,
    graph: nx.DiGraph,
    cluster_id: int,
    jsonl_path: str | Path,
    n_mentions: int | None,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
    traversal_config: OntologyTraversalConfig | None = None,
    scoring_config: OntologyScoringConfig | None = None,
    mention_weight_config: OntologyMentionWeightConfig | None = None,
    nli_config: DirectNLIConfig | None = None,
    chunk_size: int = 16,
    overwrite_jsonl: bool = False,
    resume_from_jsonl: bool = True,
    print_progress: bool = True,
) -> Path:
    """Export compact ontology mention evidence for one cluster."""

    validate_ontology_graph(graph)

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    coref_layer = require_coref_layer(doc)
    if cluster_id not in coref_layer.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_jsonl and jsonl_path.exists():
        jsonl_path.unlink()

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    traversal_config = traversal_config or OntologyTraversalConfig()
    scoring_config = scoring_config or OntologyScoringConfig()
    mention_weight_config = mention_weight_config or OntologyMentionWeightConfig()
    nli_config = nli_config or DirectNLIConfig()

    subject = canonical_name_for_cluster(doc, cluster_id)
    total_mentions_in_cluster = len(mention_ids_for_cluster(doc, cluster_id))

    if print_progress:
        print("=" * 100)
        print("Ontology evidence JSONL export")
        print(f"cluster_id: {cluster_id}")
        print(f"subject: {subject!r}")
        print(f"requested mentions: {n_mentions}")
        print(f"total mentions in cluster: {total_mentions_in_cluster}")
        print(f"jsonl_path: {jsonl_path}")
        print(f"device: {_device_name()}")
        print("=" * 100)

    extraction_start = time.perf_counter()
    records = mention_records_for_cluster(
        doc,
        cluster_id,
        subject=subject,
        n_mentions=n_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        context_config=context_config,
        rendering_config=rendering_config,
    )
    extraction_elapsed = time.perf_counter() - extraction_start

    completed_mention_ids: set[int] = set()
    if resume_from_jsonl and jsonl_path.exists() and not overwrite_jsonl:
        completed_mention_ids = completed_mention_ids_from_jsonl(jsonl_path)

    if completed_mention_ids:
        before = len(records)
        records = [record for record in records if record.mention_id not in completed_mention_ids]
        skipped_from_existing_jsonl = before - len(records)
    else:
        skipped_from_existing_jsonl = 0

    if print_progress:
        print(f"extracted records: {len(records) + skipped_from_existing_jsonl}")
        print(f"already completed in JSONL: {skipped_from_existing_jsonl}")
        print(f"records left to score/write: {len(records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")

    n_records_to_score = len(records)
    n_chunks = (n_records_to_score + chunk_size - 1) // chunk_size if n_records_to_score else 0
    newly_written_records = 0
    scoring_start = time.perf_counter()

    for chunk_index, start in enumerate(range(0, n_records_to_score, chunk_size), start=1):
        end = min(start + chunk_size, n_records_to_score)
        chunk_records = records[start:end]
        chunk_start = time.perf_counter()

        for record in chunk_records:
            payload = _evidence_record_for_mention(
                graph=graph,
                record=record,
                traversal_config=traversal_config,
                scoring_config=scoring_config,
                mention_weight_config=mention_weight_config,
                nli_config=nli_config,
            )
            append_jsonl(jsonl_path, payload)
            newly_written_records += 1

        chunk_elapsed = time.perf_counter() - chunk_start
        if print_progress:
            done_total = skipped_from_existing_jsonl + newly_written_records
            print(
                f"[chunk {chunk_index}/{n_chunks}] "
                f"mentions={start}:{end} | "
                f"chunk_time={chunk_elapsed:.2f}s | "
                f"done_total={done_total}/{n_records_to_score + skipped_from_existing_jsonl}"
            )

        del chunk_records
        gc.collect()

    if print_progress:
        elapsed = time.perf_counter() - scoring_start
        print("=" * 100)
        print("ONTOLOGY EVIDENCE JSONL EXPORT COMPLETE")
        print(f"new records written: {newly_written_records}")
        print(f"skipped from existing JSONL: {skipped_from_existing_jsonl}")
        print(f"elapsed scoring/export time: {elapsed:.2f}s")
        print(f"jsonl saved to: {jsonl_path}")
        print("=" * 100)

    return jsonl_path


def _all_cluster_ids_from_doc(doc: Any) -> list[int]:
    """Return every coreference cluster ID in deterministic order."""

    coref_layer = require_coref_layer(doc)
    cluster_ids = sorted(int(cluster_id) for cluster_id in coref_layer.clusters)
    if not cluster_ids:
        raise ValueError("doc._.coref_layer has no clusters to ontology-type.")
    return cluster_ids


def export_ontology_evidence_jsonls(
    doc: Any,
    graph: nx.DiGraph,
    config: OntologyEvidenceExportConfig,
) -> dict[int, Path]:
    """Export ontology evidence JSONLs for every cluster in ``doc._.coref_layer``."""

    validate_ontology_graph(graph)
    cluster_ids = _all_cluster_ids_from_doc(doc)

    output_dir = ontology_typing_output_dir(
        output_root=config.output_root,
        n_mentions_per_cluster=config.n_mentions_per_cluster,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.print_progress:
        print("=" * 100)
        print("All-cluster ontology evidence export")
        print(f"cluster source: doc._.coref_layer.clusters")
        print(f"n_clusters: {len(cluster_ids)}")
        print(f"cluster_ids: {cluster_ids}")
        print(f"n_mentions_per_cluster: {config.n_mentions_per_cluster}")
        print(f"output_dir: {output_dir}")
        print("=" * 100)

    jsonl_paths: dict[int, Path] = {}

    for cluster_position, cluster_id in enumerate(cluster_ids, start=1):
        subject = canonical_name_for_cluster(doc, cluster_id)
        jsonl_path = default_cluster_jsonl_path(
            output_dir,
            cluster_id=cluster_id,
            subject=subject,
            n_mentions=config.n_mentions_per_cluster,
        )
        per_cluster_seed = _cluster_random_seed(config.random_seed, cluster_id)

        if config.print_progress:
            print()
            print("-" * 100)
            print(f"cluster {cluster_position}/{len(cluster_ids)}")
            print(f"cluster_id: {cluster_id}")
            print(f"subject: {subject!r}")
            print(f"jsonl_path: {jsonl_path}")
            print(f"per_cluster_seed: {per_cluster_seed}")
            print("-" * 100)

        export_ontology_evidence_jsonl_for_cluster(
            doc=doc,
            graph=graph,
            cluster_id=cluster_id,
            jsonl_path=jsonl_path,
            n_mentions=config.n_mentions_per_cluster,
            random_seed=per_cluster_seed,
            sort_sample_by_cluster_order=config.sort_sample_by_cluster_order,
            context_config=config.context_config,
            rendering_config=config.rendering_config,
            traversal_config=config.traversal_config,
            scoring_config=config.scoring_config,
            mention_weight_config=config.mention_weight_config,
            nli_config=config.nli_config,
            chunk_size=config.chunk_size,
            overwrite_jsonl=config.overwrite_jsonl,
            resume_from_jsonl=config.resume_from_jsonl,
            print_progress=config.print_progress,
        )
        jsonl_paths[int(cluster_id)] = jsonl_path

    return jsonl_paths
