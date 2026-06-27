"""CSV-first OCEAN probability scoring.

This module is the model-inference/export stage of the OCEAN staging pipeline.
It consumes ``doc._.annotation_layer.entities`` and an explicit list of ``cluster_ids``.
It does not choose clusters by semantic type, does not annotate the Doc, and
never mutates ``doc._.annotation_layer.entities``.

Output contract:
    ``./outputs/OCEAN_profiles/{n_mentions}/OCEAN_scores_cluster_*.csv``

The CSV raw probability columns are the source of truth. Final collapsed OCEAN
scores are intentionally computed later by ``ocean_annotator.py``.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field, is_dataclass
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import re
from typing import Any, Iterable, Optional


from annotation_layer.entity_annotations import OCEAN_TRAITS
from annotation_layer.spacy_extension import require_entities

from neural_runtime.mentions import (
    ContextConfig as OceanContextConfig,
    MentionRenderingConfig as OceanMentionRenderingConfig,
    MentionRecord,
    canonical_name_for_cluster,
    cluster_random_seed,
    mention_ids_for_cluster,
    mention_records_for_cluster,
    normalize_context_for_dedup,
)
from neural_runtime.nli import (
    DirectNLIConfig as OceanDirectNLIConfig,
    device_name,
    direct_entailment_logits_for_pairs,
    release_chunk_memory,
    softmax_values,
    sync_cuda_if_available,
)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE",
    "OCEAN_LABELS",
    "OCEAN_WEIGHT_LABELS",
    "OceanContextConfig",
    "OceanMentionRenderingConfig",
    "OceanDirectNLIConfig",
    "OceanScoringConfig",
    "OceanWeightConfig",
    "OceanProbabilityExportConfig",
    "MentionRecord",
    "normalize_context_for_dedup",
    "mention_ids_for_cluster",
    "canonical_name_for_cluster",
    "mention_records_for_cluster",
    "ocean_profiles_output_dir",
    "default_cluster_csv_path",
    "export_ocean_probability_csv_for_cluster",
    "export_ocean_probability_csvs",
]


# =============================================================================
# Configuration and labels
# =============================================================================

DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_HYPOTHESIS_TEMPLATE = "This text shows {}."
DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE = "{subject} shows {} in this text."
PROBABILITY_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class OceanScoringConfig:
    """Runtime configuration for direct-NLI OCEAN probability scoring."""

    model_name: str = DEFAULT_MODEL_NAME
    generic_hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE
    subject_hypothesis_template: str = DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE
    subject_aware: bool = True
    truncation: bool = True
    multi_label: bool = False  # retained for config readability; direct scorer uses grouped softmax



@dataclass(frozen=True)
class OceanWeightConfig:
    """Configuration for raw OCEAN_weight probability scoring.

    OCEAN_weight is always scored in the refactored pipeline because
    ``ocean_annotator.py`` requires the three raw weight probability columns.
    """

    hypothesis_template: str = (
        "In this text, {subject}'s behavior or inner state provides {}."
    )



@dataclass(frozen=True)
class OceanProbabilityExportConfig:
    """Configuration for multi-cluster OCEAN probability export.

    ``cluster_ids`` is intentionally explicit. The caller decides how that list
    is produced. This module does not know or care whether the list came from
    manual testing, semantic typing, ontology fitting, or some future selector.
    """

    cluster_ids: list[int]
    n_mentions_per_cluster: int | None

    output_root: str | Path = "./outputs"
    random_seed: int | None = None
    sort_sample_by_cluster_order: bool = True

    overwrite_csv: bool = False
    resume_from_csv: bool = True
    use_sqlite_cache: bool = True

    chunk_size: int = 16

    context_config: OceanContextConfig = field(default_factory=OceanContextConfig)
    rendering_config: OceanMentionRenderingConfig = field(default_factory=OceanMentionRenderingConfig)
    scoring_config: OceanScoringConfig = field(default_factory=OceanScoringConfig)
    weight_config: OceanWeightConfig = field(default_factory=OceanWeightConfig)
    nli_config: OceanDirectNLIConfig = field(default_factory=OceanDirectNLIConfig)
    return_dataframes: bool = False
    print_progress: bool = True


OCEAN_LABELS: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "high openness: curiosity, imagination, exploration, intellectual interest, "
            "openness to new experiences"
        ),
        "negative": (
            "low openness: rigidity, lack of curiosity, conventionality, "
            "resistance to new experiences"
        ),
        "neutral": "no clear evidence about openness, curiosity, imagination, or exploration",
    },
    "conscientiousness": {
        "positive": (
            "high conscientiousness: carefulness, planning, responsibility, "
            "discipline, persistence, duty"
        ),
        "negative": (
            "low conscientiousness: carelessness, impulsiveness, irresponsibility, "
            "disorganization, lack of planning"
        ),
        "neutral": "no clear evidence about conscientiousness, carefulness, planning, or responsibility",
    },
    "extraversion": {
        "positive": (
            "high extraversion: sociability, friendliness, outgoing social engagement, "
            "enthusiasm, enjoyment of interaction"
        ),
        "negative": (
            "low extraversion: social withdrawal, reserve, avoidance of interaction, "
            "quietness, lack of social engagement"
        ),
        "neutral": "no clear evidence about extraversion, sociability, assertiveness, or outgoing energy",
    },
    "agreeableness": {
        "positive": (
            "high agreeableness: kindness, compassion, cooperation, helpfulness, "
            "trust, concern for others"
        ),
        "negative": (
            "low agreeableness: hostility, selfishness, cruelty, refusal to help, "
            "uncooperative behavior"
        ),
        "neutral": "no clear evidence about agreeableness, kindness, compassion, cooperation, or hostility",
    },
    "neuroticism": {
        "positive": "high neuroticism: fear, anxiety, sadness, distress, emotional instability, worry",
        "negative": "low neuroticism: calmness, confidence, emotional stability, composure, lack of distress",
        "neutral": "no clear evidence about neuroticism, fear, anxiety, sadness, worry, or emotional stability",
    },
}


OCEAN_WEIGHT_LABELS: dict[str, str] = {
    "high": (
        "strong personality evidence about the character: behavior, emotion, "
        "motivation, decision, social interaction, preference, fear, desire, "
        "moral choice, reaction, or stable tendency"
    ),
    "medium": (
        "weak or ambiguous personality evidence about the character: indirect, "
        "minor, or context-dependent evidence about behavior, emotion, or motivation"
    ),
    "low": (
        "no useful personality evidence about the character: plot movement, "
        "physical description, setting, object description, bookkeeping dialogue, "
        "or events that do not reveal personality"
    ),
}


def _hypothesis_template(config: OceanScoringConfig, subject: str | None) -> str:
    if config.subject_aware and subject:
        return config.subject_hypothesis_template.replace("{subject}", subject)
    return config.generic_hypothesis_template


def _replace_subject_in_template(template: str, subject: str | None) -> str:
    if "{subject}" in template:
        return template.replace("{subject}", subject or "the character")
    return template


def _format_candidate_hypothesis(template: str, label_text: str) -> str:
    try:
        return template.format(label_text)
    except Exception as exc:
        raise ValueError(
            "Hypothesis template must contain exactly one positional {} placeholder "
            f"after subject replacement. Got: {template!r}"
        ) from exc


@dataclass(frozen=True)
class ProbabilityTask:
    task_name: str
    label_texts: dict[str, str]
    hypothesis_template: str


def _trait_probability_tasks(
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
) -> list[ProbabilityTask]:
    hypothesis_template = _hypothesis_template(scoring_config, subject)
    return [
        ProbabilityTask(
            task_name=trait,
            label_texts={
                "positive": labels["positive"],
                "negative": labels["negative"],
                "neutral": labels["neutral"],
            },
            hypothesis_template=hypothesis_template,
        )
        for trait, labels in trait_labels.items()
    ]


def _weight_probability_task(*, subject: str | None, weight_config: OceanWeightConfig) -> ProbabilityTask:
    return ProbabilityTask(
        task_name="OCEAN_weight",
        label_texts={
            "high": OCEAN_WEIGHT_LABELS["high"],
            "medium": OCEAN_WEIGHT_LABELS["medium"],
            "low": OCEAN_WEIGHT_LABELS["low"],
        },
        hypothesis_template=_replace_subject_in_template(
            weight_config.hypothesis_template,
            subject,
        ),
    )


def _score_probability_payloads_for_chunk(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: OceanDirectNLIConfig,
) -> list[dict[str, Any]]:
    tasks = _trait_probability_tasks(
        subject=subject,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
    )
    tasks.append(_weight_probability_task(subject=subject, weight_config=weight_config))

    pair_metadata: list[tuple[int, str, str]] = []
    pairs: list[tuple[str, str]] = []

    for record_index, record in enumerate(records):
        for task in tasks:
            for label_key, label_text in task.label_texts.items():
                hypothesis = _format_candidate_hypothesis(task.hypothesis_template, label_text)
                pairs.append((record.context_text, hypothesis))
                pair_metadata.append((record_index, task.task_name, label_key))

    entailment_logits = direct_entailment_logits_for_pairs(
        pairs,
        model_name=scoring_config.model_name,
        nli_config=nli_config,
    )

    grouped_logits: dict[tuple[int, str], dict[str, float]] = {}
    for (record_index, task_name, label_key), logit in zip(pair_metadata, entailment_logits):
        grouped_logits.setdefault((record_index, task_name), {})[label_key] = logit

    payloads: list[dict[str, Any]] = []
    for record_index, _record in enumerate(records):
        payload: dict[str, Any] = {}
        for trait in OCEAN_TRAITS:
            label_logits = grouped_logits[(record_index, trait)]
            ordered_keys = ["positive", "negative", "neutral"]
            probabilities = softmax_values([label_logits[key] for key in ordered_keys])
            probability_by_key = dict(zip(ordered_keys, probabilities))
            payload[f"{trait}_positive_probability"] = float(probability_by_key["positive"])
            payload[f"{trait}_negative_probability"] = float(probability_by_key["negative"])
            payload[f"{trait}_neutral_probability"] = float(probability_by_key["neutral"])

        label_logits = grouped_logits[(record_index, "OCEAN_weight")]
        ordered_keys = ["high", "medium", "low"]
        probabilities = softmax_values([label_logits[key] for key in ordered_keys])
        probability_by_key = dict(zip(ordered_keys, probabilities))
        payload["OCEAN_weight_high_probability"] = float(probability_by_key["high"])
        payload["OCEAN_weight_medium_probability"] = float(probability_by_key["medium"])
        payload["OCEAN_weight_low_probability"] = float(probability_by_key["low"])

        payloads.append(payload)

    del tasks, pair_metadata, pairs, entailment_logits, grouped_logits
    release_chunk_memory()
    return payloads


# =============================================================================
# SQLite cache and CSV export
# =============================================================================


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


class SQLiteProbabilityCache:
    """SQLite cache for resumable direct-NLI probability payloads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ocean_probability_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at_unix REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        row = self.connection.execute(
            "SELECT payload_json FROM ocean_probability_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set_many(self, items: Iterable[tuple[str, dict[str, Any]]]) -> None:
        now = time.time()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO ocean_probability_cache(cache_key, payload_json, created_at_unix)
            VALUES (?, ?, ?)
            """,
            [(key, _stable_json(_json_ready(payload)), now) for key, payload in items],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteProbabilityCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _probability_cache_key(
    record: MentionRecord,
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: OceanDirectNLIConfig,
) -> str:
    payload = {
        "schema_version": PROBABILITY_CACHE_SCHEMA_VERSION,
        "normalized_context_text": normalize_context_for_dedup(record.context_text),
        "subject": subject,
        "scoring_config": scoring_config,
        "trait_labels": trait_labels,
        "weight_config": {"config": weight_config, "labels": OCEAN_WEIGHT_LABELS},
        "nli_config": {
            "truncation": nli_config.truncation,
            "max_length": nli_config.max_length,
            "device": nli_config.device,
        },
    }
    return hashlib.sha256(_stable_json(_json_ready(payload)).encode("utf-8")).hexdigest()


def _score_probability_rows_for_chunk_cached(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: OceanDirectNLIConfig,
    cache: SQLiteProbabilityCache | None,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> tuple[list[dict[str, Any]], int, int]:
    if cache is None:
        payloads = _score_probability_payloads_for_chunk(
            records,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        rows = [
            _probability_payload_to_row(
                record,
                payload,
                chunk_index=chunk_index,
                elapsed_seconds_at_completion=elapsed_seconds_at_completion,
            )
            for record, payload in zip(records, payloads)
        ]
        return rows, 0, len(records)

    payloads_by_index: list[dict[str, Any] | None] = [None] * len(records)
    miss_records: list[MentionRecord] = []
    miss_indexes: list[int] = []
    miss_keys: list[str] = []
    cache_hits = 0

    for record_index, record in enumerate(records):
        key = _probability_cache_key(
            record,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        cached_payload = cache.get(key)
        if cached_payload is None:
            miss_records.append(record)
            miss_indexes.append(record_index)
            miss_keys.append(key)
        else:
            payloads_by_index[record_index] = cached_payload
            cache_hits += 1

    if miss_records:
        miss_payloads = _score_probability_payloads_for_chunk(
            miss_records,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        cache.set_many(zip(miss_keys, miss_payloads))
        for record_index, payload in zip(miss_indexes, miss_payloads):
            payloads_by_index[record_index] = payload

    rows = [
        _probability_payload_to_row(
            record,
            payload,
            chunk_index=chunk_index,
            elapsed_seconds_at_completion=elapsed_seconds_at_completion,
        )
        for record, payload in zip(records, payloads_by_index)
        if payload is not None
    ]
    cache_misses = len(miss_records)
    release_chunk_memory()
    return rows, cache_hits, cache_misses


def _probability_payload_to_row(
    record: MentionRecord,
    payload: dict[str, Any],
    *,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cluster_id": record.cluster_id,
        "subject": record.subject,
        "mention_index_in_cluster": record.mention_index_in_cluster,
        "mention_id": record.mention_id,
        "mention_text": record.mention_text,
        "mention_start": record.mention_start,
        "mention_end": record.mention_end,
        "sentence_index": record.sentence_index,
        "mention_render_rule": record.mention_render_rule,
        "mention_render_was_changed": record.mention_render_was_changed,
        "original_context_text": record.original_context_text,
        "rendered_context_text": record.rendered_context_text or record.context_text,
        "context_text": record.context_text,
        "normalized_context_text": record.normalized_context_text,
        "chunk_index": chunk_index,
        "elapsed_seconds_at_completion": elapsed_seconds_at_completion,
    }
    row.update(payload)
    return row


REQUIRED_RAW_COMPLETION_COLUMNS: tuple[str, ...] = tuple(
    f"{trait}_{polarity}_probability"
    for trait in OCEAN_TRAITS
    for polarity in ("positive", "negative", "neutral")
) + (
    "OCEAN_weight_high_probability",
    "OCEAN_weight_medium_probability",
    "OCEAN_weight_low_probability",
)

DEPRECATED_FINAL_COLUMNS: set[str] = set(OCEAN_TRAITS) | {"OCEAN_weight"}


def _completed_mention_ids_from_csv(csv_path: str | Path) -> set[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    completed: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mention_id" not in reader.fieldnames:
            raise ValueError(f"Cannot resume from {csv_path}: CSV has no mention_id column.")
        missing_columns = set(REQUIRED_RAW_COMPLETION_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Cannot resume from {csv_path}: CSV is missing raw probability columns: "
                f"{sorted(missing_columns)}. Use overwrite_csv=True."
            )
        if DEPRECATED_FINAL_COLUMNS & set(reader.fieldnames):
            raise ValueError(
                f"Cannot resume into legacy CSV with deprecated final columns: {csv_path}. "
                "Use overwrite_csv=True to create the refactored raw-only CSV."
            )
        for row in reader:
            raw_mention_id = row.get("mention_id")
            if raw_mention_id in (None, ""):
                continue
            if any(row.get(column) in (None, "") for column in REQUIRED_RAW_COMPLETION_COLUMNS):
                continue
            completed.add(int(float(raw_mention_id)))
    return completed


def _safe_filename_component(value: Any, *, default: str = "unknown") -> str:
    text = str(value if value is not None else default).strip() or default
    text = normalize_context_for_dedup(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def ocean_profiles_output_dir(
    *,
    output_root: str | Path,
    n_mentions_per_cluster: int | None,
) -> Path:
    n_part = "all" if n_mentions_per_cluster is None else str(n_mentions_per_cluster)
    return Path(output_root) / "OCEAN_profiles" / n_part


def default_cluster_csv_path(
    output_dir: str | Path,
    *,
    cluster_id: int,
    subject: str,
    n_mentions: int | None,
) -> Path:
    subject_part = _safe_filename_component(subject, default="unknown_subject")
    n_part = "all" if n_mentions is None else str(n_mentions)
    return Path(output_dir) / f"OCEAN_scores_cluster_{int(cluster_id)}_{subject_part}_{n_part}.csv"


def export_ocean_probability_csv_for_cluster(
    *,
    doc: Any,
    cluster_id: int,
    csv_path: str | Path,
    n_mentions: int | None,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: OceanContextConfig | None = None,
    rendering_config: OceanMentionRenderingConfig | None = None,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    weight_config: OceanWeightConfig | None = None,
    nli_config: OceanDirectNLIConfig | None = None,
    chunk_size: int = 16,
    overwrite_csv: bool = False,
    resume_from_csv: bool = True,
    use_sqlite_cache: bool = True,
    cache_path: str | Path | None = None,
    return_dataframe: bool = False,
    print_progress: bool = True,
) -> Any:
    import pandas as pd

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    entities = require_entities(doc)
    if cluster_id not in entities.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = csv_path.with_suffix(csv_path.suffix + ".cache.sqlite3")
    cache_path = Path(cache_path)

    if overwrite_csv and csv_path.exists():
        csv_path.unlink()

    context_config = context_config or OceanContextConfig(deduplicate=False)
    rendering_config = rendering_config or OceanMentionRenderingConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    weight_config = weight_config or OceanWeightConfig()
    nli_config = nli_config or OceanDirectNLIConfig()

    subject = canonical_name_for_cluster(doc, cluster_id)
    total_mentions_in_cluster = len(mention_ids_for_cluster(doc, cluster_id))

    if print_progress:
        print("=" * 100)
        print("OCEAN probability CSV export")
        print(f"cluster_id: {cluster_id}")
        print(f"subject: {subject!r}")
        print(f"requested mentions: {n_mentions}")
        print(f"total mentions in cluster: {total_mentions_in_cluster}")
        print(f"csv_path: {csv_path}")
        print(f"cache_path: {cache_path if use_sqlite_cache else None}")
        print(f"device: {device_name(nli_config.device)}")
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
    if resume_from_csv and csv_path.exists() and not overwrite_csv:
        completed_mention_ids = _completed_mention_ids_from_csv(csv_path)

    if completed_mention_ids:
        before = len(records)
        records = [record for record in records if record.mention_id not in completed_mention_ids]
        skipped_from_existing_csv = before - len(records)
    else:
        skipped_from_existing_csv = 0

    if print_progress:
        print(f"extracted records: {len(records) + skipped_from_existing_csv}")
        print(f"already completed in CSV: {skipped_from_existing_csv}")
        print(f"records left to score/write: {len(records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")

    n_records_to_score = len(records)
    n_chunks = (n_records_to_score + chunk_size - 1) // chunk_size if n_records_to_score else 0
    csv_header_written = csv_path.exists() and csv_path.stat().st_size > 0 and not overwrite_csv
    newly_written_rows = 0
    total_cache_hits = 0
    total_cache_misses = 0
    kept_rows: list[dict[str, Any]] = []
    scoring_start = time.perf_counter()

    cache_manager = SQLiteProbabilityCache(cache_path) if use_sqlite_cache else None
    try:
        for chunk_index, start in enumerate(range(0, n_records_to_score, chunk_size), start=1):
            end = min(start + chunk_size, n_records_to_score)
            chunk_records = records[start:end]
            sync_cuda_if_available()
            chunk_start = time.perf_counter()
            elapsed_before = time.perf_counter() - scoring_start
            chunk_rows, cache_hits, cache_misses = _score_probability_rows_for_chunk_cached(
                chunk_records,
                subject=subject,
                scoring_config=scoring_config,
                trait_labels=trait_labels,
                weight_config=weight_config,
                nli_config=nli_config,
                cache=cache_manager,
                chunk_index=chunk_index,
                elapsed_seconds_at_completion=elapsed_before,
            )
            sync_cuda_if_available()
            chunk_elapsed = time.perf_counter() - chunk_start
            elapsed_so_far = time.perf_counter() - scoring_start
            for row in chunk_rows:
                row["elapsed_seconds_at_completion"] = elapsed_so_far

            chunk_df = pd.DataFrame(chunk_rows)
            chunk_df.to_csv(
                csv_path,
                mode="a",
                header=not csv_header_written,
                index=False,
                encoding="utf-8",
            )
            csv_header_written = True

            if return_dataframe:
                kept_rows.extend(chunk_rows)

            newly_written_rows += len(chunk_rows)
            total_cache_hits += cache_hits
            total_cache_misses += cache_misses

            if print_progress:
                n_tasks = len(OCEAN_TRAITS) + 1
                n_pairs = cache_misses * n_tasks * 3
                done_total = skipped_from_existing_csv + newly_written_rows
                print(
                    f"[chunk {chunk_index}/{n_chunks}] "
                    f"mentions={start}:{end} | "
                    f"model_pairs={n_pairs} | "
                    f"cache_hits={cache_hits} | "
                    f"cache_misses={cache_misses} | "
                    f"chunk_time={chunk_elapsed:.2f}s | "
                    f"done_total={done_total}/{n_records_to_score + skipped_from_existing_csv}"
                )

            del chunk_df, chunk_rows, chunk_records
            release_chunk_memory()
    finally:
        if cache_manager is not None:
            cache_manager.close()

    if print_progress:
        print("=" * 100)
        print("OCEAN PROBABILITY CSV EXPORT COMPLETE")
        print(f"new rows written: {newly_written_rows}")
        print(f"skipped from existing CSV: {skipped_from_existing_csv}")
        print(f"cache hits: {total_cache_hits}")
        print(f"cache misses: {total_cache_misses}")
        print(f"csv saved to: {csv_path}")
        print("=" * 100)

    if return_dataframe:
        return pd.DataFrame(kept_rows)
    return pd.DataFrame()


def export_ocean_probability_csvs(
    doc: Any,
    config: OceanProbabilityExportConfig,
) -> dict[int, Path]:
    entities = require_entities(doc)

    if not config.cluster_ids:
        raise ValueError("cluster_ids cannot be empty.")

    output_dir = ocean_profiles_output_dir(
        output_root=config.output_root,
        n_mentions_per_cluster=config.n_mentions_per_cluster,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.print_progress:
        print("=" * 100)
        print("Multi-cluster OCEAN probability export")
        print(f"cluster_ids: {config.cluster_ids}")
        print(f"n_mentions_per_cluster: {config.n_mentions_per_cluster}")
        print(f"output_dir: {output_dir}")
        print("=" * 100)

    csv_paths: dict[int, Path] = {}

    for cluster_position, cluster_id in enumerate(config.cluster_ids, start=1):
        if cluster_id not in entities.clusters:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")

        subject = canonical_name_for_cluster(doc, cluster_id)
        csv_path = default_cluster_csv_path(
            output_dir,
            cluster_id=cluster_id,
            subject=subject,
            n_mentions=config.n_mentions_per_cluster,
        )
        per_cluster_seed = cluster_random_seed(config.random_seed, cluster_id)

        if config.print_progress:
            print()
            print("-" * 100)
            print(f"cluster {cluster_position}/{len(config.cluster_ids)}")
            print(f"cluster_id: {cluster_id}")
            print(f"subject: {subject!r}")
            print(f"csv_path: {csv_path}")
            print(f"per_cluster_seed: {per_cluster_seed}")
            print("-" * 100)

        export_ocean_probability_csv_for_cluster(
            doc=doc,
            cluster_id=cluster_id,
            csv_path=csv_path,
            n_mentions=config.n_mentions_per_cluster,
            random_seed=per_cluster_seed,
            sort_sample_by_cluster_order=config.sort_sample_by_cluster_order,
            context_config=config.context_config,
            rendering_config=config.rendering_config,
            scoring_config=config.scoring_config,
            trait_labels=OCEAN_LABELS,
            weight_config=config.weight_config,
            nli_config=config.nli_config,
            chunk_size=config.chunk_size,
            overwrite_csv=config.overwrite_csv,
            resume_from_csv=config.resume_from_csv,
            use_sqlite_cache=config.use_sqlite_cache,
            return_dataframe=config.return_dataframes,
            print_progress=config.print_progress,
        )
        csv_paths[cluster_id] = csv_path

    return csv_paths
