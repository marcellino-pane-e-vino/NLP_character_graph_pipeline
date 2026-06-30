"""OCEAN CSV annotator.

This module reads raw probability CSVs produced by ``ocean_probability_scoring``
and attaches final cluster-level OCEAN profiles to entity clusters stored in
``doc._.annotation_layer.entities``. Mention-level OCEAN evidence remains in the
CSV artifacts and is not persisted in the core annotation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings
import math

import pandas as pd

from annotation_layer.entity_annotations import (
    OCEAN_TRAITS,
    ClusterOceanProfile,
    OceanTraitScores,
)
from annotation_layer.spacy_extension import require_annotation_layer, require_entities


__all__ = [
    "OceanAnnotationConfig",
    "OceanAnnotationError",
    "collapse_bipolar",
    "collapse_ocean_weight",
    "build_cluster_ocean_profiles_from_folder",
    "attach_ocean_from_folder",
]


@dataclass(frozen=True)
class OceanAnnotationConfig:
    neutral_score: float = 50.0
    evidence_low: float = 0.0
    evidence_high: float = 0.40
    rounding_digits: int = 2

    collapse_method: str = "soft_linear_bipolar"
    weight_method: str = "high_medium_low_expected_value"
    aggregation_method: str = "trait_effective_weight"


class OceanAnnotationError(ValueError):
    """Raised when OCEAN CSV artifacts cannot annotate the given Doc."""


@dataclass(frozen=True, slots=True)
class TraitRawEvidence:
    positive: float
    neutral: float
    negative: float


@dataclass(frozen=True, slots=True)
class MentionOceanScoringRecord:
    mention_id: int
    cluster_id: int
    raw: dict[str, TraitRawEvidence]
    scores: OceanTraitScores
    ocean_weight: float
    source_csv_path: str
    source_row_index: int


IDENTITY_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "mention_id",
    "mention_text",
    "mention_start",
    "mention_end",
)

WEIGHT_PROBABILITY_COLUMNS: tuple[str, ...] = (
    "OCEAN_weight_high_probability",
    "OCEAN_weight_medium_probability",
    "OCEAN_weight_low_probability",
)

DEPRECATED_FINAL_COLUMNS: set[str] = set(OCEAN_TRAITS) | {"OCEAN_weight"}


def _required_raw_trait_columns() -> tuple[str, ...]:
    return tuple(
        f"{trait}_{polarity}_probability"
        for trait in OCEAN_TRAITS
        for polarity in ("positive", "neutral", "negative")
    )


def collapse_bipolar(
    *,
    positive: float,
    negative: float,
    neutral: float,
    evidence_low: float,
    evidence_high: float,
    neutral_score: float = 50.0,
    eps: float = 1e-9,
) -> float:
    """Collapse positive/negative/neutral probabilities into a 0-100 score."""

    if evidence_high <= evidence_low:
        raise ValueError("evidence_high must be greater than evidence_low.")

    bipolar_score = 100.0 * positive / (positive + negative + eps)
    evidence_strength = max(positive, negative) - neutral

    gate = (evidence_strength - evidence_low) / (evidence_high - evidence_low)
    gate = max(0.0, min(1.0, gate))

    return neutral_score + gate * (bipolar_score - neutral_score)


def collapse_ocean_weight(
    *,
    high: float,
    medium: float,
    low: float,
    rounding_digits: int = 2,
) -> float:
    """Collapse high/medium/low OCEAN-weight probabilities into a 0-100 weight."""

    return round(100.0 * high + 50.0 * medium + 0.0 * low, rounding_digits)


def _read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise OceanAnnotationError(f"Could not read OCEAN CSV: {csv_path}") from exc


def _validate_required_columns(df: pd.DataFrame, *, csv_path: Path) -> None:
    required = set(IDENTITY_COLUMNS) | set(_required_raw_trait_columns()) | set(WEIGHT_PROBABILITY_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise OceanAnnotationError(
            f"OCEAN CSV is missing required columns: {missing}. CSV: {csv_path}"
        )


def _warn_deprecated_final_columns(df: pd.DataFrame, *, csv_path: Path) -> None:
    deprecated = sorted(DEPRECATED_FINAL_COLUMNS & set(df.columns))
    if not deprecated:
        return
    warnings.warn(
        "Deprecated final OCEAN columns found in "
        f"{csv_path}: {deprecated}. They will be ignored and recomputed "
        "from raw probabilities.",
        stacklevel=2,
    )


def _require_single_cluster_id(df: pd.DataFrame, *, csv_path: Path) -> int:
    cluster_ids = sorted({int(float(value)) for value in df["cluster_id"].dropna().tolist()})
    if len(cluster_ids) != 1:
        raise OceanAnnotationError(
            f"Each OCEAN CSV must contain exactly one cluster_id. "
            f"Found {cluster_ids} in {csv_path}"
        )
    return cluster_ids[0]


def _coerce_required_numeric_columns(df: pd.DataFrame, *, csv_path: Path) -> pd.DataFrame:
    numeric_columns = list(IDENTITY_COLUMNS[:2]) + [
        "mention_start",
        "mention_end",
    ] + list(_required_raw_trait_columns()) + list(WEIGHT_PROBABILITY_COLUMNS)

    working = df.copy()
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    missing_numeric = [column for column in numeric_columns if working[column].isna().any()]
    if missing_numeric:
        raise OceanAnnotationError(
            f"OCEAN CSV contains non-numeric/missing values in required columns: "
            f"{missing_numeric}. CSV: {csv_path}"
        )

    return working


def _validate_row_alignment(
    *,
    row: Any,
    entities: Any,
    csv_path: Path,
    row_index: int,
) -> tuple[int, int]:
    cluster_id = int(row.cluster_id)
    mention_id = int(row.mention_id)

    if cluster_id not in entities.clusters:
        raise OceanAnnotationError(
            f"CSV references unknown cluster_id={cluster_id}. CSV: {csv_path}, row={row_index}"
        )

    if mention_id not in entities.mentions:
        raise OceanAnnotationError(
            f"CSV references unknown mention_id={mention_id}. CSV: {csv_path}, row={row_index}"
        )

    mention = entities.mentions[mention_id]

    if int(mention.cluster_id) != cluster_id:
        raise OceanAnnotationError(
            f"CSV/entity cluster mismatch for mention_id={mention_id}: "
            f"CSV cluster_id={cluster_id}, entity cluster_id={mention.cluster_id}. "
            f"CSV: {csv_path}, row={row_index}"
        )

    if int(row.mention_start) != int(mention.start) or int(row.mention_end) != int(mention.end):
        raise OceanAnnotationError(
            f"CSV/entity span mismatch for mention_id={mention_id}: "
            f"CSV span=({int(row.mention_start)}, {int(row.mention_end)}), "
            f"entity span=({mention.start}, {mention.end}). CSV: {csv_path}, row={row_index}"
        )

    if str(row.mention_text) != str(mention.text):
        raise OceanAnnotationError(
            f"CSV/entity text mismatch for mention_id={mention_id}: "
            f"CSV text={str(row.mention_text)!r}, entity text={str(mention.text)!r}. "
            f"CSV: {csv_path}, row={row_index}"
        )

    return mention_id, cluster_id


def _raw_trait_evidence_from_row(row: Any, trait: str) -> TraitRawEvidence:
    return TraitRawEvidence(
        positive=float(getattr(row, f"{trait}_positive_probability")),
        neutral=float(getattr(row, f"{trait}_neutral_probability")),
        negative=float(getattr(row, f"{trait}_negative_probability")),
    )


def _mention_scoring_record_from_row(
    *,
    row: Any,
    mention_id: int,
    cluster_id: int,
    csv_path: Path,
    row_index: int,
    config: OceanAnnotationConfig,
) -> MentionOceanScoringRecord:
    raw: dict[str, TraitRawEvidence] = {
        trait: _raw_trait_evidence_from_row(row, trait)
        for trait in OCEAN_TRAITS
    }

    collapsed_scores = {
        trait: round(
            float(
                collapse_bipolar(
                    positive=raw[trait].positive,
                    negative=raw[trait].negative,
                    neutral=raw[trait].neutral,
                    evidence_low=config.evidence_low,
                    evidence_high=config.evidence_high,
                    neutral_score=config.neutral_score,
                )
            ),
            config.rounding_digits,
        )
        for trait in OCEAN_TRAITS
    }

    ocean_weight = collapse_ocean_weight(
        high=float(row.OCEAN_weight_high_probability),
        medium=float(row.OCEAN_weight_medium_probability),
        low=float(row.OCEAN_weight_low_probability),
        rounding_digits=config.rounding_digits,
    )

    return MentionOceanScoringRecord(
        mention_id=mention_id,
        cluster_id=cluster_id,
        raw=raw,
        scores=OceanTraitScores.from_mapping(collapsed_scores),
        ocean_weight=ocean_weight,
        source_csv_path=str(csv_path),
        source_row_index=row_index,
    )


def skewing_activation_function(score: float) -> float:
    """
    Pushes OCEAN scores away from 50 toward 0/100.

    0   -> 0
    50  -> 50
    100 -> 100
    """

    score = max(0.0, min(100.0, float(score)))
    x = (score - 50.0) / 50.0  # maps [0, 100] to [-1, 1]
    y = math.sin((math.pi / 2.0) * x)

    activated = 50.0 + 50.0 * y
    return max(0.0, min(100.0, activated))

def _aggregate_cluster_trait(
    mention_records: list[MentionOceanScoringRecord],
    trait: str,
    *,
    neutral_score: float,
    rounding_digits: int,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0

    for record in mention_records:
        raw = record.raw[trait]
        effective_weight = record.ocean_weight * (1.0 - raw.neutral)
        if effective_weight <= 0.0:
            continue
        weighted_sum += record.scores[trait] * effective_weight
        total_weight += effective_weight

    if total_weight <= 1e-9:
        return float(neutral_score)

    trait_score = weighted_sum / total_weight
    activated_score = skewing_activation_function(trait_score)
    return round(float(activated_score), rounding_digits)


def _cluster_profile_from_mentions(
    *,
    cluster_id: int,
    mention_records: list[MentionOceanScoringRecord],
    config: OceanAnnotationConfig,
) -> ClusterOceanProfile:
    scores = {
        trait: _aggregate_cluster_trait(
            mention_records,
            trait,
            neutral_score=config.neutral_score,
            rounding_digits=config.rounding_digits,
        )
        for trait in OCEAN_TRAITS
    }
    source_csv_paths = sorted({record.source_csv_path for record in mention_records})

    return ClusterOceanProfile(
        scores=OceanTraitScores.from_mapping(scores),
        n_mentions_scored=len(mention_records),
        evidence_ref="|".join(source_csv_paths),
        aggregation_method=config.aggregation_method,
        collapse_method=config.collapse_method,
    )


def build_cluster_ocean_profiles_from_folder(
    doc: Any,
    folder_path: str | Path,
    *,
    config: OceanAnnotationConfig | None = None,
    pattern: str = "OCEAN_scores_cluster_*.csv",
) -> dict[int, ClusterOceanProfile]:
    config = config or OceanAnnotationConfig()
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise OceanAnnotationError(f"OCEAN folder does not exist or is not a directory: {folder}")

    csv_paths = sorted(folder.glob(pattern))
    if not csv_paths:
        raise OceanAnnotationError(f"No OCEAN CSV files found in {folder} with pattern {pattern!r}")

    entities = require_entities(doc)
    records_by_cluster: dict[int, list[MentionOceanScoringRecord]] = {}
    seen_mention_ids: set[int] = set()

    for csv_path in csv_paths:
        df = _read_csv(csv_path)
        if df.empty:
            raise OceanAnnotationError(f"OCEAN CSV is empty: {csv_path}")

        _warn_deprecated_final_columns(df, csv_path=csv_path)
        _validate_required_columns(df, csv_path=csv_path)
        df = _coerce_required_numeric_columns(df, csv_path=csv_path)
        cluster_id = _require_single_cluster_id(df, csv_path=csv_path)

        if cluster_id in records_by_cluster:
            raise OceanAnnotationError(
                f"Duplicate OCEAN CSVs for cluster_id={cluster_id}. "
                f"Ambiguous cluster annotation in folder {folder}"
            )

        cluster_records: list[MentionOceanScoringRecord] = []

        for row_index, row in enumerate(df.itertuples(index=False), start=0):
            mention_id, row_cluster_id = _validate_row_alignment(
                row=row,
                entities=entities,
                csv_path=csv_path,
                row_index=row_index,
            )

            if mention_id in seen_mention_ids:
                raise OceanAnnotationError(
                    f"Duplicate mention_id={mention_id} across OCEAN CSVs in {folder}."
                )
            seen_mention_ids.add(mention_id)

            cluster_records.append(
                _mention_scoring_record_from_row(
                    row=row,
                    mention_id=mention_id,
                    cluster_id=row_cluster_id,
                    csv_path=csv_path,
                    row_index=row_index,
                    config=config,
                )
            )

        records_by_cluster[cluster_id] = cluster_records

    return {
        cluster_id: _cluster_profile_from_mentions(
            cluster_id=cluster_id,
            mention_records=mention_records,
            config=config,
        )
        for cluster_id, mention_records in records_by_cluster.items()
    }


def attach_ocean_from_folder(
    doc: Any,
    folder_path: str | Path,
    *,
    config: OceanAnnotationConfig | None = None,
    pattern: str = "OCEAN_scores_cluster_*.csv",
    overwrite: bool = False,
) -> dict[int, ClusterOceanProfile]:
    profiles = build_cluster_ocean_profiles_from_folder(
        doc=doc,
        folder_path=folder_path,
        config=config,
        pattern=pattern,
    )
    ann = require_annotation_layer(doc)
    ann.require_entities().attach_cluster_ocean_profiles(profiles, overwrite=overwrite)
    ann.mark_ocean_complete()
    return profiles
