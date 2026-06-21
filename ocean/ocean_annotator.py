"""OCEAN CSV annotator.

This module reads raw probability CSVs produced by ``ocean_probability_scoring``
and creates a fresh ``doc._.ocean_layer``. It does not decide which clusters
should have been scored and does not inspect semantic types.
"""

from __future__ import annotations

from coreference.coref_schema import require_coref_layer
from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Any

import pandas as pd

from ocean.ocean_schema import (
    OCEAN_TRAITS,
    ClusterOceanAnnotation,
    MentionOceanAnnotation,
    OceanLayer,
    OceanTraitScores,
    OceanWeightEvidence,
    TraitRawEvidence,
    register_spacy_ocean_extension,
)


__all__ = [
    "OceanAnnotationConfig",
    "OceanAnnotationError",
    "collapse_bipolar",
    "collapse_ocean_weight",
    "annotate_doc_with_ocean_folder",
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


# def _require_coref_layer(doc: Any) -> Any:
#     if not hasattr(doc, "_") or not hasattr(doc._, "coref_layer"):
#         raise OceanAnnotationError("doc has no doc._.coref_layer")
#     coref_layer = doc._.coref_layer
#     if coref_layer is None:
#         raise OceanAnnotationError("doc._.coref_layer is None")
#     return coref_layer


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


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _validate_row_alignment(
    *,
    row: Any,
    coref_layer: Any,
    csv_path: Path,
    row_index: int,
) -> tuple[int, int]:
    cluster_id = int(row.cluster_id)
    mention_id = int(row.mention_id)

    if cluster_id not in coref_layer.clusters:
        raise OceanAnnotationError(
            f"CSV references unknown cluster_id={cluster_id}. CSV: {csv_path}, row={row_index}"
        )

    if mention_id not in coref_layer.mentions:
        raise OceanAnnotationError(
            f"CSV references unknown mention_id={mention_id}. CSV: {csv_path}, row={row_index}"
        )

    mention = coref_layer.mentions[mention_id]

    if int(mention.cluster_id) != cluster_id:
        raise OceanAnnotationError(
            f"CSV/coref cluster mismatch for mention_id={mention_id}: "
            f"CSV cluster_id={cluster_id}, coref cluster_id={mention.cluster_id}. "
            f"CSV: {csv_path}, row={row_index}"
        )

    if int(row.mention_start) != int(mention.start) or int(row.mention_end) != int(mention.end):
        raise OceanAnnotationError(
            f"CSV/coref span mismatch for mention_id={mention_id}: "
            f"CSV span=({int(row.mention_start)}, {int(row.mention_end)}), "
            f"coref span=({mention.start}, {mention.end}). CSV: {csv_path}, row={row_index}"
        )

    if str(row.mention_text) != str(mention.text):
        raise OceanAnnotationError(
            f"CSV/coref text mismatch for mention_id={mention_id}: "
            f"CSV text={str(row.mention_text)!r}, coref text={str(mention.text)!r}. "
            f"CSV: {csv_path}, row={row_index}"
        )

    return mention_id, cluster_id


def _raw_trait_evidence_from_row(row: Any, trait: str) -> TraitRawEvidence:
    return TraitRawEvidence(
        positive=float(getattr(row, f"{trait}_positive_probability")),
        neutral=float(getattr(row, f"{trait}_neutral_probability")),
        negative=float(getattr(row, f"{trait}_negative_probability")),
    )


def _mention_annotation_from_row(
    *,
    row: Any,
    mention_id: int,
    cluster_id: int,
    csv_path: Path,
    row_index: int,
    config: OceanAnnotationConfig,
) -> MentionOceanAnnotation:
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

    weight_raw = OceanWeightEvidence(
        high=float(row.OCEAN_weight_high_probability),
        medium=float(row.OCEAN_weight_medium_probability),
        low=float(row.OCEAN_weight_low_probability),
    )
    ocean_weight = collapse_ocean_weight(
        high=weight_raw.high,
        medium=weight_raw.medium,
        low=weight_raw.low,
        rounding_digits=config.rounding_digits,
    )

    return MentionOceanAnnotation(
        mention_id=mention_id,
        cluster_id=cluster_id,
        raw=raw,
        scores=OceanTraitScores.from_mapping(collapsed_scores),
        ocean_weight=ocean_weight,
        ocean_weight_raw=weight_raw,
        source_csv_path=str(csv_path),
        source_row_index=row_index,
        context_text=getattr(row, "context_text", None),
        normalized_context_text=getattr(row, "normalized_context_text", None),
        original_context_text=getattr(row, "original_context_text", None),
        rendered_context_text=getattr(row, "rendered_context_text", None),
        mention_render_rule=getattr(row, "mention_render_rule", None),
        mention_render_was_changed=_bool_or_none(getattr(row, "mention_render_was_changed", None)),
        collapse_method=config.collapse_method,
    )


def _aggregate_cluster_trait(
    mention_annotations: list[MentionOceanAnnotation],
    trait: str,
    *,
    neutral_score: float,
    rounding_digits: int,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0

    for annotation in mention_annotations:
        raw = annotation.raw[trait]
        effective_weight = annotation.ocean_weight * (1.0 - raw.neutral)
        if effective_weight <= 0.0:
            continue
        weighted_sum += annotation.scores[trait] * effective_weight
        total_weight += effective_weight

    if total_weight <= 1e-9:
        return float(neutral_score)

    return round(float(weighted_sum / total_weight), rounding_digits)


def _cluster_annotation_from_mentions(
    *,
    cluster_id: int,
    mention_annotations: list[MentionOceanAnnotation],
    config: OceanAnnotationConfig,
) -> ClusterOceanAnnotation:
    scores = {
        trait: _aggregate_cluster_trait(
            mention_annotations,
            trait,
            neutral_score=config.neutral_score,
            rounding_digits=config.rounding_digits,
        )
        for trait in OCEAN_TRAITS
    }
    source_csv_paths = sorted({annotation.source_csv_path for annotation in mention_annotations})

    return ClusterOceanAnnotation(
        cluster_id=cluster_id,
        scores=OceanTraitScores.from_mapping(scores),
        n_mentions_scored=len(mention_annotations),
        source_csv_paths=source_csv_paths,
        aggregation_method=config.aggregation_method,
        collapse_method=config.collapse_method,
    )


def annotate_doc_with_ocean_folder(
    doc: Any,
    folder_path: str | Path,
    *,
    config: OceanAnnotationConfig | None = None,
    pattern: str = "OCEAN_scores_cluster_*.csv",
) -> Any:
    """Annotate ``doc`` with a fresh ``doc._.ocean_layer`` from a CSV folder.

    The annotator processes every matching CSV in the folder. It does not decide
    whether those CSVs should exist and does not inspect cluster semantic types.
    """

    config = config or OceanAnnotationConfig()
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise OceanAnnotationError(f"OCEAN folder does not exist or is not a directory: {folder}")

    csv_paths = sorted(folder.glob(pattern))
    if not csv_paths:
        raise OceanAnnotationError(f"No OCEAN CSV files found in {folder} with pattern {pattern!r}")

    coref_layer = require_coref_layer(doc)
    register_spacy_ocean_extension()

    ocean_layer = OceanLayer(source_folder=str(folder))
    mentions_by_cluster: dict[int, list[MentionOceanAnnotation]] = {}
    seen_cluster_ids: set[int] = set()
    seen_mention_ids: set[int] = set()

    for csv_path in csv_paths:
        df = _read_csv(csv_path)
        if df.empty:
            raise OceanAnnotationError(f"OCEAN CSV is empty: {csv_path}")

        _warn_deprecated_final_columns(df, csv_path=csv_path)
        _validate_required_columns(df, csv_path=csv_path)
        df = _coerce_required_numeric_columns(df, csv_path=csv_path)
        cluster_id = _require_single_cluster_id(df, csv_path=csv_path)

        if cluster_id in seen_cluster_ids:
            raise OceanAnnotationError(
                f"Duplicate OCEAN CSVs for cluster_id={cluster_id}. "
                f"Ambiguous cluster annotation in folder {folder}"
            )
        seen_cluster_ids.add(cluster_id)

        cluster_mentions: list[MentionOceanAnnotation] = []

        for row_index, row in enumerate(df.itertuples(index=False), start=0):
            mention_id, row_cluster_id = _validate_row_alignment(
                row=row,
                coref_layer=coref_layer,
                csv_path=csv_path,
                row_index=row_index,
            )

            if mention_id in seen_mention_ids:
                raise OceanAnnotationError(
                    f"Duplicate mention_id={mention_id} across OCEAN CSVs in {folder}."
                )
            seen_mention_ids.add(mention_id)

            annotation = _mention_annotation_from_row(
                row=row,
                mention_id=mention_id,
                cluster_id=row_cluster_id,
                csv_path=csv_path,
                row_index=row_index,
                config=config,
            )
            ocean_layer.mentions[mention_id] = annotation
            cluster_mentions.append(annotation)

        mentions_by_cluster[cluster_id] = cluster_mentions

    for cluster_id, mention_annotations in mentions_by_cluster.items():
        ocean_layer.clusters[cluster_id] = _cluster_annotation_from_mentions(
            cluster_id=cluster_id,
            mention_annotations=mention_annotations,
            config=config,
        )

    doc._.ocean_layer = ocean_layer
    return doc
