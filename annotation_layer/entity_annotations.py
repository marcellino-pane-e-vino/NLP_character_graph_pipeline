from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from annotation_layer.ids import ClassIri

OCEAN_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass(frozen=True, slots=True)
class ClusterTypingProfile:
    """Ontology typing payload attached to one entity cluster."""

    class_iri: ClassIri
    confidence: float | None = None
    evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class OceanTraitScores:
    """Final collapsed OCEAN scores on a 0-100 scale."""

    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def as_dict(self) -> dict[str, float]:
        return {
            "openness": float(self.openness),
            "conscientiousness": float(self.conscientiousness),
            "extraversion": float(self.extraversion),
            "agreeableness": float(self.agreeableness),
            "neuroticism": float(self.neuroticism),
        }

    def __getitem__(self, trait: str) -> float:
        if trait not in OCEAN_TRAITS:
            raise KeyError(f"Unknown OCEAN trait: {trait!r}")
        return float(getattr(self, trait))

    @classmethod
    def from_mapping(cls, scores: Mapping[str, float]) -> "OceanTraitScores":
        missing = set(OCEAN_TRAITS) - set(scores)
        if missing:
            raise ValueError(f"Missing OCEAN score(s): {sorted(missing)}")
        return cls(**{trait: float(scores[trait]) for trait in OCEAN_TRAITS})


@dataclass(frozen=True, slots=True)
class ClusterOceanProfile:
    """Aggregated OCEAN payload attached to one entity cluster."""

    scores: OceanTraitScores
    n_mentions_scored: int
    evidence_ref: str | None = None
    aggregation_method: str = "trait_effective_weight"
    collapse_method: str = "soft_linear_bipolar"
