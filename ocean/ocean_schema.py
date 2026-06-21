"""Data structures for the OCEAN staging layer.

This module intentionally does not know how OCEAN scores are produced.
It only defines the annotation objects stored at ``doc._.ocean_layer``.

Design boundary:
    - ``doc._.coref_layer`` remains the stable coreference identity layer.
    - ``doc._.ocean_layer`` is an independent staging layer keyed by
      ``mention_id`` and ``cluster_id`` from the coreference layer.
    - A future final node layer can materialize coref + OCEAN + ontology
      staging layers into one consolidated graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "ocean_schema.py requires spaCy. Install it with: pip install spacy"
    ) from exc


__all__ = [
    "OCEAN_TRAITS",
    "TraitRawEvidence",
    "OceanWeightEvidence",
    "OceanTraitScores",
    "MentionOceanAnnotation",
    "ClusterOceanAnnotation",
    "OceanLayer",
    "register_spacy_ocean_extension",
    "require_ocean_layer",
]


OCEAN_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass
class TraitRawEvidence:
    """Raw three-way evidence for one OCEAN trait."""

    positive: float
    neutral: float
    negative: float


@dataclass
class OceanWeightEvidence:
    """Raw three-way evidence for OCEAN mention relevance/weight."""

    high: float
    medium: float
    low: float


@dataclass
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
    def from_mapping(cls, scores: dict[str, float]) -> "OceanTraitScores":
        missing = set(OCEAN_TRAITS) - set(scores)
        if missing:
            raise ValueError(f"Missing OCEAN score(s): {sorted(missing)}")
        return cls(**{trait: float(scores[trait]) for trait in OCEAN_TRAITS})


@dataclass
class MentionOceanAnnotation:
    """OCEAN annotation for one coreference mention.

    The annotation stores only stable identifiers, never direct references to
    coreference Mention/Cluster objects. This keeps the layer easy to merge into
    a future final node layer.
    """

    mention_id: int
    cluster_id: int

    raw: dict[str, TraitRawEvidence]
    scores: OceanTraitScores

    ocean_weight: float
    ocean_weight_raw: OceanWeightEvidence

    source_csv_path: str
    source_row_index: int

    context_text: str | None = None
    normalized_context_text: str | None = None
    original_context_text: str | None = None
    rendered_context_text: str | None = None

    mention_render_rule: str | None = None
    mention_render_was_changed: bool | None = None

    collapse_method: str = "soft_linear_bipolar"


@dataclass
class ClusterOceanAnnotation:
    """Aggregated OCEAN profile for one coreference cluster."""

    cluster_id: int

    scores: OceanTraitScores

    n_mentions_scored: int
    source_csv_paths: list[str]

    aggregation_method: str = "trait_effective_weight"
    collapse_method: str = "soft_linear_bipolar"


@dataclass
class OceanLayer:
    """Independent OCEAN staging layer stored at ``doc._.ocean_layer``."""

    mentions: dict[int, MentionOceanAnnotation] = field(default_factory=dict)
    clusters: dict[int, ClusterOceanAnnotation] = field(default_factory=dict)

    source_folder: str | None = None

    def mention(self, mention_id: int) -> MentionOceanAnnotation:
        return self.mentions[int(mention_id)]

    def cluster(self, cluster_id: int) -> ClusterOceanAnnotation:
        return self.clusters[int(cluster_id)]

    def maybe_mention(self, mention_id: int) -> MentionOceanAnnotation | None:
        return self.mentions.get(int(mention_id))

    def maybe_cluster(self, cluster_id: int) -> ClusterOceanAnnotation | None:
        return self.clusters.get(int(cluster_id))

    def mention_score(self, mention_id: int, trait: str) -> float:
        return self.mention(mention_id).scores[trait]

    def cluster_score(self, cluster_id: int, trait: str) -> float:
        return self.cluster(cluster_id).scores[trait]

    def scored_mention_ids(self) -> list[int]:
        return sorted(self.mentions)

    def scored_cluster_ids(self) -> list[int]:
        return sorted(self.clusters)

    def summary(self) -> dict[str, int]:
        return {
            "n_ocean_mentions": len(self.mentions),
            "n_ocean_clusters": len(self.clusters),
        }


def register_spacy_ocean_extension(*, force: bool = False) -> None:
    """Register ``doc._.ocean_layer`` as the OCEAN staging extension."""

    if Doc.has_extension("ocean_layer"):
        if force:
            Doc.set_extension("ocean_layer", default=None, force=True)
        return

    Doc.set_extension("ocean_layer", default=None)


def require_ocean_layer(doc: Doc) -> OceanLayer:
    """Return ``doc._.ocean_layer`` or fail early when absent."""

    if not Doc.has_extension("ocean_layer") or doc._.ocean_layer is None:
        raise ValueError("This Doc has no OCEAN layer.")

    return doc._.ocean_layer
