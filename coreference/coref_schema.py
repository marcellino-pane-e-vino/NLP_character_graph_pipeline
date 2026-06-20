from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

try:
    from spacy.tokens import Doc, Span
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "coref_schema.py requires spaCy. Install it with: pip install spacy"
    ) from exc


__all__ = [
    "Mention",
    "MentionRecord",
    "Cluster",
    "ClusterRecord",
    "CorefLayer",
    "register_spacy_coref_extension",
    "require_coref_layer",
]


@dataclass(slots=True)
class Mention:
    """
    One final document-level coreference mention.

    start/end are spaCy token offsets, half-open, so doc[start:end]
    reconstructs the mention span.
    """

    mention_id: int
    cluster_id: int
    start: int
    end: int
    text: str
    head_token_i: int | None = None


MentionRecord = Mention


@dataclass(slots=True)
class Cluster:
    """One final document-level coreference cluster."""

    cluster_id: int
    mention_ids: list[int]
    canonical_name: str
    semantic_type: str = "UNKNOWN"


ClusterRecord = Cluster


@dataclass(slots=True)
class CorefLayer:
    """
    Single coreference wrapper stored at doc._.coref_layer.

    The layer is the final mention/reference substrate. It deliberately does not
    store CSV provenance, mention types, descriptors, or taxonomy data.
    """

    mentions: dict[int, MentionRecord]
    clusters: dict[int, ClusterRecord]
    token_to_mention_ids: dict[int, list[int]] = field(default_factory=dict)
    head_token_to_mention_ids: dict[int, list[int]] = field(default_factory=dict)
    span_to_mention_ids: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def mentions_from_token(self, token_i: int) -> list[Mention]:
        """Return all mentions covering token_i."""
        return [
            self.mentions[mention_id]
            for mention_id in self.token_to_mention_ids.get(token_i, [])
            if mention_id in self.mentions
        ]

    def clusters_from_token(self, token_i: int) -> list[Cluster]:
        """Return all distinct clusters whose mentions cover token_i."""
        clusters: list[Cluster] = []
        seen_cluster_ids: set[int] = set()

        for mention in self.mentions_from_token(token_i):
            if mention.cluster_id in seen_cluster_ids:
                continue
            cluster = self.clusters.get(mention.cluster_id)
            if cluster is None:
                continue
            clusters.append(cluster)
            seen_cluster_ids.add(cluster.cluster_id)

        return clusters

    def mentions_from_head_token(self, token_i: int) -> list[Mention]:
        """Return all mentions whose known syntactic head is token_i."""
        return [
            self.mentions[mention_id]
            for mention_id in self.head_token_to_mention_ids.get(token_i, [])
            if mention_id in self.mentions
        ]

    def mentions_from_span(self, start: int, end: int) -> list[Mention]:
        """Return all mentions with exactly the requested half-open token span."""
        return [
            self.mentions[mention_id]
            for mention_id in self.span_to_mention_ids.get((start, end), [])
            if mention_id in self.mentions
        ]

    def span_for_mention(self, doc: Doc, mention_id: int) -> Span:
        """Return the spaCy Span corresponding to mention_id."""
        mention = self.mentions[mention_id]
        return doc[mention.start : mention.end]

    def spans_for_cluster(self, doc: Doc, cluster_id: int) -> list[Span]:
        """Return spaCy spans for all mentions in a cluster."""
        cluster = self.clusters[cluster_id]
        return [self.span_for_mention(doc, mention_id) for mention_id in cluster.mention_ids]

    def iter_cluster_mentions(self, cluster_id: int) -> Iterable[Mention]:
        """Yield mentions belonging to cluster_id in textual order."""
        cluster = self.clusters[cluster_id]
        for mention_id in cluster.mention_ids:
            yield self.mentions[mention_id]

    def summary(self) -> dict[str, int]:
        """Return compact counts useful for notebook sanity checks."""
        return {
            "n_mentions": len(self.mentions),
            "n_clusters": len(self.clusters),
            "n_indexed_tokens": len(self.token_to_mention_ids),
            "n_indexed_heads": len(self.head_token_to_mention_ids),
            "n_indexed_spans": len(self.span_to_mention_ids),
        }


def register_spacy_coref_extension(*, force: bool = False) -> None:
    """Register doc._.coref_layer as the only spaCy extension for coreference."""
    if Doc.has_extension("coref_layer"):
        if force:
            Doc.set_extension("coref_layer", default=None, force=True)
        return
    Doc.set_extension("coref_layer", default=None)


def require_coref_layer(doc: Doc) -> CorefLayer:
    """Return doc._.coref_layer or fail early when the layer is absent."""
    if not Doc.has_extension("coref_layer") or doc._.coref_layer is None:
        raise ValueError("This Doc has no coreference layer.")
    return doc._.coref_layer
