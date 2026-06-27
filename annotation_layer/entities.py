from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

try:
    from spacy.tokens import Doc, Span
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "annotation_layer.entities requires spaCy. Install it with: pip install spacy"
    ) from exc

from annotation_layer.entity_annotations import ClusterOceanProfile, ClusterTypingProfile
from annotation_layer.ids import ClusterId, MentionId


@dataclass(frozen=True, slots=True)
class EntityMentionRecord:
    """One final document-level entity mention."""

    mention_id: MentionId
    cluster_id: ClusterId
    start: int
    end: int
    text: str
    head_token_i: int | None = None


@dataclass(slots=True)
class EntityClusterRecord:
    """One final document-level entity cluster."""

    cluster_id: ClusterId
    mention_ids: tuple[MentionId, ...]
    canonical_name: str
    typing: ClusterTypingProfile | None = None
    ocean: ClusterOceanProfile | None = None


@dataclass(slots=True)
class EntitySubLayer:
    """Entity identity substrate stored under doc._.annotation_layer.entities."""

    mentions: dict[MentionId, EntityMentionRecord]
    clusters: dict[ClusterId, EntityClusterRecord]

    token_to_mention_ids: dict[int, tuple[MentionId, ...]] = field(default_factory=dict)
    head_token_to_mention_ids: dict[int, tuple[MentionId, ...]] = field(default_factory=dict)
    span_to_mention_ids: dict[tuple[int, int], tuple[MentionId, ...]] = field(default_factory=dict)
    cluster_name_index: dict[str, ClusterId] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        *,
        mentions: dict[MentionId, EntityMentionRecord],
        clusters: dict[ClusterId, EntityClusterRecord],
    ) -> "EntitySubLayer":
        layer = cls(mentions=mentions, clusters=clusters)
        layer.rebuild_indexes()
        return layer

    def mention(self, mention_id: MentionId | int) -> EntityMentionRecord:
        return self.mentions[int(mention_id)]

    def cluster(self, cluster_id: ClusterId | int) -> EntityClusterRecord:
        return self.clusters[int(cluster_id)]

    def maybe_cluster(self, cluster_id: ClusterId | int) -> EntityClusterRecord | None:
        return self.clusters.get(int(cluster_id))

    def maybe_mention(self, mention_id: MentionId | int) -> EntityMentionRecord | None:
        return self.mentions.get(int(mention_id))

    def cluster_ids(self) -> tuple[ClusterId, ...]:
        return tuple(sorted(self.clusters))

    def mention_ids(self) -> tuple[MentionId, ...]:
        return tuple(sorted(self.mentions))

    def cluster_by_name(self, name: str) -> EntityClusterRecord:
        key = normalize_cluster_name(name)
        return self.cluster(self.cluster_name_index[key])

    def mention_ids_for_cluster(self, cluster_id: ClusterId | int) -> tuple[MentionId, ...]:
        return self.cluster(cluster_id).mention_ids

    def mentions_for_cluster(self, cluster_id: ClusterId | int) -> tuple[EntityMentionRecord, ...]:
        return tuple(self.mention(mid) for mid in self.mention_ids_for_cluster(cluster_id))

    def iter_cluster_mentions(self, cluster_id: ClusterId | int) -> Iterable[EntityMentionRecord]:
        for mention_id in self.mention_ids_for_cluster(cluster_id):
            yield self.mention(mention_id)

    def mentions_from_token(self, token_i: int) -> tuple[EntityMentionRecord, ...]:
        return tuple(
            self.mentions[mention_id]
            for mention_id in self.token_to_mention_ids.get(int(token_i), ())
            if mention_id in self.mentions
        )

    def clusters_from_token(self, token_i: int) -> tuple[EntityClusterRecord, ...]:
        clusters: list[EntityClusterRecord] = []
        seen: set[int] = set()
        for mention in self.mentions_from_token(token_i):
            if mention.cluster_id in seen:
                continue
            cluster = self.clusters.get(mention.cluster_id)
            if cluster is None:
                continue
            clusters.append(cluster)
            seen.add(cluster.cluster_id)
        return tuple(clusters)

    def mentions_from_head_token(self, token_i: int) -> tuple[EntityMentionRecord, ...]:
        return tuple(
            self.mentions[mention_id]
            for mention_id in self.head_token_to_mention_ids.get(int(token_i), ())
            if mention_id in self.mentions
        )

    def mentions_from_span(self, start: int, end: int) -> tuple[EntityMentionRecord, ...]:
        return tuple(
            self.mentions[mention_id]
            for mention_id in self.span_to_mention_ids.get((int(start), int(end)), ())
            if mention_id in self.mentions
        )

    def span_for_mention(self, doc: Doc, mention_id: MentionId | int) -> Span:
        mention = self.mention(mention_id)
        return doc[mention.start : mention.end]

    def spans_for_cluster(self, doc: Doc, cluster_id: ClusterId | int) -> list[Span]:
        return [self.span_for_mention(doc, mention_id) for mention_id in self.cluster(cluster_id).mention_ids]

    def class_iri(self, cluster_id: ClusterId | int) -> str | None:
        typing = self.cluster(cluster_id).typing
        return typing.class_iri if typing is not None else None

    def attach_cluster_typing_profiles(
        self,
        profiles: dict[ClusterId, ClusterTypingProfile],
        *,
        overwrite: bool = False,
    ) -> None:
        for cluster_id, profile in profiles.items():
            cluster = self.cluster(cluster_id)
            if cluster.typing is not None and not overwrite:
                raise ValueError(f"Cluster {cluster_id} already has cluster typing.")
            cluster.typing = profile

    def attach_cluster_ocean_profiles(
        self,
        profiles: dict[ClusterId, ClusterOceanProfile],
        *,
        overwrite: bool = False,
    ) -> None:
        for cluster_id, profile in profiles.items():
            cluster = self.cluster(cluster_id)
            if cluster.ocean is not None and not overwrite:
                raise ValueError(f"Cluster {cluster_id} already has OCEAN profile.")
            cluster.ocean = profile

    def rebuild_indexes(self) -> None:
        token_index: dict[int, list[int]] = {}
        head_index: dict[int, list[int]] = {}
        span_index: dict[tuple[int, int], list[int]] = {}

        for mention_id, mention in self.mentions.items():
            for token_i in range(int(mention.start), int(mention.end)):
                token_index.setdefault(token_i, []).append(mention_id)

            if mention.head_token_i is not None:
                head_index.setdefault(int(mention.head_token_i), []).append(mention_id)

            span_index.setdefault((int(mention.start), int(mention.end)), []).append(mention_id)

        self.token_to_mention_ids = {key: tuple(value) for key, value in token_index.items()}
        self.head_token_to_mention_ids = {key: tuple(value) for key, value in head_index.items()}
        self.span_to_mention_ids = {key: tuple(value) for key, value in span_index.items()}
        self.cluster_name_index = {
            normalize_cluster_name(cluster.canonical_name): cluster_id
            for cluster_id, cluster in self.clusters.items()
        }

    def summary(self) -> dict[str, int]:
        return {
            "n_mentions": len(self.mentions),
            "n_clusters": len(self.clusters),
            "n_typed_clusters": sum(1 for cluster in self.clusters.values() if cluster.typing is not None),
            "n_ocean_clusters": sum(1 for cluster in self.clusters.values() if cluster.ocean is not None),
            "n_indexed_tokens": len(self.token_to_mention_ids),
            "n_indexed_heads": len(self.head_token_to_mention_ids),
            "n_indexed_spans": len(self.span_to_mention_ids),
        }


def normalize_cluster_name(value: str) -> str:
    return " ".join(str(value).casefold().strip().split())
