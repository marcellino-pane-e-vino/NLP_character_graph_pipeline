from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "cluster_typing_schema.py requires spaCy. Install it with: pip install spacy"
    ) from exc

from .graph_contract import (
    ontology_descendants,
    resolve_class_label,
    validate_ontology_graph,
)


__all__ = [
    "ClusterTypingAnnotation",
    "ClusterTypingLayer",
    "register_spacy_cluster_typing_extension",
    "require_cluster_typing_layer",
]


@dataclass(frozen=True)
class ClusterTypingAnnotation:
    """Final cluster type annotation for one coreference cluster.

    The final annotation is intentionally slim: no confidence, no path, no edge
    scores, and no mention-level evidence.
    """

    cluster_id: int
    class_id: Any
    class_label: str
    class_human_readable_label: str


@dataclass
class ClusterTypingLayer:
    """Independent cluster typing layer stored at ``doc._.cluster_typing_layer``."""

    graph: nx.DiGraph
    clusters: dict[int, ClusterTypingAnnotation] = field(default_factory=dict)
    source_folder: str | None = None

    def __post_init__(self) -> None:
        validate_ontology_graph(self.graph)

    def cluster(self, cluster_id: int) -> ClusterTypingAnnotation:
        return self.clusters[int(cluster_id)]

    def maybe_cluster(self, cluster_id: int) -> ClusterTypingAnnotation | None:
        return self.clusters.get(int(cluster_id))

    def typed_cluster_ids(self) -> list[int]:
        return sorted(self.clusters)

    def class_id(self, cluster_id: int) -> Any:
        return self.cluster(cluster_id).class_id

    def class_label(self, cluster_id: int) -> str:
        return self.cluster(cluster_id).class_label

    def class_human_readable_label(self, cluster_id: int) -> str:
        return self.cluster(cluster_id).class_human_readable_label

    def cluster_ids_exactly(self, class_label: str) -> list[int]:
        """Return clusters whose final class has exactly this ontology label.

        Lookup is case-insensitive but searches only the graph node ``label``
        attribute, not node IDs or human-readable labels.
        """

        class_id = resolve_class_label(self.graph, class_label)
        return [
            cluster_id
            for cluster_id, annotation in self.clusters.items()
            if annotation.class_id == class_id
        ]

    def cluster_ids_under(self, class_label: str) -> list[int]:
        """Return clusters whose final class is under ``class_label``.

        The lookup of ``class_label`` is label-only and case-insensitive.
        """

        valid_class_ids = ontology_descendants(
            self.graph,
            class_label,
            include_self=True,
        )
        return [
            cluster_id
            for cluster_id, annotation in self.clusters.items()
            if annotation.class_id in valid_class_ids
        ]

    def clusters_by_class_label(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for cluster_id, annotation in self.clusters.items():
            out.setdefault(annotation.class_label, []).append(cluster_id)
        return {
            label: sorted(cluster_ids)
            for label, cluster_ids in sorted(out.items())
        }

    def summary(self) -> dict[str, int]:
        return {
            "n_typed_clusters": len(self.clusters),
            "n_classes_used": len(
                {annotation.class_id for annotation in self.clusters.values()}
            ),
        }


def register_spacy_cluster_typing_extension(*, force: bool = False) -> None:
    """Register ``doc._.cluster_typing_layer`` as the cluster typing extension."""

    if Doc.has_extension("cluster_typing_layer"):
        if force:
            Doc.set_extension("cluster_typing_layer", default=None, force=True)
        return

    Doc.set_extension("cluster_typing_layer", default=None)


def require_cluster_typing_layer(doc: Doc) -> ClusterTypingLayer:
    """Return ``doc._.cluster_typing_layer`` or fail early when absent."""

    if not Doc.has_extension("cluster_typing_layer") or doc._.cluster_typing_layer is None:
        raise ValueError("This Doc has no cluster typing layer.")

    return doc._.cluster_typing_layer
