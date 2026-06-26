from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "cluster_typing_schema.py requires spaCy. Install it with: pip install spacy"
    ) from exc

from cluster_typing.graph_contract import (
    class_human_readable_label,
    class_label,
    ontology_descendants,
    resolve_class_label,
    validate_ontology_graph,
)
from ontology.tbox import local_name


__all__ = [
    "ClusterTypingAnnotation",
    "ClusterTypingLayer",
    "register_spacy_cluster_typing_extension",
    "require_cluster_typing_layer",
]


@dataclass(frozen=True, slots=True)
class ClusterTypingAnnotation:
    """Final ontology class selected for one coreference cluster.

    The contract is intentionally minimal. ``class_iri`` is the only canonical
    class identity stored on the annotation. Labels, local names, and prompt
    labels are derived from the raw ontology class graph stored on the layer.
    """

    cluster_id: int
    class_iri: str


@dataclass(slots=True)
class ClusterTypingLayer:
    """Independent cluster typing layer stored at ``doc._.cluster_typing_layer``."""

    class_graph: nx.DiGraph
    clusters: dict[int, ClusterTypingAnnotation] = field(default_factory=dict)
    source_folder: str | None = None

    def __post_init__(self) -> None:
        validate_ontology_graph(self.class_graph)

    def cluster(self, cluster_id: int) -> ClusterTypingAnnotation:
        return self.clusters[int(cluster_id)]

    def maybe_cluster(self, cluster_id: int) -> ClusterTypingAnnotation | None:
        return self.clusters.get(int(cluster_id))

    def typed_cluster_ids(self) -> list[int]:
        return sorted(self.clusters)

    def class_iri(self, cluster_id: int) -> str:
        return self.cluster(cluster_id).class_iri

    def class_label(self, cluster_id: int) -> str:
        return class_label(self.class_graph, self.class_iri(cluster_id))

    def class_human_readable_label(self, cluster_id: int) -> str:
        return class_human_readable_label(self.class_graph, self.class_iri(cluster_id))

    def class_local_name(self, cluster_id: int) -> str:
        class_iri = self.class_iri(cluster_id)
        attrs = self.class_graph.nodes[class_iri]
        return str(attrs.get("local_name", "")).strip() or local_name(class_iri)

    def cluster_ids_exactly(self, ontology_class_label: str) -> list[int]:
        """Return clusters whose final class has exactly this ontology label.

        Lookup is case-insensitive but searches only the graph node ``label``
        attribute. The stored contract remains ``class_iri``.
        """

        class_iri = resolve_class_label(self.class_graph, ontology_class_label)
        return [
            cluster_id
            for cluster_id, annotation in self.clusters.items()
            if annotation.class_iri == class_iri
        ]

    def cluster_ids_under(self, ontology_class_label: str) -> list[int]:
        """Return clusters whose final class is under ``ontology_class_label``."""

        valid_class_iris = ontology_descendants(
            self.class_graph,
            ontology_class_label,
            include_self=True,
        )
        return [
            cluster_id
            for cluster_id, annotation in self.clusters.items()
            if annotation.class_iri in valid_class_iris
        ]

    def clusters_by_class_label(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for cluster_id, annotation in self.clusters.items():
            label = class_label(self.class_graph, annotation.class_iri)
            out.setdefault(label, []).append(cluster_id)
        return {
            label: sorted(cluster_ids)
            for label, cluster_ids in sorted(out.items())
        }

    def summary(self) -> dict[str, int]:
        return {
            "n_typed_clusters": len(self.clusters),
            "n_classes_used": len(
                {annotation.class_iri for annotation in self.clusters.values()}
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
