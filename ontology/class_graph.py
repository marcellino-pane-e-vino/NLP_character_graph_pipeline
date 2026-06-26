"""Ontology class graph wrapper.

This module provides the canonical class-identity boundary for the pipeline.
Machine-facing class identity is always a class IRI string. Labels, local names,
human-readable labels, prompt labels, and hierarchy traversal are derived views
computed from the underlying ontology class DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import networkx as nx

from cluster_typing.graph_contract import (
    resolve_class_label,
    validate_ontology_graph,
)


def local_name(iri: str) -> str:
    text = str(iri)
    return text.rsplit("#", 1)[-1] if "#" in text else text.rstrip("/").rsplit("/", 1)[-1]


def human_label(label: str) -> str:
    label = re.sub(r"[_\-]+", " ", str(label))
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in label.split())


__all__ = [
    "OntologyClassRef",
    "OntologyClassGraph",
]


@dataclass(frozen=True, slots=True)
class OntologyClassRef:
    """Derived display/reference view for one ontology class."""

    iri: str
    label: str
    human_readable_label: str
    local_name: str
    description: str = ""


class OntologyClassGraph:
    """Canonical wrapper around the ontology class DAG.

    The wrapped ``networkx.DiGraph`` must use class IRI strings as node IDs and
    parent -> child edges. Public pipeline contracts should pass class IRI
    strings; all other forms are derived through this wrapper.
    """

    def __init__(self, graph: nx.DiGraph):
        validate_ontology_graph(graph)
        self._graph = graph

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying NetworkX DAG for lower-level integrations."""

        return self._graph

    def require_class_iri(self, class_iri: str) -> str:
        """Return ``class_iri`` if it is a known class, otherwise fail."""

        value = str(class_iri).strip()
        if not value:
            raise ValueError("class_iri cannot be empty.")
        if value not in self._graph:
            raise KeyError(f"Unknown ontology class IRI: {value!r}")
        return value

    def resolve_label(self, class_label: str) -> str:
        """Resolve an ontology node label to its canonical class IRI."""

        return str(resolve_class_label(self._graph, class_label))

    def ref(self, class_iri: str) -> OntologyClassRef:
        """Return all derived display metadata for ``class_iri``."""

        iri = self.require_class_iri(class_iri)
        attrs = self._graph.nodes[iri]

        label = str(attrs.get("label", "")).strip()
        if not label:
            raise ValueError(f"Ontology class {iri!r} has no non-empty label.")

        readable = str(attrs.get("human_readable_label", "")).strip() or human_label(label)
        short_name = str(attrs.get("local_name", "")).strip() or local_name(iri)
        description = str(attrs.get("description", ""))

        return OntologyClassRef(
            iri=iri,
            label=label,
            human_readable_label=readable,
            local_name=short_name,
            description=description,
        )

    def label(self, class_iri: str) -> str:
        return self.ref(class_iri).label

    def human_readable_label(self, class_iri: str) -> str:
        return self.ref(class_iri).human_readable_label

    def local_name(self, class_iri: str) -> str:
        return self.ref(class_iri).local_name

    def description(self, class_iri: str) -> str:
        return self.ref(class_iri).description

    def prompt_label(self, class_iri: str) -> str:
        """Return the label text to use in model-facing prompts."""

        return self.human_readable_label(class_iri)

    def roots(self) -> tuple[str, ...]:
        roots = [
            str(class_iri)
            for class_iri, indegree in self._graph.in_degree()
            if indegree == 0
        ]
        return tuple(sorted(roots, key=lambda iri: self.label(iri).casefold()))

    def children(self, class_iri: str) -> tuple[str, ...]:
        iri = self.require_class_iri(class_iri)
        children = [str(child_iri) for child_iri in self._graph.successors(iri)]
        return tuple(sorted(children, key=lambda child_iri: self.label(child_iri).casefold()))

    def descendants(self, class_iri: str, *, include_self: bool = False) -> frozenset[str]:
        iri = self.require_class_iri(class_iri)
        descendants = {str(descendant) for descendant in nx.descendants(self._graph, iri)}
        if include_self:
            descendants.add(iri)
        return frozenset(descendants)

    def is_leaf(self, class_iri: str) -> bool:
        iri = self.require_class_iri(class_iri)
        return self._graph.out_degree(iri) == 0
