"""External ontology graph builder.

This module is intentionally outside the ``cluster_typing`` package API.
It is only one possible way to create the ``networkx.DiGraph`` consumed by
``cluster_typing``.

The cluster-typing subsystem should not depend on this module. In future, the
graph may come from another source: JSON, a database, manual construction, or
another ontology tool.

Graph contract produced here:
    - type: networkx.DiGraph
    - node IDs: string IRIs
    - edges: parent -> child
    - required node attribute: label
    - optional node attributes: human_readable_label, description, local_name
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import re

import networkx as nx
from rdflib import Graph, Literal, OWL, RDF, RDFS, SKOS, URIRef


OWL_THING = str(OWL.Thing)
OWL_NOTHING = str(OWL.Nothing)


class OntologyGraphBuilderError(ValueError):
    """Raised when an ontology graph cannot be built."""


class OntologyGraphCycleError(OntologyGraphBuilderError):
    """Raised when subclass relations do not form a DAG."""


def iri_to_local_name(iri: str) -> str:
    text = str(iri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def camel_to_words(name: str) -> str:
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def label_to_human_readable_label(label: str) -> str:
    words = camel_to_words(label)
    if not words:
        return label
    return " ".join(word if word.isupper() else word.capitalize() for word in words.split())


def _literal_lang(value: Literal) -> Optional[str]:
    return value.language.lower() if value.language else None


def _first_literal(
    graph: Graph,
    subject: URIRef,
    predicates: Iterable[URIRef],
    *,
    preferred_languages: Sequence[Optional[str]],
) -> Optional[str]:
    literals: list[Literal] = []
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            if isinstance(obj, Literal) and str(obj).strip():
                literals.append(obj)

    if not literals:
        return None

    by_lang: dict[Optional[str], list[Literal]] = {}
    for literal in literals:
        by_lang.setdefault(_literal_lang(literal), []).append(literal)

    for lang in preferred_languages:
        if lang in by_lang:
            return str(by_lang[lang][0])

    return str(literals[0])


def _collect_named_classes(graph: Graph) -> set[str]:
    classes: set[str] = set()

    for cls in graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef):
            classes.add(str(cls))

    for cls in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(cls, URIRef):
            classes.add(str(cls))

    # Lightweight ontologies may omit explicit class declarations.
    for child, _predicate, parent in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(child, URIRef):
            classes.add(str(child))
        if isinstance(parent, URIRef):
            classes.add(str(parent))

    classes.discard(OWL_THING)
    classes.discard(OWL_NOTHING)
    return classes


def _collect_subclass_edges(graph: Graph, class_iris: set[str]) -> set[tuple[str, str]]:
    """Return edges as (parent, child)."""

    edges: set[tuple[str, str]] = set()

    for child, _predicate, parent in graph.triples((None, RDFS.subClassOf, None)):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            # Anonymous OWL expressions are intentionally ignored as navigation nodes.
            continue

        child_iri = str(child)
        parent_iri = str(parent)

        if child_iri == parent_iri:
            continue

        if child_iri in class_iris and parent_iri in class_iris:
            edges.add((parent_iri, child_iri))

    return edges


def _assert_unique_labels_case_insensitive(graph: nx.DiGraph) -> None:
    labels: dict[str, list[str]] = {}
    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", "")).strip()
        if not label:
            raise OntologyGraphBuilderError(f"Node {node_id!r} has no label.")
        key = " ".join(label.split()).casefold()
        labels.setdefault(key, []).append(str(node_id))

    duplicates = {key: ids for key, ids in labels.items() if len(ids) > 1}
    if duplicates:
        raise OntologyGraphBuilderError(
            "Ontology labels must be unique after case-insensitive normalization "
            f"because cluster_typing resolves classes by label only: {duplicates}"
        )


def build_networkx_graph_from_ttl(
    ttl_path: str | Path,
    *,
    preferred_languages: Sequence[Optional[str]] = ("en", None),
) -> nx.DiGraph:
    """Build a ``networkx.DiGraph`` ontology class DAG from a Turtle file.

    This builder is external to ``cluster_typing``. It is a convenience adapter,
    not part of the cluster-typing subsystem.
    """

    path = Path(ttl_path)
    if path.suffix.lower() != ".ttl":
        raise OntologyGraphBuilderError(f"Expected a .ttl file, got: {path}")
    if not path.exists():
        raise OntologyGraphBuilderError(f"TTL file does not exist: {path}")
    if not path.is_file():
        raise OntologyGraphBuilderError(f"TTL path is not a file: {path}")

    rdf_graph = Graph()
    rdf_graph.parse(str(path), format="turtle")

    class_iris = _collect_named_classes(rdf_graph)
    edges = _collect_subclass_edges(rdf_graph, class_iris)

    graph = nx.DiGraph()
    graph.graph["source_path"] = str(path)
    graph.graph["builder"] = "ontology_graph_builder.build_networkx_graph_from_ttl"

    preferred_languages = tuple(
        lang.lower() if isinstance(lang, str) else None
        for lang in preferred_languages
    )

    for iri in sorted(class_iris):
        uri = URIRef(iri)
        local_name = iri_to_local_name(iri)
        label = (
            _first_literal(
                rdf_graph,
                uri,
                [RDFS.label, SKOS.prefLabel],
                preferred_languages=preferred_languages,
            )
            or local_name
        )
        description = (
            _first_literal(
                rdf_graph,
                uri,
                [RDFS.comment, SKOS.definition],
                preferred_languages=preferred_languages,
            )
            or ""
        )

        graph.add_node(
            str(iri),
            local_name=local_name,
            label=str(label),
            human_readable_label=label_to_human_readable_label(str(label)),
            description=str(description),
        )

    graph.add_edges_from(sorted(edges))

    if not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph)
        except Exception:
            cycle = []
        raise OntologyGraphCycleError(
            f"Subclass relations contain a cycle, so no DAG can be built: {cycle}"
        )

    _assert_unique_labels_case_insensitive(graph)

    return graph


def ontology_label_tree(graph: nx.DiGraph) -> str:
    """Return a readable tree-like view. DAG nodes may appear more than once."""

    roots = sorted(
        [node_id for node_id, indegree in graph.in_degree() if indegree == 0],
        key=lambda node_id: str(graph.nodes[node_id].get("human_readable_label", graph.nodes[node_id].get("label", node_id))),
    )

    lines: list[str] = []

    def rec(node_id: str, level: int) -> None:
        attrs = graph.nodes[node_id]
        label = attrs.get("human_readable_label") or attrs.get("label") or node_id
        lines.append("  " * level + f"- {label}")
        children = sorted(
            graph.successors(node_id),
            key=lambda child_id: str(graph.nodes[child_id].get("human_readable_label", graph.nodes[child_id].get("label", child_id))),
        )
        for child_id in children:
            rec(child_id, level + 1)

    for root in roots:
        rec(root, 0)

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a .ttl ontology file and print a NetworkX-backed label tree."
    )
    parser.add_argument("ttl", help="Path to a Turtle ontology file, e.g. ontology.ttl")
    args = parser.parse_args()

    graph = build_networkx_graph_from_ttl(args.ttl)
    print(f"Loaded classes: {graph.number_of_nodes()}")
    print(f"Loaded subclass edges: {graph.number_of_edges()}")
    print("\nHierarchy:")
    print(ontology_label_tree(graph))
