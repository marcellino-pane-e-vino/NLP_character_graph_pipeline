"""Ontology TBox loading, metadata, validation, and class-graph construction.

This module owns schema-level ontology operations:
- class/property IRI and label helpers,
- RDF annotation lookup for labels/descriptions,
- object-property description validation,
- named-class hierarchy export as a NetworkX DiGraph,
- one-call TBox loading.

The public class graph representation is a raw ``networkx.DiGraph`` with class
IRI strings as node ids and parent -> child subclass edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import networkx as nx
from rdflib import DCTERMS, Graph, Literal, Namespace, RDFS, SKOS, URIRef

from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from ontology.io import load_ontology


SCHEMA = Namespace("https://schema.org/")
DESCRIPTION_PREDICATES = (
    RDFS.comment,
    SKOS.definition,
    DCTERMS.description,
    SCHEMA.description,
)
LABEL_PREDICATES = (
    RDFS.label,
    SKOS.prefLabel,
)


__all__ = [
    "SCHEMA",
    "DESCRIPTION_PREDICATES",
    "LABEL_PREDICATES",
    "OntologyGraphContractError",
    "iri_text",
    "local_name",
    "human_label",
    "parse_rdf_graph",
    "first_literal",
    "require_object_property_descriptions",
    "validate_class_graph",
    "build_class_graph",
    "load_tbox",
]


class OntologyGraphContractError(ValueError):
    """Raised when the class hierarchy graph violates the ontology contract."""


def iri_text(x: Any) -> str:
    """Best-effort string IRI for OWLAPY-like objects."""

    return getattr(x, "str", None) or str(getattr(x, "iri", x))


def local_name(iri: str) -> str:
    """Return the compact local name of an IRI-like string."""

    text = str(iri)
    return text.rsplit("#", 1)[-1] if "#" in text else text.rstrip("/").rsplit("/", 1)[-1]


def human_label(label: str) -> str:
    """Convert compact/camel/snake labels into a readable title-like label."""

    label = re.sub(r"[_\-]+", " ", str(label))
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in label.split())


def parse_rdf_graph(path: str | Path) -> Graph:
    """Parse RDF annotations from the ontology file.

    OWLAPY remains the ontology backend. RDFLib is used here only for
    serialization-level annotation lookup, such as labels and descriptions.
    """

    path = Path(path)
    graph = Graph()

    if path.suffix.lower() == ".ttl":
        graph.parse(str(path), format="turtle")
        return graph

    try:
        graph.parse(str(path), format="xml")
    except Exception:
        graph.parse(str(path))

    return graph


def first_literal(graph: Graph, iri: str, predicates: tuple[URIRef, ...]) -> str | None:
    """Return the first non-empty literal for ``iri`` over ``predicates``."""

    for predicate in predicates:
        for obj in graph.objects(URIRef(iri), predicate):
            if isinstance(obj, Literal) and str(obj).strip():
                return str(obj).strip()
    return None


def require_object_property_descriptions(
    onto: SyncOntology,
    ontology_path: str | Path,
) -> None:
    """Fail if an OWL object property has no description annotation.

    Accepted descriptions: rdfs:comment, skos:definition, dcterms:description,
    schema:description.
    """

    graph = parse_rdf_graph(ontology_path)
    missing: list[str] = []

    for prop in onto.object_properties_in_signature():
        prop_iri = iri_text(prop)
        if not first_literal(graph, prop_iri, DESCRIPTION_PREDICATES):
            missing.append(prop_iri)

    if missing:
        joined = "\n".join(f"- {iri}" for iri in sorted(missing))
        raise ValueError(
            "Every owl:ObjectProperty must have a description annotation.\n"
            "Add rdfs:comment, skos:definition, dcterms:description, or "
            "schema:description for:\n"
            f"{joined}"
        )


def validate_class_graph(class_graph: nx.DiGraph) -> None:
    """Validate the raw ontology class graph contract.

    The graph uses class IRI strings as node ids. Edges are parent -> child
    subclass edges. Labels and descriptions are node metadata.
    """

    if not isinstance(class_graph, nx.DiGraph):
        raise OntologyGraphContractError(
            f"Expected networkx.DiGraph, got {type(class_graph)!r}."
        )

    if class_graph.number_of_nodes() == 0:
        raise OntologyGraphContractError("Ontology class graph is empty.")

    non_string_nodes = [
        node_id
        for node_id in class_graph.nodes
        if not isinstance(node_id, str)
    ]
    if non_string_nodes:
        raise OntologyGraphContractError(
            "Ontology class graph node ids must be class IRI strings. "
            f"Non-string nodes: {non_string_nodes[:20]}"
        )

    if not nx.is_directed_acyclic_graph(class_graph):
        raise OntologyGraphContractError("Ontology class graph must be a DAG.")

    missing_labels = [
        class_iri
        for class_iri, attrs in class_graph.nodes(data=True)
        if not str(attrs.get("label", "")).strip()
    ]
    if missing_labels:
        raise OntologyGraphContractError(
            "Every ontology class node must have a non-empty 'label' attribute. "
            f"Missing for class IRIs: {missing_labels[:20]}"
        )


def build_class_graph(
    onto: SyncOntology,
    ontology_path: str | Path | None = None,
) -> nx.DiGraph:
    """Export the named-class taxonomy as a NetworkX DiGraph.

    Node ids are class IRI strings. Edges are parent -> child subclass edges.
    """

    rdf_graph = parse_rdf_graph(ontology_path) if ontology_path is not None else None
    reasoner = SyncReasoner(ontology=onto, reasoner="Structural")

    classes = list(onto.classes_in_signature())
    class_iris = {iri_text(cls) for cls in classes}

    class_graph = nx.DiGraph()

    for cls in classes:
        iri = iri_text(cls)
        label = (
            first_literal(rdf_graph, iri, LABEL_PREDICATES)
            if rdf_graph is not None
            else None
        )
        description = (
            first_literal(rdf_graph, iri, DESCRIPTION_PREDICATES)
            if rdf_graph is not None
            else None
        )
        label = label or local_name(iri)

        class_graph.add_node(
            iri,
            iri=iri,
            label=label,
            human_readable_label=human_label(label),
            description=description or "",
            local_name=local_name(iri),
        )

    for parent in classes:
        parent_iri = iri_text(parent)

        for child in reasoner.sub_classes(parent, direct=True):
            child_iri = iri_text(child)

            if child_iri in class_iris and child_iri != parent_iri:
                class_graph.add_edge(parent_iri, child_iri)

    validate_class_graph(class_graph)
    return class_graph


def load_tbox(
    path: str | Path,
    *,
    require_property_descriptions: bool = True,
) -> tuple[SyncOntology, nx.DiGraph]:
    """Load ontology, validate TBox, and return ontology + raw class graph."""

    onto = load_ontology(path)
    if require_property_descriptions:
        require_object_property_descriptions(onto, path)
    return onto, build_class_graph(onto, path)
