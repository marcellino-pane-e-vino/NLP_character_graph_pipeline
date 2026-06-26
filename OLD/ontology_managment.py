"""Minimal OWLAPY ontology helpers for the NLP ontology-population pipeline.

Design:
- OWLAPY is the ontology backend.
- RDFLibReasoner, shipped by OWLAPY, is used to derive the class hierarchy.
- RDFLib is used only for rdfs:label / rdfs:comment lookup because those are
  RDF annotations and OWLAPY examples do not expose a simpler metadata API.
- NetworkX is only a derived view, not the ontology source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import networkx as nx
from rdflib import DCTERMS, Graph, Literal, Namespace, RDFS, SKOS, URIRef

from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLClassAssertionAxiom,
    OWLDataPropertyAssertionAxiom,
    OWLDeclarationAxiom,
    OWLObjectPropertyAssertionAxiom,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import SyncReasoner


SCHEMA = Namespace("https://schema.org/")
DESCRIPTION_PREDICATES = (RDFS.comment, SKOS.definition, DCTERMS.description, SCHEMA.description)
LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)


# ---------- tiny generic helpers ----------


def load_ontology(path: str | Path) -> SyncOntology:
    """Load an OWL/OWL2/Turtle ontology with OWLAPY."""

    return SyncOntology(str(Path(path)))


def save_ontology(onto: SyncOntology, path: str | Path, *, document_format: str | None = None) -> None:
    """Save the ontology using OWLAPY."""

    onto.save(path=str(Path(path)), document_format=document_format)


def iri_text(x: Any) -> str:
    """Best-effort string IRI for OWLAPY objects."""

    return getattr(x, "str", None) or str(getattr(x, "iri", x))


def local_name(iri: str) -> str:
    text = str(iri)
    return text.rsplit("#", 1)[-1] if "#" in text else text.rstrip("/").rsplit("/", 1)[-1]


def human_label(label: str) -> str:
    label = re.sub(r"[_\-]+", " ", str(label))
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in label.split())


def _rdf(path: str | Path) -> Graph:
    """Parse RDF annotations from the same ontology file.

    This is deliberately small. OWLAPY remains the ontology backend; RDFLib is
    only used to read labels/comments from the concrete serialization.
    """

    path = Path(path)
    g = Graph()
    if path.suffix.lower() == ".ttl":
        g.parse(str(path), format="turtle")
    else:
        try:
            g.parse(str(path), format="xml")
        except Exception:
            g.parse(str(path))
    return g


def _first_literal(g: Graph, iri: str, predicates: tuple[URIRef, ...]) -> str | None:
    for p in predicates:
        for obj in g.objects(URIRef(iri), p):
            if isinstance(obj, Literal) and str(obj).strip():
                return str(obj).strip()
    return None


# ---------- validation ----------


def require_object_property_descriptions(onto: SyncOntology, ontology_path: str | Path) -> None:
    """Fail if an OWL object property has no description annotation.

    Accepted descriptions: rdfs:comment, skos:definition, dcterms:description,
    schema:description.
    """

    g = _rdf(ontology_path)
    missing = []

    for prop in onto.object_properties_in_signature():
        prop_iri = iri_text(prop)
        if not _first_literal(g, prop_iri, DESCRIPTION_PREDICATES):
            missing.append(prop_iri)

    if missing:
        joined = "\n".join(f"- {iri}" for iri in sorted(missing))
        raise ValueError(
            "Every owl:ObjectProperty must have a description annotation.\n"
            "Add rdfs:comment, skos:definition, dcterms:description, or schema:description for:\n"
            f"{joined}"
        )


# ---------- class DAG export ----------


def class_dag(onto: SyncOntology, ontology_path: str | Path | None = None) -> nx.DiGraph:
    """Export the named-class taxonomy as a NetworkX DiGraph.

    Node ids are class IRI strings. Edges are parent -> child.
    """

    g = _rdf(ontology_path) if ontology_path is not None else None
    reasoner = SyncReasoner(ontology=onto, reasoner="Structural")

    classes = list(onto.classes_in_signature())
    class_iris = {iri_text(c) for c in classes}

    graph = nx.DiGraph()

    for cls in classes:
        iri = iri_text(cls)
        label = _first_literal(g, iri, LABEL_PREDICATES) if g is not None else None
        desc = _first_literal(g, iri, DESCRIPTION_PREDICATES) if g is not None else None
        label = label or local_name(iri)

        graph.add_node(
            iri,
            label=label,
            human_readable_label=human_label(label),
            description=desc or "",
            local_name=local_name(iri),
        )

    for parent in classes:
        parent_iri = iri_text(parent)

        for child in reasoner.sub_classes(parent, direct=True):
            child_iri = iri_text(child)

            if child_iri in class_iris and child_iri != parent_iri:
                graph.add_edge(parent_iri, child_iri)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"The class hierarchy is not a DAG: {nx.find_cycle(graph)}")

    return graph


# ---------- ABox population ----------


def cluster_individual(individual_ns: str, cluster_id: int | str) -> OWLNamedIndividual:
    """Create the OWL individual representing a coreference cluster."""

    if str(cluster_id).startswith(("http://", "https://", "urn:")):
        return OWLNamedIndividual(IRI.create(str(cluster_id)))

    sep = "" if individual_ns.endswith(("#", "/")) else "#"
    return OWLNamedIndividual(IRI.create(f"{individual_ns}{sep}cluster_{cluster_id}"))


def assert_cluster_type(
    onto: SyncOntology,
    *,
    individual_ns: str,
    cluster_id: int | str,
    class_iri: str,
) -> None:
    """Add: cluster_N rdf:type Class."""

    ind = cluster_individual(individual_ns, cluster_id)
    cls = OWLClass(IRI.create(class_iri))
    onto.add_axiom([OWLDeclarationAxiom(ind), OWLClassAssertionAxiom(ind, cls)])


def assert_cluster_relation(
    onto: SyncOntology,
    *,
    individual_ns: str,
    source_cluster_id: int | str,
    property_iri: str,
    target_cluster_id: int | str,
) -> None:
    """Add: cluster_A objectProperty cluster_B."""

    src = cluster_individual(individual_ns, source_cluster_id)
    dst = cluster_individual(individual_ns, target_cluster_id)
    prop = OWLObjectProperty(IRI.create(property_iri))
    onto.add_axiom([OWLDeclarationAxiom(src), OWLDeclarationAxiom(dst), OWLObjectPropertyAssertionAxiom(src, prop, dst)])


def assert_cluster_value(
    onto: SyncOntology,
    *,
    individual_ns: str,
    cluster_id: int | str,
    property_iri: str,
    value: Any,
) -> None:
    """Add: cluster_N dataProperty literal_value."""

    ind = cluster_individual(individual_ns, cluster_id)
    prop = OWLDataProperty(IRI.create(property_iri))
    onto.add_axiom([OWLDeclarationAxiom(ind), OWLDataPropertyAssertionAxiom(ind, prop, OWLLiteral(value))])


# ---------- one-call convenience ----------


def load_tbox(
    path: str | Path,
    *,
    require_property_descriptions: bool = True,
) -> tuple[SyncOntology, nx.DiGraph]:
    """Load ontology, validate object-property descriptions, return ontology + class DAG."""

    onto = load_ontology(path)
    if require_property_descriptions:
        require_object_property_descriptions(onto, path)
    return onto, class_dag(onto, path)
