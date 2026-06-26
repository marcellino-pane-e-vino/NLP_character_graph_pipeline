"""ABox population helpers.

This module owns ontology mutation for concrete pipeline outputs: coreference
cluster individuals, class assertions, relation assertions, and literal values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


__all__ = [
    "cluster_individual",
    "assert_cluster_type",
    "assert_cluster_relation",
    "assert_cluster_value",
]


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
    """Add ``cluster_N rdf:type Class``."""

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
    """Add ``cluster_A objectProperty cluster_B``."""

    src = cluster_individual(individual_ns, source_cluster_id)
    dst = cluster_individual(individual_ns, target_cluster_id)
    prop = OWLObjectProperty(IRI.create(property_iri))
    onto.add_axiom(
        [
            OWLDeclarationAxiom(src),
            OWLDeclarationAxiom(dst),
            OWLObjectPropertyAssertionAxiom(src, prop, dst),
        ]
    )


def assert_cluster_value(
    onto: SyncOntology,
    *,
    individual_ns: str,
    cluster_id: int | str,
    property_iri: str,
    value: Any,
) -> None:
    """Add ``cluster_N dataProperty literal_value``."""

    ind = cluster_individual(individual_ns, cluster_id)
    prop = OWLDataProperty(IRI.create(property_iri))
    onto.add_axiom(
        [
            OWLDeclarationAxiom(ind),
            OWLDataPropertyAssertionAxiom(ind, prop, OWLLiteral(value)),
        ]
    )
