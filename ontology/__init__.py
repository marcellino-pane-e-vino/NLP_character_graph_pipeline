"""Ontology package public exports."""

from ontology.ontology_management import (
    NarrativeOntology,
    RelationPropertySpec,
    RelationRouter,
    human_label,
    local_name,
)


__all__ = [
    "NarrativeOntology",
    "RelationPropertySpec",
    "RelationRouter",
    "human_label",
    "local_name",
]
