"""Ontology file I/O.

This module is the only place where ontology persistence is handled.  It wraps
OWLAPY loading/saving so the rest of the pipeline does not construct or
serialize ``SyncOntology`` objects directly.
"""

from __future__ import annotations

from pathlib import Path

from owlapy.owl_ontology import SyncOntology


__all__ = [
    "load_ontology",
    "save_ontology",
]


def load_ontology(path: str | Path) -> SyncOntology:
    """Load an OWL/OWL2/Turtle ontology with OWLAPY."""

    return SyncOntology(str(Path(path)))


def save_ontology(
    onto: SyncOntology,
    path: str | Path,
    *,
    document_format: str | None = None,
) -> None:
    """Save an ontology through OWLAPY."""

    onto.save(path=str(Path(path)), document_format=document_format)
