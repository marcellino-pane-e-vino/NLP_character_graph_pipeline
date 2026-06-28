"""Public ontology API for the NLP character graph pipeline.

Notebook and orchestration code should import from this module.  Low-level
ontology implementation remains split across ``io``, ``tbox``, ``relations``,
and ``abox``.
"""

from __future__ import annotations

from ontology.narrative_ontology import NarrativeOntology
from ontology.relations import (
    RelationPropertySpec,
    RelationRouter,
)
from ontology.tbox import (
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
