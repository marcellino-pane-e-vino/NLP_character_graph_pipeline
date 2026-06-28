"""Project-level façade over OWLAPY SyncOntology.

The façade owns only the wrapped ``SyncOntology``.  TBox paths, class graphs,
and relation routers are treated as construction arguments or derived views, not
as persistent state of this object.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from owlapy.owl_ontology import SyncOntology

from annotation_layer.annotation_layer import AnnotationLayer
from ontology.abox import populate_abox
from ontology.io import load_ontology, save_ontology
from ontology.relations import RelationRouter, build_relation_router
from ontology.tbox import build_class_graph, require_object_property_descriptions


__all__ = [
    "NarrativeOntology",
]


class NarrativeOntology:
    """Small project façade over an OWLAPY ``SyncOntology``."""

    def __init__(
        self,
        tbox_path: str | Path,
        *,
        require_property_descriptions: bool = True,
    ) -> None:
        self.onto: SyncOntology = self.load(
            tbox_path,
            require_property_descriptions=require_property_descriptions,
        )

    @staticmethod
    def load(
        tbox_path: str | Path,
        *,
        require_property_descriptions: bool = True,
    ) -> SyncOntology:
        """Load the TBox into a ``SyncOntology``."""

        onto = load_ontology(tbox_path)

        if require_property_descriptions:
            require_object_property_descriptions(
                onto,
                tbox_path,
            )

        return onto

    def to_networkx_class_graph(
        self,
        *,
        ontology_path: str | Path | None = None,
    ) -> nx.DiGraph:
        """Build and return the ontology class graph.

        The graph is a derived view.  It is not cached in this façade.
        """

        return build_class_graph(
            self.onto,
            ontology_path,
        )

    def build_relation_router(
        self,
        *,
        ontology_path: str | Path,
        class_graph: nx.DiGraph,
    ) -> RelationRouter:
        """Build and return the relation router.

        The router is a derived view consumed by relationship extraction.  It is
        not cached in this façade.
        """

        return build_relation_router(
            onto=self.onto,
            ontology_path=ontology_path,
            class_graph=class_graph,
        )

    def populate(
        self,
        annotation_layer: AnnotationLayer,
        *,
        abox_base_iri: str,
        print_result_summary: bool = False,
    ) -> None:
        """Populate the wrapped ontology with ABox axioms."""

        populate_abox(
            self.onto,
            annotation_layer,
            abox_base_iri=abox_base_iri,
            print_result_summary=print_result_summary,
        )

    def save(
        self,
        output_path: str | Path,
        *,
        document_format: str | None = "turtle",
    ) -> None:
        """Save the wrapped ontology to a new file."""

        save_ontology(
            self.onto,
            output_path,
            document_format=document_format,
        )
