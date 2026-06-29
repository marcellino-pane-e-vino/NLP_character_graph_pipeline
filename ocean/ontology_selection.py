"""Ontology-driven entity selection for OCEAN scoring."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from ontology.data_properties import ocean_data_property_iris_for_class_iri
from ontology.tbox import class_iris_with_data_properties


__all__ = [
    "entity_cluster_ids_with_data_properties",
    "entity_cluster_ids_with_ocean_data_fields",
]


def entity_cluster_ids_with_data_properties(
    entities,
    class_graph: nx.DiGraph,
    required_property_iris: Iterable[str],
) -> list[int]:
    """Return typed entity-cluster IDs whose ontology class has all fields.

    The class graph is expected to expose inherited data-property metadata,
    as produced by ``ontology.tbox.build_class_graph``.
    """

    valid_class_iris = class_iris_with_data_properties(
        class_graph,
        required_property_iris,
        include_inherited=True,
    )

    return [
        int(cluster_id)
        for cluster_id, cluster in entities.clusters.items()
        if cluster.typing is not None
        and str(cluster.typing.class_iri) in valid_class_iris
    ]


def entity_cluster_ids_with_ocean_data_fields(
    entities,
    class_graph: nx.DiGraph,
    *,
    field_owner_class_iri: str,
) -> list[int]:
    """Return entity-cluster IDs whose class exposes the full OCEAN field set.

    ``field_owner_class_iri`` is used only to derive the expected property IRIs
    from the ontology namespace. For example, passing the IRI of
    ``AgentiveEntity`` expects properties such as
    ``<same namespace>hasOceanOpenness``.
    """

    return entity_cluster_ids_with_data_properties(
        entities,
        class_graph,
        ocean_data_property_iris_for_class_iri(field_owner_class_iri),
    )
