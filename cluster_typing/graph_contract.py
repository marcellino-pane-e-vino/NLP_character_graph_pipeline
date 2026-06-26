from __future__ import annotations

from typing import Any

import networkx as nx


VIRTUAL_ROOT = "__VIRTUAL_ROOT__"
STAY = "__STAY__"


class OntologyGraphContractError(ValueError):
    """Raised when a graph does not satisfy the cluster-typing DAG contract."""


class AmbiguousOntologyLabelError(OntologyGraphContractError):
    """Raised when a case-insensitive label lookup matches multiple nodes."""


def normalize_label(label: Any) -> str:
    """Normalize an ontology class label for case-insensitive exact matching."""

    return " ".join(str(label).strip().split()).casefold()


def validate_ontology_graph(graph: nx.DiGraph) -> None:
    """Validate the minimal ontology class DAG contract.

    The graph must use ontology class IRI strings as node IDs. Edges are
    parent -> child subclass edges. Labels are required only as derived display
    and notebook-query metadata; they are not the canonical class identity.
    """

    if not isinstance(graph, nx.DiGraph):
        raise OntologyGraphContractError(
            f"Expected networkx.DiGraph, got {type(graph)!r}."
        )

    if graph.number_of_nodes() == 0:
        raise OntologyGraphContractError("Ontology graph is empty.")

    non_string_nodes = [node_id for node_id in graph.nodes if not isinstance(node_id, str)]
    if non_string_nodes:
        raise OntologyGraphContractError(
            "Ontology graph node IDs must be class IRI strings because selected-path "
            f"JSONL artifacts persist class IRIs. Non-string nodes: {non_string_nodes[:20]}"
        )

    if not nx.is_directed_acyclic_graph(graph):
        raise OntologyGraphContractError("Ontology graph must be a DAG.")

    missing_labels = [
        class_iri
        for class_iri, attrs in graph.nodes(data=True)
        if not str(attrs.get("label", "")).strip()
    ]
    if missing_labels:
        raise OntologyGraphContractError(
            "Every ontology class node must have a non-empty 'label' attribute. "
            f"Missing for class IRIs: {missing_labels[:20]}"
        )

    labels_by_key: dict[str, list[str]] = {}
    for class_iri, attrs in graph.nodes(data=True):
        key = normalize_label(attrs["label"])
        labels_by_key.setdefault(key, []).append(str(class_iri))

    duplicates = {
        key: class_iris
        for key, class_iris in labels_by_key.items()
        if len(class_iris) > 1
    }
    if duplicates:
        raise AmbiguousOntologyLabelError(
            "Ontology graph has duplicate labels after case-insensitive "
            f"normalization: {duplicates}"
        )


def resolve_class_label(graph: nx.DiGraph, class_label: str) -> str:
    """Resolve an ontology class by actual node ``label`` only.

    Matching is case-insensitive and whitespace-normalized. Class IRIs, local
    names, and human-readable labels are intentionally not searched here.
    """

    key = normalize_label(class_label)
    matches = [
        str(class_iri)
        for class_iri, attrs in graph.nodes(data=True)
        if normalize_label(attrs.get("label", "")) == key
    ]

    if not matches:
        raise KeyError(f"Unknown ontology class label: {class_label!r}")

    if len(matches) > 1:
        raise AmbiguousOntologyLabelError(
            f"Ambiguous ontology class label {class_label!r}; "
            f"matching class IRIs: {matches}"
        )

    return matches[0]


def class_label(graph: nx.DiGraph, class_iri: str) -> str:
    """Return the compact ontology label for ``class_iri``."""

    if class_iri not in graph:
        raise KeyError(f"Unknown ontology class IRI: {class_iri!r}")
    label = str(graph.nodes[class_iri].get("label", "")).strip()
    if not label:
        raise OntologyGraphContractError(
            f"Ontology class {class_iri!r} has no non-empty 'label'."
        )
    return label


def class_human_readable_label(graph: nx.DiGraph, class_iri: str) -> str:
    """Return the human-readable label for ``class_iri``, falling back to label."""

    if class_iri not in graph:
        raise KeyError(f"Unknown ontology class IRI: {class_iri!r}")
    value = str(graph.nodes[class_iri].get("human_readable_label", "")).strip()
    return value or class_label(graph, class_iri)


def ontology_roots(graph: nx.DiGraph) -> list[str]:
    """Return ontology root class IRIs sorted by class label."""

    validate_ontology_graph(graph)
    roots = [str(class_iri) for class_iri, indegree in graph.in_degree() if indegree == 0]
    return sorted(roots, key=lambda class_iri: class_label(graph, class_iri).casefold())


def ontology_children(graph: nx.DiGraph, class_iri: str) -> list[str]:
    """Return child class IRIs of ``class_iri`` sorted by class label."""

    if class_iri not in graph:
        raise KeyError(f"Unknown ontology class IRI: {class_iri!r}")
    return sorted(
        [str(child_iri) for child_iri in graph.successors(class_iri)],
        key=lambda child_iri: class_label(graph, child_iri).casefold(),
    )


def ontology_descendants(
    graph: nx.DiGraph,
    class_label_ref: str,
    *,
    include_self: bool = False,
) -> set[str]:
    """Return descendant class IRIs of the class whose label matches ``class_label_ref``."""

    class_iri = resolve_class_label(graph, class_label_ref)
    descendants = {str(descendant) for descendant in nx.descendants(graph, class_iri)}
    if include_self:
        descendants.add(class_iri)
    return descendants


def is_ontology_leaf(graph: nx.DiGraph, class_iri: str) -> bool:
    """Return True when ``class_iri`` has no outgoing child edges."""

    if class_iri not in graph:
        raise KeyError(f"Unknown ontology class IRI: {class_iri!r}")
    return graph.out_degree(class_iri) == 0


def class_prompt_label(graph: nx.DiGraph, class_iri: str) -> str:
    """Return the label text used in zero-shot hypotheses."""

    return class_human_readable_label(graph, class_iri)


def validate_selected_path_edge(
    graph: nx.DiGraph,
    *,
    parent_class_iri: str,
    child_class_iri: str | None,
    edge_kind: str,
) -> None:
    """Validate one selected-path edge from a schema-v2 JSONL evidence record."""

    if edge_kind == "root":
        roots = set(ontology_roots(graph))
        if parent_class_iri != VIRTUAL_ROOT:
            raise OntologyGraphContractError(
                f"Root edge must have parent_class_iri={VIRTUAL_ROOT!r}, "
                f"got {parent_class_iri!r}."
            )
        if child_class_iri not in roots:
            raise OntologyGraphContractError(
                f"Root edge child_class_iri={child_class_iri!r} is not an ontology root."
            )
        return

    if edge_kind == "stay":
        if child_class_iri is not None:
            raise OntologyGraphContractError(
                f"Stay edge must have child_class_iri=None, got {child_class_iri!r}."
            )
        if parent_class_iri not in graph:
            raise OntologyGraphContractError(
                f"Stay edge references unknown parent_class_iri={parent_class_iri!r}."
            )
        return

    if edge_kind == "child":
        if parent_class_iri not in graph:
            raise OntologyGraphContractError(
                f"Child edge references unknown parent_class_iri={parent_class_iri!r}."
            )
        if child_class_iri not in graph:
            raise OntologyGraphContractError(
                f"Child edge references unknown child_class_iri={child_class_iri!r}."
            )
        if not graph.has_edge(parent_class_iri, child_class_iri):
            raise OntologyGraphContractError(
                "Selected child edge is not present in ontology graph: "
                f"{parent_class_iri!r} -> {child_class_iri!r}."
            )
        return

    raise OntologyGraphContractError(f"Unknown selected edge kind: {edge_kind!r}.")
