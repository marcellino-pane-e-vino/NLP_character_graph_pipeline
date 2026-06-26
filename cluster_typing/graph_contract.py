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
    """Validate the minimal ontology graph contract.

    The function does not care how the graph was built. It only enforces the
    contract required by cluster typing.
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
            "Ontology graph node IDs must be strings because selected-path "
            f"JSONL artifacts persist node IDs. Non-string nodes: {non_string_nodes[:20]}"
        )

    if not nx.is_directed_acyclic_graph(graph):
        raise OntologyGraphContractError("Ontology graph must be a DAG.")

    missing_labels = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if not str(attrs.get("label", "")).strip()
    ]
    if missing_labels:
        raise OntologyGraphContractError(
            "Every ontology node must have a non-empty 'label' attribute. "
            f"Missing for nodes: {missing_labels[:20]}"
        )

    # Because public resolution is label-only, labels must be unique after
    # normalization.
    labels_by_key: dict[str, list[Any]] = {}
    for node_id, attrs in graph.nodes(data=True):
        key = normalize_label(attrs["label"])
        labels_by_key.setdefault(key, []).append(node_id)

    duplicates = {
        key: node_ids
        for key, node_ids in labels_by_key.items()
        if len(node_ids) > 1
    }
    if duplicates:
        raise AmbiguousOntologyLabelError(
            "Ontology graph has duplicate labels after case-insensitive "
            f"normalization: {duplicates}"
        )


def resolve_class_label(graph: nx.DiGraph, class_label: str) -> Any:
    """Resolve an ontology class by actual node ``label`` only.

    Matching is case-insensitive and whitespace-normalized. Node IDs are not
    searched. ``human_readable_label`` is not searched.
    """

    key = normalize_label(class_label)
    matches = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if normalize_label(attrs.get("label", "")) == key
    ]

    if not matches:
        raise KeyError(f"Unknown ontology class label: {class_label!r}")

    if len(matches) > 1:
        raise AmbiguousOntologyLabelError(
            f"Ambiguous ontology class label {class_label!r}; "
            f"matching node IDs: {matches}"
        )

    return matches[0]


def class_label(graph: nx.DiGraph, class_id: Any) -> str:
    """Return the compact ontology label for a graph node."""

    if class_id not in graph:
        raise KeyError(f"Unknown ontology class node: {class_id!r}")
    label = str(graph.nodes[class_id].get("label", "")).strip()
    if not label:
        raise OntologyGraphContractError(
            f"Ontology node {class_id!r} has no non-empty 'label'."
        )
    return label


def class_human_readable_label(graph: nx.DiGraph, class_id: Any) -> str:
    """Return the human-readable label, falling back to ``label``."""

    if class_id not in graph:
        raise KeyError(f"Unknown ontology class node: {class_id!r}")
    value = str(graph.nodes[class_id].get("human_readable_label", "")).strip()
    return value or class_label(graph, class_id)


def ontology_roots(graph: nx.DiGraph) -> list[Any]:
    """Return ontology root nodes sorted by class label."""

    validate_ontology_graph(graph)
    roots = [node_id for node_id, indegree in graph.in_degree() if indegree == 0]
    return sorted(roots, key=lambda node_id: class_label(graph, node_id).casefold())


def ontology_children(graph: nx.DiGraph, class_id: Any) -> list[Any]:
    """Return child classes of ``class_id`` sorted by class label."""

    if class_id not in graph:
        raise KeyError(f"Unknown ontology class node: {class_id!r}")
    return sorted(
        list(graph.successors(class_id)),
        key=lambda node_id: class_label(graph, node_id).casefold(),
    )


def ontology_descendants(
    graph: nx.DiGraph,
    class_label_ref: str,
    *,
    include_self: bool = False,
) -> set[Any]:
    """Return descendants of the class whose actual label matches ``class_label_ref``."""

    class_id = resolve_class_label(graph, class_label_ref)
    descendants = set(nx.descendants(graph, class_id))
    if include_self:
        descendants.add(class_id)
    return descendants


def is_ontology_leaf(graph: nx.DiGraph, class_id: Any) -> bool:
    """Return True when ``class_id`` has no outgoing child edges."""

    if class_id not in graph:
        raise KeyError(f"Unknown ontology class node: {class_id!r}")
    return graph.out_degree(class_id) == 0


def class_prompt_label(graph: nx.DiGraph, class_id: Any) -> str:
    """Return the label text used in zero-shot hypotheses.

    The model-facing label is human-readable when available; otherwise it falls
    back to the compact ontology label.
    """

    return class_human_readable_label(graph, class_id)


def validate_selected_path_edge(
    graph: nx.DiGraph,
    *,
    parent_id: Any,
    child_id: Any | None,
    edge_kind: str,
) -> None:
    """Validate one selected-path edge from a JSONL evidence record."""

    if edge_kind == "root":
        roots = set(ontology_roots(graph))
        if parent_id != VIRTUAL_ROOT:
            raise OntologyGraphContractError(
                f"Root edge must have parent_id={VIRTUAL_ROOT!r}, got {parent_id!r}."
            )
        if child_id not in roots:
            raise OntologyGraphContractError(
                f"Root edge child_id={child_id!r} is not an ontology root."
            )
        return

    if edge_kind == "stay":
        if child_id is not None:
            raise OntologyGraphContractError(
                f"Stay edge must have child_id=None, got {child_id!r}."
            )
        if parent_id not in graph:
            raise OntologyGraphContractError(
                f"Stay edge references unknown parent_id={parent_id!r}."
            )
        return

    if edge_kind == "child":
        if parent_id not in graph:
            raise OntologyGraphContractError(
                f"Child edge references unknown parent_id={parent_id!r}."
            )
        if child_id not in graph:
            raise OntologyGraphContractError(
                f"Child edge references unknown child_id={child_id!r}."
            )
        if not graph.has_edge(parent_id, child_id):
            raise OntologyGraphContractError(
                f"Selected child edge is not present in ontology graph: "
                f"{parent_id!r} -> {child_id!r}."
            )
        return

    raise OntologyGraphContractError(f"Unknown selected edge kind: {edge_kind!r}.")
