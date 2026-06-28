from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from graphviz import Digraph
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

try:
    from rdflib.namespace import SKOS
except ImportError:  # pragma: no cover - compatibility guard for old RDFLib versions
    SKOS = None  # type: ignore[assignment]


SCHEMA_CLASSES = {
    OWL.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.Ontology,
    RDF.Property,
    RDFS.Class,
}

SCHEMA_PREDICATES = {
    RDF.type,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.domain,
    RDFS.range,
    RDFS.label,
    RDFS.comment,
    OWL.equivalentClass,
    OWL.disjointWith,
    OWL.inverseOf,
    OWL.imports,
    OWL.versionIRI,
    OWL.priorVersion,
    OWL.deprecated,
}

LABEL_PREDICATES = tuple(
    p
    for p in (
        RDFS.label,
        getattr(SKOS, "prefLabel", None) if SKOS is not None else None,
        getattr(SKOS, "altLabel", None) if SKOS is not None else None,
    )
    if p is not None
)

PALETTE = [
    "#E8F1FA",
    "#EAF6EA",
    "#FFF3D9",
    "#F7E9F3",
    "#EDE7F6",
    "#E0F7FA",
    "#FBE9E7",
    "#F1F8E9",
    "#ECEFF1",
    "#FFF8E1",
]

BORDER_PALETTE = [
    "#5B7C99",
    "#5F8D5A",
    "#B68B2C",
    "#A05A8F",
    "#7E64A8",
    "#4F9CA6",
    "#B36B55",
    "#7EA14A",
    "#607D8B",
    "#B99A2F",
]


@dataclass(frozen=True)
class KGVisualStyle:
    """Graphviz style options for a static knowledge-graph image."""

    engine: str = "sfdp"
    rankdir: str = "LR"
    graph_bgcolor: str = "transparent"
    node_font: str = "Helvetica"
    edge_font: str = "Helvetica"
    graph_font: str = "Helvetica"
    edge_color: str = "#6B7280"
    literal_font_color: str = "#4B5563"
    title: str | None = None
    show_legend: bool = True


@dataclass(frozen=True)
class KGVisualConfig:
    """Configuration for projecting RDF data into a readable ABox image."""

    rdf_format: str | None = None
    include_schema_edges: bool = False
    include_literals: bool = False
    include_isolated_nodes: bool = False
    max_literal_properties_per_node: int = 4
    max_literal_chars: int = 90
    max_nodes: int | None = None
    max_edges: int | None = None
    focus_classes: frozenset[str] = field(default_factory=frozenset)
    focus_properties: frozenset[str] = field(default_factory=frozenset)
    hide_properties: frozenset[str] = field(default_factory=frozenset)
    hide_classes: frozenset[str] = field(default_factory=frozenset)
    style: KGVisualStyle = field(default_factory=KGVisualStyle)


@dataclass(frozen=True)
class KGNode:
    id: str
    label: str
    classes: tuple[str, ...] = ()
    literal_properties: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class KGEdge:
    source: str
    target: str
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class KGProjection:
    nodes: Mapping[str, KGNode]
    edges: tuple[KGEdge, ...]
    class_counts: Mapping[str, int]


def render_knowledge_graph(
    input_path: str | Path,
    output_path: str | Path,
    config: KGVisualConfig | None = None,
) -> Path:
    """Read an RDF/OWL file and render a static knowledge-graph image.

    Args:
        input_path: RDF file path, e.g. `.ttl`, `.rdf`, `.owl`, `.nt`, `.jsonld`.
        output_path: Target image path. Supported by Graphviz, e.g. `.svg`, `.png`, `.pdf`.
        config: Optional visualization and semantic-projection configuration.

    Returns:
        The rendered output path.

    Raises:
        FileNotFoundError: If the RDF input file does not exist.
        RuntimeError: If Graphviz executable is unavailable.
        ValueError: If the graph projection is empty or output format is missing.
    """

    cfg = config or KGVisualConfig()
    source = Path(input_path)
    target = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"RDF input file not found: {source}")
    if not target.suffix:
        raise ValueError("output_path must include an image suffix, for example .svg, .png, or .pdf")
    if shutil.which(cfg.style.engine) is None:
        raise RuntimeError(
            f"Graphviz executable '{cfg.style.engine}' was not found on PATH. "
            "Install Graphviz and ensure its binaries are available."
        )

    rdf_graph = Graph()
    rdf_graph.parse(source, format=cfg.rdf_format or guess_rdf_format(source))

    projection = project_abox(rdf_graph, cfg)
    if not projection.nodes or not projection.edges:
        raise ValueError(
            "The projected knowledge graph is empty. Try include_schema_edges=True, "
            "include_isolated_nodes=True, or relax focus/hide filters."
        )

    dot = build_graphviz(projection, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)

    output_format = target.suffix.lstrip(".").lower()
    stem_without_suffix = str(target.with_suffix(""))
    rendered_path = Path(dot.render(stem_without_suffix, format=output_format, cleanup=True))
    return rendered_path


def project_abox(rdf_graph: Graph, config: KGVisualConfig) -> KGProjection:
    """Project RDF triples into an ABox-oriented directed property graph."""

    labels = _labels_by_resource(rdf_graph)
    types_by_resource = _types_by_resource(rdf_graph)
    resource_nodes: set[URIRef | BNode] = set()
    grouped_predicates: dict[tuple[str, str], set[str]] = defaultdict(set)

    for subject, predicate, object_ in rdf_graph:
        if not isinstance(subject, (URIRef, BNode)):
            continue
        if not isinstance(predicate, URIRef):
            continue

        predicate_id = str(predicate)
        if predicate_id in config.hide_properties:
            continue
        if config.focus_properties and predicate_id not in config.focus_properties:
            continue
        if not config.include_schema_edges and predicate in SCHEMA_PREDICATES:
            continue

        if isinstance(object_, (URIRef, BNode)):
            if not config.include_schema_edges and _looks_like_schema_triple(rdf_graph, subject, predicate, object_):
                continue
            if _resource_hidden_by_class(subject, types_by_resource, config):
                continue
            if _resource_hidden_by_class(object_, types_by_resource, config):
                continue
            if config.focus_classes and not (
                _resource_has_focus_class(subject, types_by_resource, config)
                or _resource_has_focus_class(object_, types_by_resource, config)
            ):
                continue

            source_id = _resource_id(subject)
            target_id = _resource_id(object_)
            resource_nodes.add(subject)
            resource_nodes.add(object_)
            grouped_predicates[(source_id, target_id)].add(_compact_label(rdf_graph, predicate, labels))

    if config.include_isolated_nodes:
        for resource, classes in types_by_resource.items():
            if _resource_hidden_by_class(resource, types_by_resource, config):
                continue
            if config.focus_classes and not _resource_has_focus_class(resource, types_by_resource, config):
                continue
            if not config.include_schema_edges and any(cls in SCHEMA_CLASSES for cls in classes):
                continue
            resource_nodes.add(resource)

    literal_properties = _literal_properties_by_resource(rdf_graph, labels, config)
    node_candidates = list(resource_nodes)

    if config.max_nodes is not None and len(node_candidates) > config.max_nodes:
        node_candidates = _top_resources_by_degree(node_candidates, grouped_predicates, config.max_nodes)

    allowed_node_ids = {_resource_id(resource) for resource in node_candidates}
    nodes = {
        _resource_id(resource): KGNode(
            id=_resource_id(resource),
            label=_resource_label(rdf_graph, resource, labels),
            classes=tuple(
                sorted(
                    _compact_label(rdf_graph, cls, labels)
                    for cls in types_by_resource.get(resource, ())
                    if not _is_schema_class(cls)
                )
            ),
            literal_properties=tuple(literal_properties.get(resource, ())),
        )
        for resource in node_candidates
    }

    edges = tuple(
        KGEdge(source=source, target=target, predicates=tuple(sorted(predicates)))
        for (source, target), predicates in grouped_predicates.items()
        if source in allowed_node_ids and target in allowed_node_ids
    )
    if config.max_edges is not None:
        edges = edges[: config.max_edges]

    connected_node_ids = {edge.source for edge in edges} | {edge.target for edge in edges}
    if not config.include_isolated_nodes:
        nodes = {node_id: node for node_id, node in nodes.items() if node_id in connected_node_ids}

    class_counts = Counter(
        node.classes[0] if node.classes else "Resource"
        for node in nodes.values()
    )

    return KGProjection(nodes=nodes, edges=edges, class_counts=dict(class_counts))


def build_graphviz(projection: KGProjection, config: KGVisualConfig) -> Digraph:
    """Build a Graphviz Digraph from a projected knowledge graph."""

    style = config.style
    dot = Digraph("knowledge_graph", engine=style.engine)
    dot.attr(
        "graph",
        bgcolor=style.graph_bgcolor,
        fontname=style.graph_font,
        overlap="false",
        splines="true",
        outputorder="edgesfirst",
        rankdir=style.rankdir,
        pad="0.35",
        nodesep="0.45",
        ranksep="0.75",
        concentrate="false",
        label=style.title or "",
        labelloc="t",
        fontsize="22",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname=style.node_font,
        fontsize="11",
        margin="0.10,0.07",
        penwidth="1.35",
    )
    dot.attr(
        "edge",
        color=style.edge_color,
        fontname=style.edge_font,
        fontsize="9",
        arrowsize="0.65",
        penwidth="1.05",
    )

    class_to_palette_index = _class_palette_index(projection.nodes.values())

    for node in sorted(projection.nodes.values(), key=lambda n: n.label.lower()):
        main_class = node.classes[0] if node.classes else "Resource"
        palette_index = class_to_palette_index.get(main_class, 0) % len(PALETTE)
        dot.node(
            node.id,
            label=_node_label(node, config),
            fillcolor=PALETTE[palette_index],
            color=BORDER_PALETTE[palette_index],
        )

    for edge in projection.edges:
        edge_label = " / ".join(edge.predicates)
        dot.edge(edge.source, edge.target, label=edge_label)

    if style.show_legend and projection.class_counts:
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(
                label="Legend",
                fontsize="12",
                color="#D1D5DB",
                style="rounded",
                fontname=style.graph_font,
            )
            for class_name, count in sorted(projection.class_counts.items()):
                palette_index = class_to_palette_index.get(class_name, 0) % len(PALETTE)
                legend.node(
                    f"legend_{_safe_id(class_name)}",
                    label=f"{class_name} ({count})",
                    shape="box",
                    style="rounded,filled",
                    fillcolor=PALETTE[palette_index],
                    color=BORDER_PALETTE[palette_index],
                    fontname=style.node_font,
                    fontsize="10",
                )

    return dot


def guess_rdf_format(path: Path) -> str:
    """Infer an RDFLib parser format from a file suffix."""

    suffix = path.suffix.lower()
    if suffix in {".ttl", ".turtle"}:
        return "turtle"
    if suffix in {".rdf", ".owl", ".xml"}:
        return "xml"
    if suffix in {".nt", ".ntriples"}:
        return "nt"
    if suffix in {".nq", ".nquads"}:
        return "nquads"
    if suffix in {".jsonld", ".json"}:
        return "json-ld"
    if suffix in {".trig"}:
        return "trig"
    raise ValueError(f"Cannot infer RDF format from suffix '{suffix}'. Pass rdf_format explicitly.")


def _labels_by_resource(rdf_graph: Graph) -> dict[URIRef | BNode, str]:
    labels: dict[URIRef | BNode, str] = {}
    for predicate in LABEL_PREDICATES:
        for subject, label in rdf_graph.subject_objects(predicate):
            if isinstance(subject, (URIRef, BNode)) and isinstance(label, Literal):
                labels.setdefault(subject, str(label))
    return labels


def _types_by_resource(rdf_graph: Graph) -> dict[URIRef | BNode, set[URIRef]]:
    types: dict[URIRef | BNode, set[URIRef]] = defaultdict(set)
    for subject, object_ in rdf_graph.subject_objects(RDF.type):
        if isinstance(subject, (URIRef, BNode)) and isinstance(object_, URIRef):
            types[subject].add(object_)
    return types


def _literal_properties_by_resource(
    rdf_graph: Graph,
    labels: Mapping[URIRef | BNode, str],
    config: KGVisualConfig,
) -> dict[URIRef | BNode, list[tuple[str, str]]]:
    if not config.include_literals:
        return {}

    literal_properties: dict[URIRef | BNode, list[tuple[str, str]]] = defaultdict(list)
    for subject, predicate, object_ in rdf_graph:
        if not isinstance(subject, (URIRef, BNode)):
            continue
        if not isinstance(predicate, URIRef):
            continue
        if not isinstance(object_, Literal):
            continue
        if predicate in SCHEMA_PREDICATES or predicate in LABEL_PREDICATES:
            continue
        if str(predicate) in config.hide_properties:
            continue
        value = _truncate(str(object_), config.max_literal_chars)
        literal_properties[subject].append((_compact_label(rdf_graph, predicate, labels), value))

    for subject, props in literal_properties.items():
        literal_properties[subject] = sorted(props)[: config.max_literal_properties_per_node]
    return literal_properties


def _top_resources_by_degree(
    resources: Iterable[URIRef | BNode],
    grouped_predicates: Mapping[tuple[str, str], set[str]],
    max_nodes: int,
) -> list[URIRef | BNode]:
    degree = Counter()
    for source_id, target_id in grouped_predicates:
        degree[source_id] += 1
        degree[target_id] += 1
    return sorted(resources, key=lambda resource: degree[_resource_id(resource)], reverse=True)[:max_nodes]


def _looks_like_schema_triple(
    rdf_graph: Graph,
    subject: URIRef | BNode,
    predicate: URIRef,
    object_: URIRef | BNode,
) -> bool:
    if predicate in SCHEMA_PREDICATES:
        return True
    if isinstance(subject, URIRef) and (subject, RDF.type, None) in rdf_graph:
        if any(cls in SCHEMA_CLASSES for cls in rdf_graph.objects(subject, RDF.type)):
            return True
    if isinstance(object_, URIRef) and (object_, RDF.type, None) in rdf_graph:
        if any(cls in SCHEMA_CLASSES for cls in rdf_graph.objects(object_, RDF.type)):
            return True
    return False


def _resource_hidden_by_class(
    resource: URIRef | BNode,
    types_by_resource: Mapping[URIRef | BNode, set[URIRef]],
    config: KGVisualConfig,
) -> bool:
    resource_types = {str(cls) for cls in types_by_resource.get(resource, set())}
    return bool(resource_types & set(config.hide_classes))


def _resource_has_focus_class(
    resource: URIRef | BNode,
    types_by_resource: Mapping[URIRef | BNode, set[URIRef]],
    config: KGVisualConfig,
) -> bool:
    return bool({str(cls) for cls in types_by_resource.get(resource, set())} & set(config.focus_classes))


def _resource_label(
    rdf_graph: Graph,
    resource: URIRef | BNode,
    labels: Mapping[URIRef | BNode, str],
) -> str:
    if resource in labels:
        return labels[resource]
    if isinstance(resource, BNode):
        return f"blank:{str(resource)[:8]}"
    return _qname_or_local(rdf_graph, resource)


def _compact_label(
    rdf_graph: Graph,
    resource: URIRef | BNode,
    labels: Mapping[URIRef | BNode, str],
) -> str:
    if resource in labels:
        return labels[resource]
    if isinstance(resource, BNode):
        return f"blank:{str(resource)[:8]}"
    return _qname_or_local(rdf_graph, resource)


def _qname_or_local(rdf_graph: Graph, uri: URIRef) -> str:
    try:
        qname = rdf_graph.namespace_manager.normalizeUri(uri)
        if qname.startswith("<") and qname.endswith(">"):
            return _local_name(str(uri))
        return qname
    except Exception:  # pragma: no cover - defensive fallback for malformed namespaces
        return _local_name(str(uri))


def _local_name(uri: str) -> str:
    tail = re.split(r"[/#]", uri.rstrip("/#"))[-1]
    if not tail:
        return uri
    return re.sub(r"[_-]+", " ", tail).strip()


def _resource_id(resource: URIRef | BNode) -> str:
    raw = str(resource)
    return "n_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _safe_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _node_label(node: KGNode, config: KGVisualConfig) -> str:
    lines = [_truncate(node.label, 45)]
    if node.classes:
        lines.append(f"«{', '.join(node.classes[:2])}»")
    if config.include_literals and node.literal_properties:
        lines.append("────────")
        for key, value in node.literal_properties:
            lines.append(f"{_truncate(key, 24)}: {_truncate(value, config.max_literal_chars)}")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 1)].rstrip() + "…"


def _is_schema_class(cls: URIRef) -> bool:
    return cls in SCHEMA_CLASSES or str(cls).startswith(str(XSD))


def _class_palette_index(nodes: Iterable[KGNode]) -> dict[str, int]:
    classes = sorted({(node.classes[0] if node.classes else "Resource") for node in nodes})
    return {class_name: index for index, class_name in enumerate(classes)}


def _split_csv_uris(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a populated RDF/OWL ontology as a readable ABox knowledge-graph image."
    )
    parser.add_argument("input", type=Path, help="Input RDF file, for example populated_ontology.ttl")
    parser.add_argument("output", type=Path, help="Output image path, for example knowledge_graph.svg")
    parser.add_argument("--format", dest="rdf_format", default=None, help="Explicit RDFLib parser format, e.g. turtle")
    parser.add_argument(
        "--engine",
        default="sfdp",
        choices=("dot", "neato", "fdp", "sfdp", "circo", "twopi"),
        help="Graphviz layout engine. Use sfdp/fdp for dense ABox graphs, dot for hierarchical graphs.",
    )
    parser.add_argument("--title", default=None, help="Optional title rendered above the graph")
    parser.add_argument("--max-nodes", type=int, default=None, help="Keep only the highest-degree N nodes")
    parser.add_argument("--max-edges", type=int, default=None, help="Keep at most N rendered edges")
    parser.add_argument("--include-literals", action="store_true", help="Show literal data properties inside node labels")
    parser.add_argument("--include-schema-edges", action="store_true", help="Render schema/TBox triples too")
    parser.add_argument("--include-isolated-nodes", action="store_true", help="Render typed resources even when disconnected")
    parser.add_argument("--no-legend", action="store_true", help="Hide the class legend")
    parser.add_argument("--focus-classes", default=None, help="Comma-separated full class IRIs to keep")
    parser.add_argument("--focus-properties", default=None, help="Comma-separated full predicate IRIs to keep")
    parser.add_argument("--hide-classes", default=None, help="Comma-separated full class IRIs to hide")
    parser.add_argument("--hide-properties", default=None, help="Comma-separated full predicate IRIs to hide")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = KGVisualConfig(
        rdf_format=args.rdf_format,
        include_schema_edges=args.include_schema_edges,
        include_literals=args.include_literals,
        include_isolated_nodes=args.include_isolated_nodes,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        focus_classes=_split_csv_uris(args.focus_classes),
        focus_properties=_split_csv_uris(args.focus_properties),
        hide_classes=_split_csv_uris(args.hide_classes),
        hide_properties=_split_csv_uris(args.hide_properties),
        style=KGVisualStyle(
            engine=args.engine,
            title=args.title,
            show_legend=not args.no_legend,
        ),
    )
    output_path = render_knowledge_graph(args.input, args.output, config)
    print(f"Rendered knowledge graph: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
