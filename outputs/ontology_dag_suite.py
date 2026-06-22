from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import re

from rdflib import Graph, Literal, OWL, RDF, RDFS, SKOS, URIRef


# ---------------------------------------------------------------------------
# 1. Constants and errors
# ---------------------------------------------------------------------------

OWL_THING = str(OWL.Thing)
OWL_NOTHING = str(OWL.Nothing)


class OntologyDAGError(Exception):
    """Base exception for ontology DAG construction/navigation errors."""


class OntologyInputError(OntologyDAGError):
    """Raised when the input ontology file is invalid for this module."""


class OntologyCycleError(OntologyDAGError):
    """Raised when subclass relations do not form a DAG."""


class AmbiguousClassReferenceError(OntologyDAGError):
    """Raised when a non-IRI class reference resolves to multiple classes."""


# ---------------------------------------------------------------------------
# 2. Small utilities
# ---------------------------------------------------------------------------


def iri_to_local_name(iri: str) -> str:
    """Return the fragment/local part of an IRI."""
    text = str(iri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def camel_to_words(name: str) -> str:
    """Convert CamelCase/snake_case-ish identifiers into readable words."""
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def label_to_human_readible_name(label: str) -> str:
    """
    Normalize an ontology label into a compact natural-language class name.

    Example:
        forensicTrace -> Forensic Trace
        DNAEvidence   -> DNA Evidence
    """
    words = camel_to_words(label)
    if not words:
        return label
    return " ".join(word if word.isupper() else word.capitalize() for word in words.split())


def normalize_ref(text: str) -> str:
    """Normalize a class reference for label/local-name lookup."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def literal_to_str(value: Any) -> Optional[str]:
    if isinstance(value, Literal):
        return str(value)
    return None


def literal_lang(value: Literal) -> Optional[str]:
    return value.language.lower() if value.language else None


# ---------------------------------------------------------------------------
# 3. Data model
# ---------------------------------------------------------------------------


@dataclass
class OntologyNode:
    """A named ontology class node."""

    iri: str
    local_name: str
    label: str
    human_readible_label: str
    description: str = ""
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    depth_min: Optional[int] = None
    depth_max: Optional[int] = None

    def to_class_spec(self) -> Dict[str, str]:
        """Return a human_readible-compatible class spec."""
        description = self.description.strip()
        if not description:
            description = f"An entity belonging to the ontology class {self.human_readible_label}."
        return {"name": self.human_readible_label, "description": description}


@dataclass
class OntologyDAG:
    """
    Navigable class DAG extracted from a Turtle ontology.

    Edges are stored as parent -> child.
    Node references accepted by public methods:
        - full IRI
        - local name
        - rdfs/skos label
        - human_readible-normalized label
        - camel-to-words local-name form
    """

    nodes: Dict[str, OntologyNode]
    roots: List[str]
    source_path: str
    warnings: List[str] = field(default_factory=list)
    _resolution_index: Dict[str, Set[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_resolution_index()

    # ------------------------- Resolution -------------------------

    def _build_resolution_index(self) -> None:
        index: Dict[str, Set[str]] = defaultdict(set)
        for iri, node in self.nodes.items():
            candidates = {
                iri,
                node.local_name,
                camel_to_words(node.local_name),
                node.label,
                node.human_readible_label,
            }
            for candidate in candidates:
                normalized = normalize_ref(candidate)
                if normalized:
                    index[normalized].add(iri)
        self._resolution_index = dict(index)

    def resolve(self, class_ref: str) -> Optional[str]:
        """
        Resolve a class reference to exactly one IRI.

        Returns None if the class is unknown.
        Raises AmbiguousClassReferenceError if the reference matches more than one class.
        """
        if class_ref in self.nodes:
            return class_ref

        normalized = normalize_ref(class_ref)
        matches = self._resolution_index.get(normalized, set())
        if not matches:
            return None
        if len(matches) > 1:
            readable = ", ".join(sorted(matches))
            raise AmbiguousClassReferenceError(
                f"Ambiguous ontology class reference {class_ref!r}. Matching IRIs: {readable}"
            )
        return next(iter(matches))

    def has_class(self, class_ref: str) -> bool:
        try:
            return self.resolve(class_ref) is not None
        except AmbiguousClassReferenceError:
            return True

    def get_node(self, class_ref: str) -> OntologyNode:
        iri = self.resolve(class_ref)
        if iri is None:
            raise KeyError(f"Unknown ontology class: {class_ref}")
        return self.nodes[iri]

    # ------------------------- Basic navigation -------------------------

    def children(self, class_ref: str) -> List[OntologyNode]:
        node = self.get_node(class_ref)
        return [self.nodes[child] for child in node.children]

    def parents(self, class_ref: str) -> List[OntologyNode]:
        node = self.get_node(class_ref)
        return [self.nodes[parent] for parent in node.parents]

    def is_root(self, class_ref: str) -> bool:
        return self.get_node(class_ref).iri in self.roots

    def is_leaf(self, class_ref: str) -> bool:
        return len(self.get_node(class_ref).children) == 0

    def class_specs_for_roots(self) -> List[Dict[str, str]]:
        return [self.nodes[root].to_class_spec() for root in self.roots]

    def class_specs_for_children(self, class_ref: str) -> List[Dict[str, str]]:
        return [child.to_class_spec() for child in self.children(class_ref)]

    def descendants(self, class_ref: str, include_self: bool = False) -> List[OntologyNode]:
        start = self.get_node(class_ref).iri
        seen: Set[str] = set()
        queue: deque[str] = deque([start])
        out: List[OntologyNode] = []

        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if include_self or current != start:
                out.append(self.nodes[current])
            queue.extend(self.nodes[current].children)
        return out

    def ancestors(self, class_ref: str, include_self: bool = False) -> List[OntologyNode]:
        start = self.get_node(class_ref).iri
        seen: Set[str] = set()
        queue: deque[str] = deque([start])
        out: List[OntologyNode] = []

        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if include_self or current != start:
                out.append(self.nodes[current])
            queue.extend(self.nodes[current].parents)
        return out

    def paths_to_roots(self, class_ref: str, *, max_paths: Optional[int] = None) -> List[List[OntologyNode]]:
        """
        Return root -> ... -> class paths.

        Multiple paths are normal in a DAG because of multiple inheritance.
        Use max_paths to protect callers from path explosion on dense DAGs.
        """
        target = self.get_node(class_ref).iri
        paths: List[List[str]] = []

        def rec(current: str, suffix: List[str]) -> None:
            if max_paths is not None and len(paths) >= max_paths:
                return
            parents = self.nodes[current].parents
            if not parents:
                paths.append([current] + suffix)
                return
            for parent in parents:
                rec(parent, [current] + suffix)

        rec(target, [])
        return [[self.nodes[iri] for iri in path] for path in paths]

    def as_label_tree(self, class_ref: Optional[str] = None, indent: int = 0) -> str:
        """Return a readable tree-like view. DAG nodes may appear more than once."""
        starts = [self.get_node(class_ref).iri] if class_ref else self.roots
        lines: List[str] = []

        def rec(iri: str, level: int) -> None:
            node = self.nodes[iri]
            lines.append("  " * level + f"- {node.human_readible_label}")
            for child in node.children:
                rec(child, level + 1)

        for root in starts:
            rec(root, indent)
        return "\n".join(lines)

    # ------------------------- Exports -------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "roots": self.roots,
            "root_labels": [self.nodes[iri].human_readible_label for iri in self.roots],
            "nodes": {iri: asdict(node) for iri, node in self.nodes.items()},
            "warnings": self.warnings,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_dot(self) -> str:
        """Return a stable GraphViz DOT representation of the class hierarchy."""
        lines = ["digraph ontology_dag {", "  rankdir=TB;"]

        def safe_id(iri: str) -> str:
            return "n" + sha1(iri.encode("utf-8")).hexdigest()[:16]

        for iri, node in self.nodes.items():
            label = node.human_readible_label.replace('"', '\\"')
            lines.append(f'  {safe_id(iri)} [label="{label}"];')
        for iri, node in self.nodes.items():
            parent_id = safe_id(iri)
            for child in node.children:
                lines.append(f"  {parent_id} -> {safe_id(child)};")
        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, path: str | Path) -> None:
        Path(path).write_text(self.to_dot(), encoding="utf-8")

    def to_networkx(self):
        """Return a networkx.DiGraph if networkx is installed."""
        import networkx as nx

        graph = nx.DiGraph()
        for iri, node in self.nodes.items():
            graph.add_node(iri, **asdict(node))
        for iri, node in self.nodes.items():
            for child in node.children:
                graph.add_edge(iri, child)
        return graph


# ---------------------------------------------------------------------------
# 4. Turtle -> navigable DAG builder
# ---------------------------------------------------------------------------


class TurtleOntologyDAGBuilder:
    """
    Build an OntologyDAG from one Turtle .ttl ontology file.

    Supported class declarations:
        ?class rdf:type owl:Class
        ?class rdf:type rdfs:Class
        ?child rdfs:subClassOf ?parent

    Supported human-readable metadata:
        rdfs:label
        skos:prefLabel
        rdfs:comment
        skos:definition

    Anonymous OWL class expressions are intentionally ignored as navigation nodes.
    """

    def __init__(self, *, preferred_languages: Sequence[Optional[str]] = ("en", None)):
        self.preferred_languages = tuple(lang.lower() if isinstance(lang, str) else None for lang in preferred_languages)

    def from_ttl(self, ttl_path: str | Path) -> OntologyDAG:
        path = Path(ttl_path)
        self._validate_ttl_path(path)

        graph = Graph()
        graph.parse(str(path), format="turtle")
        return self.from_graph(graph, source_path=str(path))

    def from_graph(self, graph: Graph, *, source_path: str = "<rdflib.Graph>") -> OntologyDAG:
        class_iris = self._collect_named_classes(graph)
        edges = self._collect_subclass_edges(graph, class_iris)
        self._raise_if_cycles(class_iris, edges)

        nodes = self._make_nodes(graph, class_iris)
        self._attach_edges(nodes, edges)

        roots = sorted(
            [iri for iri, node in nodes.items() if not node.parents and iri != OWL_NOTHING],
            key=lambda iri: nodes[iri].human_readible_label,
        )

        dag = OntologyDAG(nodes=nodes, roots=roots, source_path=source_path)
        self._assign_depths(dag)
        return dag

    def _validate_ttl_path(self, path: Path) -> None:
        if path.suffix.lower() != ".ttl":
            raise OntologyInputError(f"Expected a .ttl file, got: {path}")
        if not path.exists():
            raise OntologyInputError(f"TTL file does not exist: {path}")
        if not path.is_file():
            raise OntologyInputError(f"TTL path is not a file: {path}")

    def _collect_named_classes(self, graph: Graph) -> Set[str]:
        classes: Set[str] = set()

        for cls in graph.subjects(RDF.type, OWL.Class):
            if isinstance(cls, URIRef):
                classes.add(str(cls))

        for cls in graph.subjects(RDF.type, RDFS.Class):
            if isinstance(cls, URIRef):
                classes.add(str(cls))

        # Lightweight ontologies may omit explicit class declarations.
        for child, _predicate, parent in graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(child, URIRef):
                classes.add(str(child))
            if isinstance(parent, URIRef):
                classes.add(str(parent))

        classes.discard(OWL_NOTHING)
        classes.discard(OWL_THING)
        return classes

    def _collect_subclass_edges(self, graph: Graph, class_iris: Set[str]) -> Set[Tuple[str, str]]:
        edges: Set[Tuple[str, str]] = set()
        for child, _predicate, parent in graph.triples((None, RDFS.subClassOf, None)):
            if not isinstance(child, URIRef):
                continue
            if not isinstance(parent, URIRef):
                # Skip anonymous restrictions/intersections/unions as navigation parents.
                continue

            child_iri = str(child)
            parent_iri = str(parent)
            if child_iri == parent_iri:
                continue
            if child_iri in class_iris and parent_iri in class_iris:
                edges.add((child_iri, parent_iri))
        return edges

    def _make_nodes(self, graph: Graph, class_iris: Set[str]) -> Dict[str, OntologyNode]:
        nodes: Dict[str, OntologyNode] = {}
        for iri in sorted(class_iris):
            uri = URIRef(iri)
            local_name = iri_to_local_name(iri)
            label = self._first_literal(graph, uri, [RDFS.label, SKOS.prefLabel]) or camel_to_words(local_name) or local_name
            description = self._first_literal(graph, uri, [RDFS.comment, SKOS.definition]) or ""
            nodes[iri] = OntologyNode(
                iri=iri,
                local_name=local_name,
                label=label,
                human_readible_label=label_to_human_readible_name(label),
                description=description,
            )
        return nodes

    def _first_literal(self, graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> Optional[str]:
        literals: List[Literal] = []
        for predicate in predicates:
            for obj in graph.objects(subject, predicate):
                if isinstance(obj, Literal) and str(obj).strip():
                    literals.append(obj)

        if not literals:
            return None

        by_lang: Dict[Optional[str], List[Literal]] = defaultdict(list)
        for lit in literals:
            by_lang[literal_lang(lit)].append(lit)

        for lang in self.preferred_languages:
            if lang in by_lang:
                return str(by_lang[lang][0])

        return str(literals[0])

    def _attach_edges(self, nodes: Dict[str, OntologyNode], edges: Set[Tuple[str, str]]) -> None:
        for child, parent in sorted(edges):
            if child not in nodes or parent not in nodes:
                continue
            nodes[child].parents.append(parent)
            nodes[parent].children.append(child)

        for node in nodes.values():
            node.parents = sorted(set(node.parents), key=lambda iri: nodes[iri].human_readible_label)
            node.children = sorted(set(node.children), key=lambda iri: nodes[iri].human_readible_label)

    def _raise_if_cycles(self, class_iris: Set[str], edges: Set[Tuple[str, str]]) -> None:
        children_by_parent: Dict[str, List[str]] = {iri: [] for iri in class_iris}
        for child, parent in edges:
            children_by_parent.setdefault(parent, []).append(child)

        visited: Set[str] = set()
        stack: Set[str] = set()
        stack_path: List[str] = []

        def dfs(node: str) -> None:
            if node in stack:
                cycle_start = stack_path.index(node)
                cycle = stack_path[cycle_start:] + [node]
                readable = " -> ".join(iri_to_local_name(iri) for iri in cycle)
                raise OntologyCycleError(f"Subclass relations contain a cycle, so no DAG can be built: {readable}")
            if node in visited:
                return

            visited.add(node)
            stack.add(node)
            stack_path.append(node)
            for child in children_by_parent.get(node, []):
                dfs(child)
            stack_path.pop()
            stack.remove(node)

        for iri in sorted(class_iris):
            dfs(iri)

    def _assign_depths(self, dag: OntologyDAG) -> None:
        """Assign shortest and longest root distances for every reachable DAG node."""
        indegree: Dict[str, int] = {iri: len(node.parents) for iri, node in dag.nodes.items()}
        queue: deque[str] = deque(sorted(dag.roots, key=lambda iri: dag.nodes[iri].human_readible_label))

        for root in dag.roots:
            dag.nodes[root].depth_min = 0
            dag.nodes[root].depth_max = 0

        while queue:
            parent = queue.popleft()
            parent_node = dag.nodes[parent]

            for child in parent_node.children:
                child_node = dag.nodes[child]
                next_min = 0 if parent_node.depth_min is None else parent_node.depth_min + 1
                next_max = 0 if parent_node.depth_max is None else parent_node.depth_max + 1

                if child_node.depth_min is None or next_min < child_node.depth_min:
                    child_node.depth_min = next_min
                if child_node.depth_max is None or next_max > child_node.depth_max:
                    child_node.depth_max = next_max

                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)


# ---------------------------------------------------------------------------
# 5. Public entrypoint
# ---------------------------------------------------------------------------


def build_ttl_dag(ttl_path: str | Path) -> OntologyDAG:
    """
    Parse one Turtle .ttl ontology file and return a navigable OntologyDAG.

    Example:
        dag = build_ttl_dag("ontology.ttl")
        print(dag.as_label_tree())
        print([node.human_readible_label for node in dag.children("Person")])
    """
    return TurtleOntologyDAGBuilder().from_ttl(ttl_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a .ttl ontology file and print a navigable DAG tree.")
    parser.add_argument("ttl", help="Path to a Turtle ontology file, e.g. ontology.ttl")
    args = parser.parse_args()

    dag = build_ttl_dag(args.ttl)
    print(f"Loaded classes: {len(dag.nodes)}")
    print("Roots:", ", ".join(dag.nodes[root].human_readible_label for root in dag.roots))
    print("\nHierarchy:")
    print(dag.as_label_tree())
