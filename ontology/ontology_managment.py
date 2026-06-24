"""Minimal OWLAPY ontology helpers for the NLP ontology-population pipeline.

Design:
- OWLAPY is the ontology backend.
- RDFLibReasoner, shipped by OWLAPY, is used to derive the class hierarchy.
- RDFLib is used only for rdfs:label / rdfs:comment lookup because those are
  RDF annotations and OWLAPY examples do not expose a simpler metadata API.
- NetworkX is only a derived view, not the ontology source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import networkx as nx
from rdflib import DCTERMS, Graph, Literal, Namespace, RDFS, SKOS, URIRef

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
from owlapy.owl_reasoner import SyncReasoner


SCHEMA = Namespace("https://schema.org/")
DESCRIPTION_PREDICATES = (RDFS.comment, SKOS.definition, DCTERMS.description, SCHEMA.description)
LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)


# ---------- tiny generic helpers ----------


def load_ontology(path: str | Path) -> SyncOntology:
    """Load an OWL/OWL2/Turtle ontology with OWLAPY."""

    return SyncOntology(str(Path(path)))


def save_ontology(onto: SyncOntology, path: str | Path, *, document_format: str | None = None) -> None:
    """Save the ontology using OWLAPY."""

    onto.save(path=str(Path(path)), document_format=document_format)


def iri_text(x: Any) -> str:
    """Best-effort string IRI for OWLAPY objects."""

    return getattr(x, "str", None) or str(getattr(x, "iri", x))


def local_name(iri: str) -> str:
    text = str(iri)
    return text.rsplit("#", 1)[-1] if "#" in text else text.rstrip("/").rsplit("/", 1)[-1]


def human_label(label: str) -> str:
    label = re.sub(r"[_\-]+", " ", str(label))
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in label.split())


def _rdf(path: str | Path) -> Graph:
    """Parse RDF annotations from the same ontology file.

    This is deliberately small. OWLAPY remains the ontology backend; RDFLib is
    only used to read labels/comments from the concrete serialization.
    """

    path = Path(path)
    g = Graph()
    if path.suffix.lower() == ".ttl":
        g.parse(str(path), format="turtle")
    else:
        try:
            g.parse(str(path), format="xml")
        except Exception:
            g.parse(str(path))
    return g


def _first_literal(g: Graph, iri: str, predicates: tuple[URIRef, ...]) -> str | None:
    for p in predicates:
        for obj in g.objects(URIRef(iri), p):
            if isinstance(obj, Literal) and str(obj).strip():
                return str(obj).strip()
    return None


# ---------- validation ----------


def require_object_property_descriptions(onto: SyncOntology, ontology_path: str | Path) -> None:
    """Fail if an OWL object property has no description annotation.

    Accepted descriptions: rdfs:comment, skos:definition, dcterms:description,
    schema:description.
    """

    g = _rdf(ontology_path)
    missing = []

    for prop in onto.object_properties_in_signature():
        prop_iri = iri_text(prop)
        if not _first_literal(g, prop_iri, DESCRIPTION_PREDICATES):
            missing.append(prop_iri)

    if missing:
        joined = "\n".join(f"- {iri}" for iri in sorted(missing))
        raise ValueError(
            "Every owl:ObjectProperty must have a description annotation.\n"
            "Add rdfs:comment, skos:definition, dcterms:description, or schema:description for:\n"
            f"{joined}"
        )


# ---------- class DAG export ----------


def class_dag(onto: SyncOntology, ontology_path: str | Path | None = None) -> nx.DiGraph:
    """Export the named-class taxonomy as a NetworkX DiGraph.

    Node ids are class IRI strings. Edges are parent -> child.
    """

    g = _rdf(ontology_path) if ontology_path is not None else None
    reasoner = SyncReasoner(ontology=onto, reasoner="Structural")

    classes = list(onto.classes_in_signature())
    class_iris = {iri_text(c) for c in classes}

    graph = nx.DiGraph()

    for cls in classes:
        iri = iri_text(cls)
        label = _first_literal(g, iri, LABEL_PREDICATES) if g is not None else None
        desc = _first_literal(g, iri, DESCRIPTION_PREDICATES) if g is not None else None
        label = label or local_name(iri)

        graph.add_node(
            iri,
            label=label,
            human_readable_label=human_label(label),
            description=desc or "",
            local_name=local_name(iri),
        )

    for parent in classes:
        parent_iri = iri_text(parent)

        for child in reasoner.sub_classes(parent, direct=True):
            child_iri = iri_text(child)

            if child_iri in class_iris and child_iri != parent_iri:
                graph.add_edge(parent_iri, child_iri)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"The class hierarchy is not a DAG: {nx.find_cycle(graph)}")

    return graph


# ---------- ABox population ----------


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
    """Add: cluster_N rdf:type Class."""

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
    """Add: cluster_A objectProperty cluster_B."""

    src = cluster_individual(individual_ns, source_cluster_id)
    dst = cluster_individual(individual_ns, target_cluster_id)
    prop = OWLObjectProperty(IRI.create(property_iri))
    onto.add_axiom([OWLDeclarationAxiom(src), OWLDeclarationAxiom(dst), OWLObjectPropertyAssertionAxiom(src, prop, dst)])


def assert_cluster_value(
    onto: SyncOntology,
    *,
    individual_ns: str,
    cluster_id: int | str,
    property_iri: str,
    value: Any,
) -> None:
    """Add: cluster_N dataProperty literal_value."""

    ind = cluster_individual(individual_ns, cluster_id)
    prop = OWLDataProperty(IRI.create(property_iri))
    onto.add_axiom([OWLDeclarationAxiom(ind), OWLDataPropertyAssertionAxiom(ind, prop, OWLLiteral(value))])


# ---------- one-call convenience ----------


def load_tbox(
    path: str | Path,
    *,
    require_property_descriptions: bool = True,
) -> tuple[SyncOntology, nx.DiGraph]:
    """Load ontology, validate object-property descriptions, return ontology + class DAG."""

    onto = load_ontology(path)
    if require_property_descriptions:
        require_object_property_descriptions(onto, path)
    return onto, class_dag(onto, path)

# ---------- relation catalog and type-pair routing ----------

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ObjectPropertySpec:
    """Small serializable view of one OWL object property."""

    iri: str
    local_name: str
    label: str
    human_readable_label: str
    description: str
    domains: tuple[str, ...]
    ranges: tuple[str, ...]


def _named_class_iris(expressions: Any) -> tuple[str, ...]:
    """Keep only named OWLClass domains/ranges for V1 hard routing."""

    iris: list[str] = []
    for expr in expressions:
        if isinstance(expr, OWLClass):
            iris.append(iri_text(expr))
    return tuple(sorted(set(iris)))


def build_relation_catalog(
    *,
    onto: SyncOntology,
    ontology_path: str | Path,
) -> dict[str, ObjectPropertySpec]:
    """Extract object-property labels, descriptions, domains, and ranges.

    This is intentionally a compact catalog for downstream neural relation
    alignment. Complex domain/range class expressions are ignored in V1 hard
    routing; they can be supported later if the ontology requires them.
    """

    rdf_graph = _rdf(ontology_path)
    reasoner = SyncReasoner(ontology=onto, reasoner="Structural")

    catalog: dict[str, ObjectPropertySpec] = {}

    for prop in onto.object_properties_in_signature():
        prop_iri = iri_text(prop)
        label = _first_literal(rdf_graph, prop_iri, LABEL_PREDICATES) or local_name(prop_iri)
        description = _first_literal(rdf_graph, prop_iri, DESCRIPTION_PREDICATES) or ""

        try:
            domains = _named_class_iris(reasoner.object_property_domains(prop))
        except Exception:
            domains = ()

        try:
            ranges = _named_class_iris(reasoner.object_property_ranges(prop))
        except Exception:
            ranges = ()

        catalog[prop_iri] = ObjectPropertySpec(
            iri=prop_iri,
            local_name=local_name(prop_iri),
            label=label,
            human_readable_label=human_label(label),
            description=description,
            domains=domains,
            ranges=ranges,
        )

    return catalog


def _normalize_identifier_text(value: str) -> str:
    """Normalize labels/local names for alias matching."""

    text = str(value).strip()
    text = local_name(text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def _is_thing_iri(value: str) -> bool:
    text = str(value).strip()
    return text in {
        "Thing",
        "owl:Thing",
        "http://www.w3.org/2002/07/owl#Thing",
        "https://www.w3.org/2002/07/owl#Thing",
    } or text.endswith("#Thing")


@dataclass(slots=True)
class RelationRouter:
    """Type-pair router over object-property domain/range constraints.

    The router is intentionally tolerant about class identifiers. It accepts:
    - full ontology class IRIs,
    - local names such as ``HumanCharacter``,
    - labels such as ``Human character``.

    Edges in ``class_graph`` must be parent -> child.
    """

    relation_catalog: dict[str, ObjectPropertySpec]
    class_graph: nx.DiGraph
    _cache: dict[tuple[str, str], tuple[ObjectPropertySpec, ...]] = field(default_factory=dict)
    _class_alias_to_iri: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._class_alias_to_iri = self._build_class_alias_index()

    def candidates_for(
        self,
        source_class_iri: str,
        target_class_iri: str,
    ) -> tuple[ObjectPropertySpec, ...]:
        source = self._canonical_class_iri(source_class_iri)
        target = self._canonical_class_iri(target_class_iri)

        key = (source, target)
        if key not in self._cache:
            self._cache[key] = self._compute_candidates(
                source_class_iri=source,
                target_class_iri=target,
            )
        return self._cache[key]

    def explain_pair(self, source_class_iri: str, target_class_iri: str) -> dict[str, Any]:
        """Small debug helper for notebook inspection."""

        source = self._canonical_class_iri(source_class_iri)
        target = self._canonical_class_iri(target_class_iri)
        candidates = self.candidates_for(source_class_iri, target_class_iri)
        return {
            "source_input": source_class_iri,
            "source_canonical": source,
            "source_in_graph": source in self.class_graph or _is_thing_iri(source),
            "target_input": target_class_iri,
            "target_canonical": target,
            "target_in_graph": target in self.class_graph or _is_thing_iri(target),
            "n_candidates": len(candidates),
            "candidate_properties": [spec.local_name for spec in candidates],
        }

    def _compute_candidates(
        self,
        *,
        source_class_iri: str,
        target_class_iri: str,
    ) -> tuple[ObjectPropertySpec, ...]:
        candidates: list[ObjectPropertySpec] = []

        for spec in self.relation_catalog.values():
            domains = self._most_specific_classes(spec.domains)
            ranges = self._most_specific_classes(spec.ranges)

            if not domains or not ranges:
                continue

            source_ok = any(
                self._is_same_or_subclass(source_class_iri, domain_iri)
                for domain_iri in domains
            )
            if not source_ok:
                continue

            target_ok = any(
                self._is_same_or_subclass(target_class_iri, range_iri)
                for range_iri in ranges
            )
            if target_ok:
                candidates.append(spec)

        return tuple(candidates)

    def _build_class_alias_index(self) -> dict[str, str]:
        aliases: dict[str, str] = {}

        for iri, data in self.class_graph.nodes(data=True):
            values = {
                str(iri),
                local_name(str(iri)),
                data.get("local_name", ""),
                data.get("label", ""),
                data.get("human_readable_label", ""),
            }

            for value in values:
                if value is None or not str(value).strip():
                    continue
                aliases[str(value).strip()] = str(iri)
                aliases[_normalize_identifier_text(str(value))] = str(iri)

        return aliases

    def _canonical_class_iri(self, value: str) -> str:
        text = str(value).strip()

        if text in self.class_graph or _is_thing_iri(text):
            return text

        if text in self._class_alias_to_iri:
            return self._class_alias_to_iri[text]

        normalized = _normalize_identifier_text(text)
        if normalized in self._class_alias_to_iri:
            return self._class_alias_to_iri[normalized]

        return text

    def _is_same_or_subclass(self, child_iri: str, parent_iri: str) -> bool:
        child = self._canonical_class_iri(child_iri)
        parent = self._canonical_class_iri(parent_iri)

        if parent == child:
            return True

        # owl:Thing is a wildcard parent for any class.
        if _is_thing_iri(parent):
            return True

        if child not in self.class_graph or parent not in self.class_graph:
            return False

        # class_dag edges are parent -> child.
        return nx.has_path(self.class_graph, parent, child)

    def _most_specific_classes(self, class_iris: tuple[str, ...]) -> tuple[str, ...]:
        """Drop superclass pollution from reasoner domain/range output.

        Example:
            (Thing, NarrativeEntity, Place) -> (Place,)
        """

        canonical = tuple(dict.fromkeys(self._canonical_class_iri(iri) for iri in class_iris))

        result: list[str] = []
        for candidate in canonical:
            if _is_thing_iri(candidate):
                # Keep Thing only if it is the only usable constraint.
                if len(canonical) == 1:
                    result.append(candidate)
                continue

            is_superclass_of_another = False
            for other in canonical:
                if other == candidate or _is_thing_iri(other):
                    continue
                if candidate in self.class_graph and other in self.class_graph:
                    if nx.has_path(self.class_graph, candidate, other):
                        is_superclass_of_another = True
                        break

            if not is_superclass_of_another:
                result.append(candidate)

        return tuple(result)

def build_relation_router(
    *,
    class_graph: nx.DiGraph,
    relation_catalog: dict[str, ObjectPropertySpec],
) -> RelationRouter:
    return RelationRouter(
        relation_catalog=relation_catalog,
        class_graph=class_graph,
    )

