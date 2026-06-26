"""Ontology relation predicates and type-pair routing.

This module treats OWL object properties as candidate relation predicates for
relation extraction. It reads object-property labels/descriptions/domains/ranges
from the ontology and exposes a router that answers:

    given source class + target class, which ontology relations are legal?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import networkx as nx

from owlapy.class_expression import OWLClass
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_reasoner import SyncReasoner

from ontology.tbox import (
    DESCRIPTION_PREDICATES,
    LABEL_PREDICATES,
    first_literal,
    human_label,
    iri_text,
    local_name,
    parse_rdf_graph,
)


__all__ = [
    "RelationPropertySpec",
    "RelationRouter",
    "build_relation_router",
]


@dataclass(frozen=True, slots=True)
class RelationPropertySpec:
    """Serializable runtime view of one ontology object property."""

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


def _build_relation_property_specs(
    *,
    onto: SyncOntology,
    ontology_path: str | Path,
) -> dict[str, RelationPropertySpec]:
    """Extract object-property specs used by relation routing."""

    rdf_graph = parse_rdf_graph(ontology_path)
    reasoner = SyncReasoner(ontology=onto, reasoner="Structural")

    specs: dict[str, RelationPropertySpec] = {}

    for prop in onto.object_properties_in_signature():
        prop_iri = iri_text(prop)
        label = first_literal(rdf_graph, prop_iri, LABEL_PREDICATES) or local_name(prop_iri)
        description = first_literal(rdf_graph, prop_iri, DESCRIPTION_PREDICATES) or ""

        try:
            domains = _named_class_iris(reasoner.object_property_domains(prop))
        except Exception:
            domains = ()

        try:
            ranges = _named_class_iris(reasoner.object_property_ranges(prop))
        except Exception:
            ranges = ()

        specs[prop_iri] = RelationPropertySpec(
            iri=prop_iri,
            local_name=local_name(prop_iri),
            label=label,
            human_readable_label=human_label(label),
            description=description,
            domains=domains,
            ranges=ranges,
        )

    return specs


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

    The router accepts full class IRIs, local names, and labels. Internally it
    canonicalizes class identifiers against the class graph and checks whether
    the source/target classes satisfy each property domain/range.
    """

    relation_specs: dict[str, RelationPropertySpec]
    class_graph: nx.DiGraph
    _cache: dict[tuple[str, str], tuple[RelationPropertySpec, ...]] = field(default_factory=dict)
    _class_alias_to_iri: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._class_alias_to_iri = self._build_class_alias_index()

    def candidates_for(
        self,
        source_class_iri: str,
        target_class_iri: str,
    ) -> tuple[RelationPropertySpec, ...]:
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
    ) -> tuple[RelationPropertySpec, ...]:
        candidates: list[RelationPropertySpec] = []

        for spec in self.relation_specs.values():
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

        if _is_thing_iri(parent):
            return True

        if child not in self.class_graph or parent not in self.class_graph:
            return False

        # Class graph edges are parent -> child.
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
    onto: SyncOntology,
    ontology_path: str | Path,
    class_graph: nx.DiGraph,
) -> RelationRouter:
    """Build a relation router directly from an ontology and class graph."""

    return RelationRouter(
        relation_specs=_build_relation_property_specs(
            onto=onto,
            ontology_path=ontology_path,
        ),
        class_graph=class_graph,
    )
