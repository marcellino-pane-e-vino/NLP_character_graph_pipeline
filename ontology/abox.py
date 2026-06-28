"""ABox population from annotation-layer outputs.

This module translates final pipeline annotations into OWLAPY ABox axioms.  It
materializes entities as named individuals, entity types as class assertions,
and relation assertions as object-property assertions.
"""

from __future__ import annotations

import re
import unicodedata

from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLAnnotation,
    OWLAnnotationAssertionAxiom,
    OWLAnnotationProperty,
    OWLClassAssertionAxiom,
    OWLDeclarationAxiom,
    OWLObjectPropertyAssertionAxiom,
)
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_property import OWLObjectProperty

from annotation_layer.annotation_layer import AnnotationLayer


RDFS_LABEL_IRI = "http://www.w3.org/2000/01/rdf-schema#label"


__all__ = [
    "populate_abox",
]


def populate_abox(
    onto: SyncOntology,
    annotation_layer: AnnotationLayer,
    *,
    abox_base_iri: str,
    print_result_summary: bool = False,
) -> None:
    entities = annotation_layer.require_entities()
    relations = annotation_layer.relations

    entity_ids = _entity_ids(entities)
    entity_names = _collect_unique_entity_names(entities, entity_ids)
    individuals = _build_individuals(entity_names, abox_base_iri)

    entity_axioms = _build_entity_axioms(
        entities=entities,
        entity_ids=entity_ids,
        entity_names=entity_names,
        individuals=individuals,
    )
    relation_axioms = _build_relation_axioms(
        relations=relations,
        individuals=individuals,
    )

    print("DEBUG DEBUG DEBUG ")
    for entity_axiom in entity_axioms:
        print(entity_axiom)
    print("DEBUG DEBUG DEBUG ")

    axioms = entity_axioms + relation_axioms

    if axioms:
        onto.add_axiom(axioms)

    if print_result_summary:
        _print_population_summary(
            entity_count=len(entity_ids),
            typed_entity_count=_count_typed_entities(entities, entity_ids),
            relation_count=len(relation_axioms),
            axiom_count=len(axioms),
        )


def _entity_ids(entities) -> list[int]:
    return [int(entity_id) for entity_id in entities.cluster_ids()]


def _collect_unique_entity_names(
    entities,
    entity_ids: list[int],
) -> dict[int, str]:
    names_by_entity_id: dict[int, str] = {}
    seen_names: dict[str, int] = {}

    for entity_id in entity_ids:
        canonical_name = entities.cluster(entity_id).canonical_name.strip()

        if not canonical_name:
            raise ValueError(f"Entity {entity_id} has no canonical_name.")

        normalized_name = canonical_name.casefold()

        if normalized_name in seen_names:
            previous_entity_id = seen_names[normalized_name]
            raise ValueError(
                "Duplicate canonical_name found while populating ABox: "
                f"{canonical_name!r} is used by both entity "
                f"{previous_entity_id} and entity {entity_id}."
            )

        seen_names[normalized_name] = entity_id
        names_by_entity_id[entity_id] = canonical_name

    return names_by_entity_id


def _build_individuals(
    entity_names: dict[int, str],
    abox_base_iri: str,
) -> dict[int, OWLNamedIndividual]:
    base_iri = _normalized_base_iri(abox_base_iri)

    return {
        entity_id: OWLNamedIndividual(
            IRI(base_iri, _iri_local_name(canonical_name))
        )
        for entity_id, canonical_name in entity_names.items()
    }


def _build_entity_axioms(
    *,
    entities,
    entity_ids: list[int],
    entity_names: dict[int, str],
    individuals: dict[int, OWLNamedIndividual],
) -> list:
    axioms = []

    for entity_id in entity_ids:
        entity = entities.cluster(entity_id)
        individual = individuals[entity_id]
        canonical_name = entity_names[entity_id]

        axioms.append(OWLDeclarationAxiom(individual))
        axioms.append(_label_axiom(individual, canonical_name))

        if entity.typing is None:
            continue

        class_iri = str(entity.typing.class_iri).strip()
        if not class_iri:
            raise ValueError(f"Entity {entity_id} has an empty class IRI.")

        axioms.append(
            OWLClassAssertionAxiom(
                individual,
                OWLClass(IRI.create(class_iri)),
            )
        )

    return axioms


def _build_relation_axioms(
    *,
    relations,
    individuals: dict[int, OWLNamedIndividual],
) -> list:
    if relations is None:
        return []

    axioms = []

    for relation in relations.all_assertions():
        source_id = int(relation.source_cluster_id)
        target_id = int(relation.target_cluster_id)

        if source_id not in individuals:
            raise ValueError(
                f"Relation {relation.assertion_id!r} references missing "
                f"source entity {source_id}."
            )

        if target_id not in individuals:
            raise ValueError(
                f"Relation {relation.assertion_id!r} references missing "
                f"target entity {target_id}."
            )

        property_iri = str(relation.object_property_iri).strip()
        if not property_iri:
            raise ValueError(
                f"Relation {relation.assertion_id!r} has an empty object_property_iri."
            )

        axioms.append(
            OWLObjectPropertyAssertionAxiom(
                individuals[source_id],
                OWLObjectProperty(IRI.create(property_iri)),
                individuals[target_id],
            )
        )

    return axioms


def _label_axiom(
    individual: OWLNamedIndividual,
    label: str,
) -> OWLAnnotationAssertionAxiom:
    return OWLAnnotationAssertionAxiom(
        individual.iri,
        OWLAnnotation(
            OWLAnnotationProperty(IRI.create(RDFS_LABEL_IRI)),
            OWLLiteral(label),
        ),
    )


def _count_typed_entities(entities, entity_ids: list[int]) -> int:
    return sum(
        entities.cluster(entity_id).typing is not None
        for entity_id in entity_ids
    )


def _print_population_summary(
    *,
    entity_count: int,
    typed_entity_count: int,
    relation_count: int,
    axiom_count: int,
) -> None:
    print("[ABox population]")
    print(f"  entities:             {entity_count}")
    print(f"  typed entities:       {typed_entity_count}")
    print(f"  untyped entities:     {entity_count - typed_entity_count}")
    print(f"  relation assertions:  {relation_count}")
    print(f"  axioms added:         {axiom_count}")


def _iri_local_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_]", "", text)

    if not text:
        raise ValueError(f"Cannot build an IRI local name from {name!r}.")

    if text[0].isdigit():
        text = f"entity_{text}"

    return text


def _normalized_base_iri(abox_base_iri: str) -> str:
    return (
        abox_base_iri
        if abox_base_iri.endswith(("#", "/", ":"))
        else abox_base_iri + "#"
    )
