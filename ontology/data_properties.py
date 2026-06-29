"""Datatype-property field helpers for ontology-level symbolic contracts.

This module adds OWL data-property restrictions to named classes.  In project
terms, these restrictions behave like schema-level "fields": they say that
instances of a class must carry a literal value for a data property.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from owlapy.class_expression import OWLClass, OWLDataExactCardinality, OWLDataSomeValuesFrom
from owlapy.iri import IRI
from owlapy.owl_axiom import (
    OWLDataPropertyDomainAxiom,
    OWLDataPropertyRangeAxiom,
    OWLDeclarationAxiom,
    OWLFunctionalDataPropertyAxiom,
    OWLSubClassOfAxiom,
)
from owlapy.owl_datatype import OWLDatatype
from owlapy.owl_ontology import SyncOntology
from owlapy.owl_property import OWLDataProperty


XSD_FLOAT_IRI = "http://www.w3.org/2001/XMLSchema#float"
XSD_DOUBLE_IRI = "http://www.w3.org/2001/XMLSchema#double"
XSD_DECIMAL_IRI = "http://www.w3.org/2001/XMLSchema#decimal"

# OWLAPY's OWLAPI mapper reliably maps native Python float literals as
# xsd:double.  Keep the OCEAN field contract aligned with that runtime
# representation.
OCEAN_DATA_PROPERTY_DATATYPE_IRI = XSD_DOUBLE_IRI

OCEAN_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

OCEAN_TRAIT_TO_DATA_PROPERTY_LOCAL_NAME: Mapping[str, str] = MappingProxyType(
    {
        "openness": "hasOceanOpenness",
        "conscientiousness": "hasOceanConscientiousness",
        "extraversion": "hasOceanExtraversion",
        "agreeableness": "hasOceanAgreeableness",
        "neuroticism": "hasOceanNeuroticism",
    }
)

OCEAN_DATA_PROPERTY_LOCAL_NAMES: tuple[str, ...] = tuple(
    OCEAN_TRAIT_TO_DATA_PROPERTY_LOCAL_NAME[trait]
    for trait in OCEAN_TRAITS
)


@dataclass(frozen=True, slots=True)
class DataPropertyFieldSpec:
    """Specification for one datatype-backed class field.

    ``exact_cardinality=1`` models a Java-like mandatory single-valued field:
    ``Class ⊑ (= 1 dataProperty datatype)``.

    ``exact_cardinality=None`` models only existence:
    ``Class ⊑ ∃ dataProperty.datatype``.
    """

    property_iri: str
    datatype_iri: str = XSD_FLOAT_IRI
    exact_cardinality: int | None = 1


def iri_namespace(absolute_iri: str) -> str:
    """Return the namespace part of an absolute hash/slash IRI."""

    iri = str(absolute_iri).strip()
    if not iri:
        raise ValueError("IRI must be non-empty.")

    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"

    if "/" in iri:
        return iri.rsplit("/", 1)[0] + "/"

    raise ValueError(f"Cannot infer namespace from IRI: {absolute_iri!r}")


def owl_class(class_iri: str) -> OWLClass:
    """Build an OWLAPY class from an absolute IRI string."""

    return OWLClass(IRI.create(str(class_iri).strip()))


def owl_data_property(property_iri: str) -> OWLDataProperty:
    """Build an OWLAPY data property from an absolute IRI string."""

    return OWLDataProperty(IRI.create(str(property_iri).strip()))


def owl_datatype(datatype_iri: str) -> OWLDatatype:
    """Build an OWLAPY datatype from an absolute datatype IRI string."""

    return OWLDatatype(IRI.create(str(datatype_iri).strip()))


def ocean_data_property_local_name_for_trait(trait: str) -> str:
    """Return the canonical OCEAN data-property local name for one trait."""

    trait_key = str(trait).strip().casefold()
    try:
        return OCEAN_TRAIT_TO_DATA_PROPERTY_LOCAL_NAME[trait_key]
    except KeyError as exc:
        raise KeyError(f"Unknown OCEAN trait: {trait!r}") from exc


def ocean_data_property_iri_for_trait(
    *,
    class_iri: str,
    trait: str,
) -> str:
    """Return the canonical OCEAN data-property IRI for one trait.

    OCEAN fields are ontology-local schema fields.  The class IRI is used only
    to infer the ontology namespace; callers do not need to know or duplicate
    the namespace construction policy.
    """

    return f"{iri_namespace(class_iri)}{ocean_data_property_local_name_for_trait(trait)}"


def ocean_data_property_iris_for_class_iri(class_iri: str) -> frozenset[str]:
    """Return the canonical OCEAN data-property IRIs for a class namespace."""

    return frozenset(
        ocean_data_property_iri_for_trait(class_iri=class_iri, trait=trait)
        for trait in OCEAN_TRAITS
    )


def ocean_data_field_specs_for_class_iri(class_iri: str) -> tuple[DataPropertyFieldSpec, ...]:
    """Return the five OCEAN field specs in the same namespace as ``class_iri``."""

    return tuple(
        DataPropertyFieldSpec(
            property_iri=ocean_data_property_iri_for_trait(
                class_iri=class_iri,
                trait=trait,
            ),
            datatype_iri=OCEAN_DATA_PROPERTY_DATATYPE_IRI,
        )
        for trait in OCEAN_TRAITS
    )


def add_data_property_field_to_class(
    onto: SyncOntology,
    *,
    class_iri: str,
    field: DataPropertyFieldSpec,
    add_domain_range: bool = True,
    add_functional_axiom: bool = False,
) -> OWLDataProperty:
    """Add one datatype-backed field restriction to one class.

    Domain/range axioms are useful metadata, but they are not what makes the
    field mandatory.  Mandatory-ness comes from the generated subclass
    restriction:

    - ``exact_cardinality=1`` -> ``Class ⊑ (= 1 property datatype)``
    - ``exact_cardinality=None`` -> ``Class ⊑ ∃ property.datatype``
    """

    cls = owl_class(class_iri)
    prop = owl_data_property(field.property_iri)
    datatype = owl_datatype(field.datatype_iri)

    if field.exact_cardinality is None:
        restriction = OWLDataSomeValuesFrom(prop, datatype)
    else:
        if field.exact_cardinality < 0:
            raise ValueError("exact_cardinality must be a non-negative integer or None.")
        restriction = OWLDataExactCardinality(field.exact_cardinality, prop, datatype)

    axioms = [
        OWLDeclarationAxiom(prop),
        OWLSubClassOfAxiom(cls, restriction),
    ]

    if add_domain_range:
        axioms.extend(
            [
                OWLDataPropertyDomainAxiom(prop, cls),
                OWLDataPropertyRangeAxiom(prop, datatype),
            ]
        )

    if add_functional_axiom:
        axioms.append(OWLFunctionalDataPropertyAxiom(prop))

    onto.add_axiom(axioms)
    return prop


def add_data_property_fields_to_class(
    onto: SyncOntology,
    *,
    class_iri: str,
    fields: Iterable[DataPropertyFieldSpec],
    add_domain_range: bool = True,
    add_functional_axiom: bool = False,
) -> tuple[OWLDataProperty, ...]:
    """Add multiple datatype-backed field restrictions to one class."""

    return tuple(
        add_data_property_field_to_class(
            onto,
            class_iri=class_iri,
            field=field,
            add_domain_range=add_domain_range,
            add_functional_axiom=add_functional_axiom,
        )
        for field in fields
    )


def add_ocean_data_fields_to_class(
    onto: SyncOntology,
    *,
    class_iri: str,
    add_domain_range: bool = True,
    add_functional_axiom: bool = False,
) -> tuple[OWLDataProperty, ...]:
    """Add the five OCEAN float fields to one ontology class."""

    return add_data_property_fields_to_class(
        onto,
        class_iri=class_iri,
        fields=ocean_data_field_specs_for_class_iri(class_iri),
        add_domain_range=add_domain_range,
        add_functional_axiom=add_functional_axiom,
    )
