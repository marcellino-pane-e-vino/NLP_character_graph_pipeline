"""Ontology package public exports."""

from ontology.ontology_management import (
    NarrativeOntology,
    RelationPropertySpec,
    RelationRouter,
    human_label,
    local_name,
)
from ontology.data_properties import (
    DataPropertyFieldSpec,
    OCEAN_DATA_PROPERTY_DATATYPE_IRI,
    OCEAN_DATA_PROPERTY_LOCAL_NAMES,
    OCEAN_TRAITS,
    OCEAN_TRAIT_TO_DATA_PROPERTY_LOCAL_NAME,
    XSD_DECIMAL_IRI,
    XSD_DOUBLE_IRI,
    XSD_FLOAT_IRI,
    add_data_property_field_to_class,
    add_data_property_fields_to_class,
    add_ocean_data_fields_to_class,
    ocean_data_field_specs_for_class_iri,
    ocean_data_property_iri_for_trait,
    ocean_data_property_iris_for_class_iri,
    ocean_data_property_local_name_for_trait,
)


__all__ = [
    "NarrativeOntology",
    "RelationPropertySpec",
    "RelationRouter",
    "human_label",
    "local_name",
    "DataPropertyFieldSpec",
    "OCEAN_DATA_PROPERTY_DATATYPE_IRI",
    "OCEAN_DATA_PROPERTY_LOCAL_NAMES",
    "OCEAN_TRAITS",
    "OCEAN_TRAIT_TO_DATA_PROPERTY_LOCAL_NAME",
    "XSD_DECIMAL_IRI",
    "XSD_DOUBLE_IRI",
    "XSD_FLOAT_IRI",
    "add_data_property_field_to_class",
    "add_data_property_fields_to_class",
    "add_ocean_data_fields_to_class",
    "ocean_data_field_specs_for_class_iri",
    "ocean_data_property_iri_for_trait",
    "ocean_data_property_iris_for_class_iri",
    "ocean_data_property_local_name_for_trait",
]
