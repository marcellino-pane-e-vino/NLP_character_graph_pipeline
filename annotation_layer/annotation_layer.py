from __future__ import annotations

from dataclasses import dataclass, field

from annotation_layer.entities import EntitySubLayer
from annotation_layer.relations import RelationSubLayer


@dataclass(slots=True)
class ArtifactManifest:
    tokenized_doc_path: str | None = None
    entity_annotated_doc_path: str | None = None
    typed_entity_doc_path: str | None = None
    ocean_entity_doc_path: str | None = None
    relation_annotated_doc_path: str | None = None

    entity_artifact_path: str | None = None
    cluster_typing_artifact_root: str | None = None
    ocean_artifact_root: str | None = None
    relation_artifact_root: str | None = None
    ontology_path: str | None = None
    populated_ontology_path: str | None = None


@dataclass(slots=True)
class AnnotationStatus:
    has_entities: bool = False
    has_cluster_typing: bool = False
    has_ocean: bool = False
    has_relations: bool = False


@dataclass(slots=True)
class AnnotationLayer:
    document_id: str
    entities: EntitySubLayer | None = None
    relations: RelationSubLayer | None = None
    artifacts: ArtifactManifest = field(default_factory=ArtifactManifest)
    status: AnnotationStatus = field(default_factory=AnnotationStatus)

    def attach_entities(self, layer: EntitySubLayer, *, overwrite: bool = False) -> None:
        if self.entities is not None and not overwrite:
            raise ValueError("AnnotationLayer already has entities.")
        self.entities = layer
        self.status.has_entities = True

    def attach_relations(self, layer: RelationSubLayer, *, overwrite: bool = False) -> None:
        if self.relations is not None and not overwrite:
            raise ValueError("AnnotationLayer already has relations.")
        self.relations = layer
        self.status.has_relations = True

    def require_entities(self) -> EntitySubLayer:
        if self.entities is None:
            raise ValueError("AnnotationLayer has no entities.")
        return self.entities

    def require_relations(self) -> RelationSubLayer:
        if self.relations is None:
            raise ValueError("AnnotationLayer has no relations.")
        return self.relations

    def mark_cluster_typing_complete(self) -> None:
        self.status.has_cluster_typing = True

    def mark_ocean_complete(self) -> None:
        self.status.has_ocean = True

    def summary(self) -> dict[str, int | bool]:
        out: dict[str, int | bool] = {
            "has_entities": self.status.has_entities,
            "has_cluster_typing": self.status.has_cluster_typing,
            "has_ocean": self.status.has_ocean,
            "has_relations": self.status.has_relations,
        }
        if self.entities is not None:
            out.update(self.entities.summary())
        if self.relations is not None:
            out.update(self.relations.summary())
        return out
