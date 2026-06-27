from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

from annotation_layer.ids import (
    ClusterId,
    MentionId,
    ObjectPropertyIri,
    RelationAssertionId,
    RelationId,
)


@dataclass(frozen=True, slots=True)
class RelationInstanceRecord:
    relation_id: RelationId
    source_mention_id: MentionId
    predicate_token_i: int
    predicate_start: int
    predicate_end: int
    target_mention_id: MentionId
    sentence_index: int | None = None


@dataclass(frozen=True, slots=True)
class RelationAssignmentRecord:
    relation_id: RelationId
    object_property_logits: dict[ObjectPropertyIri, float]
    selection_method: str

    def object_property_scores(self) -> dict[ObjectPropertyIri, float]:
        return softmax_dict(self.object_property_logits)

    def chosen_object_property_iri(self) -> ObjectPropertyIri:
        scores = self.object_property_scores()
        if not scores:
            raise ValueError(f"Relation assignment {self.relation_id!r} has no scores.")
        return max(scores, key=scores.get)

    def confidence(self) -> float:
        scores = self.object_property_scores()
        return max(scores.values()) if scores else 0.0

    def margin(self) -> float:
        values = sorted(self.object_property_scores().values(), reverse=True)
        if not values:
            return 0.0
        if len(values) == 1:
            return 1.0
        return float(values[0] - values[1])

    def entropy(self) -> float:
        return -sum(p * log(p) for p in self.object_property_scores().values() if p > 0.0)


@dataclass(frozen=True, slots=True)
class RelationAssertionRecord:
    assertion_id: RelationAssertionId
    source_cluster_id: ClusterId
    object_property_iri: ObjectPropertyIri
    target_cluster_id: ClusterId
    support_relation_ids: tuple[RelationId, ...]
    aggregation_method: str
    confidence: float | None = None


@dataclass(slots=True)
class RelationSubLayer:
    instances: dict[RelationId, RelationInstanceRecord]
    assignments: dict[RelationId, RelationAssignmentRecord]
    assertions: dict[RelationAssertionId, RelationAssertionRecord]

    by_source_cluster: dict[ClusterId, tuple[RelationAssertionId, ...]] = field(default_factory=dict)
    by_target_cluster: dict[ClusterId, tuple[RelationAssertionId, ...]] = field(default_factory=dict)
    by_property: dict[ObjectPropertyIri, tuple[RelationAssertionId, ...]] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        *,
        instances: dict[RelationId, RelationInstanceRecord],
        assignments: dict[RelationId, RelationAssignmentRecord],
        assertions: dict[RelationAssertionId, RelationAssertionRecord],
    ) -> "RelationSubLayer":
        layer = cls(instances=instances, assignments=assignments, assertions=assertions)
        layer.rebuild_indexes()
        return layer

    def instance(self, relation_id: RelationId | str) -> RelationInstanceRecord:
        return self.instances[str(relation_id)]

    def assignment(self, relation_id: RelationId | str) -> RelationAssignmentRecord | None:
        return self.assignments.get(str(relation_id))

    def assertion(self, assertion_id: RelationAssertionId | str) -> RelationAssertionRecord:
        return self.assertions[str(assertion_id)]

    def relation_mention(self, relation_id: RelationId | str) -> RelationInstanceRecord:
        return self.instance(relation_id)

    def relation_assignment(self, relation_id: RelationId | str) -> RelationAssignmentRecord | None:
        return self.assignment(relation_id)

    def cluster_assertion(self, assertion_id: RelationAssertionId | str) -> RelationAssertionRecord:
        return self.assertion(assertion_id)

    def all_assertions(self) -> tuple[RelationAssertionRecord, ...]:
        return tuple(self.assertions.values())

    def assertions_from_source(self, cluster_id: ClusterId | int) -> tuple[RelationAssertionRecord, ...]:
        ids = self.by_source_cluster.get(int(cluster_id), ())
        return tuple(self.assertion(assertion_id) for assertion_id in ids)

    def assertions_to_target(self, cluster_id: ClusterId | int) -> tuple[RelationAssertionRecord, ...]:
        ids = self.by_target_cluster.get(int(cluster_id), ())
        return tuple(self.assertion(assertion_id) for assertion_id in ids)

    def assertions_with_property(
        self,
        property_iri: ObjectPropertyIri | str,
    ) -> tuple[RelationAssertionRecord, ...]:
        ids = self.by_property.get(str(property_iri), ())
        return tuple(self.assertion(assertion_id) for assertion_id in ids)

    def rebuild_indexes(self) -> None:
        by_source: dict[int, list[str]] = {}
        by_target: dict[int, list[str]] = {}
        by_property: dict[str, list[str]] = {}

        for assertion_id, assertion in self.assertions.items():
            by_source.setdefault(int(assertion.source_cluster_id), []).append(assertion_id)
            by_target.setdefault(int(assertion.target_cluster_id), []).append(assertion_id)
            by_property.setdefault(str(assertion.object_property_iri), []).append(assertion_id)

        self.by_source_cluster = {key: tuple(value) for key, value in by_source.items()}
        self.by_target_cluster = {key: tuple(value) for key, value in by_target.items()}
        self.by_property = {key: tuple(value) for key, value in by_property.items()}

    def summary(self) -> dict[str, int]:
        return {
            "n_relation_instances": len(self.instances),
            "n_relation_assignments": len(self.assignments),
            "n_relation_assertions": len(self.assertions),
            "n_source_cluster_indexes": len(self.by_source_cluster),
            "n_target_cluster_indexes": len(self.by_target_cluster),
            "n_property_indexes": len(self.by_property),
        }


def softmax_dict(logits: dict[ObjectPropertyIri, float]) -> dict[ObjectPropertyIri, float]:
    if not logits:
        return {}

    max_logit = max(float(value) for value in logits.values())
    exps = {key: exp(float(value) - max_logit) for key, value in logits.items()}
    denominator = sum(exps.values())
    if denominator == 0.0:
        return {key: 0.0 for key in logits}
    return {key: value / denominator for key, value in exps.items()}


def make_relation_id(source_mention_id: int, predicate_token_i: int, target_mention_id: int) -> str:
    return f"rel_{int(source_mention_id)}_{int(predicate_token_i)}_{int(target_mention_id)}"


def make_relation_assertion_id(source_cluster_id: int, object_property_iri: str, target_cluster_id: int) -> str:
    safe_property = str(object_property_iri).rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return f"assertion_{int(source_cluster_id)}_{safe_property}_{int(target_cluster_id)}"
