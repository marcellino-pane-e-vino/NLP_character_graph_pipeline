from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from typing import Iterable

try:
    from spacy.tokens import Doc, Span
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "relation_schema.py requires spaCy. Install it with: pip install spacy"
    ) from exc

from coreference.coref_schema import (
    Mention,
    Cluster,
    CorefLayer,
    require_coref_layer,
)


__all__ = [
    "Mention",
    "Cluster",
    "CorefLayer",
    "require_coref_layer",
    "RelationMention",
    "RelationAssignment",
    "ClusterAssertion",
    "RelationLayer",
    "softmax_dict",
    "make_relation_mention_id",
    "make_cluster_assertion_id",
    "register_spacy_relation_extension",
    "require_relation_layer",
]


@dataclass(frozen=True, slots=True)
class RelationMention:
    """One source-predicate-target textual relation anchor.

    This object stores only primary text anchors. Cluster/canonical-name data are
    reached through the imported coreference Mention objects and CorefLayer.
    """

    relation_mention_id: str

    source_mention: Mention

    predicate_token_i: int
    predicate_start: int
    predicate_end: int

    target_mention: Mention


@dataclass(frozen=True, slots=True)
class RelationAssignment:
    relation_mention: RelationMention
    object_property_logits: dict[str, float]
    selection_method: str


@dataclass(frozen=True, slots=True)
class ClusterAssertion:
    """Aggregated cluster-level relation assertion for ontology population."""

    cluster_assertion_id: str

    source_cluster_id: int
    object_property_iri: str
    target_cluster_id: int

    support_assignment_ids: tuple[str, ...]
    aggregation_method: str


@dataclass(slots=True)
class RelationLayer:
    """Single relation wrapper stored at doc._.relation_layer.

    The layer contains assignment-level evidence and cluster-level assertions.
    It also owns secondary indexes and convenience helpers, mirroring the style
    of CorefLayer.
    """

    assignments: dict[str, RelationAssignment]
    cluster_assertions: dict[str, ClusterAssertion]

    predicate_token_to_assignment_ids: dict[int, list[str]] = field(default_factory=dict)
    source_cluster_to_assignment_ids: dict[int, list[str]] = field(default_factory=dict)
    target_cluster_to_assignment_ids: dict[int, list[str]] = field(default_factory=dict)

    source_cluster_to_assertion_ids: dict[int, list[str]] = field(default_factory=dict)
    target_cluster_to_assertion_ids: dict[int, list[str]] = field(default_factory=dict)
    object_property_to_assertion_ids: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls,
        *,
        assignments: dict[str, RelationAssignment],
        cluster_assertions: dict[str, ClusterAssertion],
    ) -> "RelationLayer":
        layer = cls(assignments=assignments, cluster_assertions=cluster_assertions)
        layer.rebuild_indexes()
        return layer

    def rebuild_indexes(self) -> None:
        """Rebuild all derived indexes from primary assignments/assertions."""

        self.predicate_token_to_assignment_ids.clear()
        self.source_cluster_to_assignment_ids.clear()
        self.target_cluster_to_assignment_ids.clear()
        self.source_cluster_to_assertion_ids.clear()
        self.target_cluster_to_assertion_ids.clear()
        self.object_property_to_assertion_ids.clear()

        for assignment_id, assignment in self.assignments.items():
            mention = assignment.relation_mention

            self.predicate_token_to_assignment_ids.setdefault(
                mention.predicate_token_i,
                [],
            ).append(assignment_id)

            self.source_cluster_to_assignment_ids.setdefault(
                int(mention.source_mention.cluster_id),
                [],
            ).append(assignment_id)

            self.target_cluster_to_assignment_ids.setdefault(
                int(mention.target_mention.cluster_id),
                [],
            ).append(assignment_id)

        for assertion_id, assertion in self.cluster_assertions.items():
            self.source_cluster_to_assertion_ids.setdefault(
                int(assertion.source_cluster_id),
                [],
            ).append(assertion_id)

            self.target_cluster_to_assertion_ids.setdefault(
                int(assertion.target_cluster_id),
                [],
            ).append(assertion_id)

            self.object_property_to_assertion_ids.setdefault(
                assertion.object_property_iri,
                [],
            ).append(assertion_id)

    # ----- RelationMention access -----

    def relation_mention(self, assignment_id: str) -> RelationMention:
        return self.assignments[assignment_id].relation_mention

    def source_mention(self, assignment_id: str) -> Mention:
        return self.relation_mention(assignment_id).source_mention

    def target_mention(self, assignment_id: str) -> Mention:
        return self.relation_mention(assignment_id).target_mention

    def source_cluster(self, coref: CorefLayer, assignment_id: str) -> Cluster:
        mention = self.source_mention(assignment_id)
        return coref.clusters[int(mention.cluster_id)]

    def target_cluster(self, coref: CorefLayer, assignment_id: str) -> Cluster:
        mention = self.target_mention(assignment_id)
        return coref.clusters[int(mention.cluster_id)]

    # ----- Text spans -----

    def source_span(self, doc: Doc, assignment_id: str) -> Span:
        mention = self.source_mention(assignment_id)
        return doc[int(mention.start) : int(mention.end)]

    def target_span(self, doc: Doc, assignment_id: str) -> Span:
        mention = self.target_mention(assignment_id)
        return doc[int(mention.start) : int(mention.end)]

    def predicate_span(self, doc: Doc, assignment_id: str) -> Span:
        mention = self.relation_mention(assignment_id)
        return doc[int(mention.predicate_start) : int(mention.predicate_end)]

    def evidence_span(self, doc: Doc, assignment_id: str) -> Span:
        """Return the minimal span covering source, predicate, and target."""

        mention = self.relation_mention(assignment_id)
        start = min(
            int(mention.source_mention.start),
            int(mention.predicate_start),
            int(mention.target_mention.start),
        )
        end = max(
            int(mention.source_mention.end),
            int(mention.predicate_end),
            int(mention.target_mention.end),
        )
        return doc[start:end]

    # ----- Neural score helpers -----

    def object_property_scores(self, assignment_id: str) -> dict[str, float]:
        return softmax_dict(self.assignments[assignment_id].object_property_logits)

    def chosen_object_property_iri(self, assignment_id: str) -> str:
        scores = self.object_property_scores(assignment_id)
        if not scores:
            raise ValueError(f"Assignment {assignment_id!r} has no object-property scores.")
        return max(scores, key=scores.get)

    def confidence(self, assignment_id: str) -> float:
        scores = self.object_property_scores(assignment_id)
        return max(scores.values()) if scores else 0.0

    def margin(self, assignment_id: str) -> float:
        values = sorted(self.object_property_scores(assignment_id).values(), reverse=True)
        if not values:
            return 0.0
        if len(values) == 1:
            return 1.0
        return values[0] - values[1]

    def entropy(self, assignment_id: str) -> float:
        scores = self.object_property_scores(assignment_id)
        return -sum(p * log(p) for p in scores.values() if p > 0.0)

    # ----- Indexed retrieval -----

    def assignments_from_predicate_token(self, token_i: int) -> list[RelationAssignment]:
        return [
            self.assignments[assignment_id]
            for assignment_id in self.predicate_token_to_assignment_ids.get(token_i, [])
            if assignment_id in self.assignments
        ]

    def assignments_from_source_cluster(self, cluster_id: int) -> list[RelationAssignment]:
        return [
            self.assignments[assignment_id]
            for assignment_id in self.source_cluster_to_assignment_ids.get(cluster_id, [])
            if assignment_id in self.assignments
        ]

    def assignments_from_target_cluster(self, cluster_id: int) -> list[RelationAssignment]:
        return [
            self.assignments[assignment_id]
            for assignment_id in self.target_cluster_to_assignment_ids.get(cluster_id, [])
            if assignment_id in self.assignments
        ]

    def assertions_from_source_cluster(self, cluster_id: int) -> list[ClusterAssertion]:
        return [
            self.cluster_assertions[assertion_id]
            for assertion_id in self.source_cluster_to_assertion_ids.get(cluster_id, [])
            if assertion_id in self.cluster_assertions
        ]

    def assertions_from_target_cluster(self, cluster_id: int) -> list[ClusterAssertion]:
        return [
            self.cluster_assertions[assertion_id]
            for assertion_id in self.target_cluster_to_assertion_ids.get(cluster_id, [])
            if assertion_id in self.cluster_assertions
        ]

    def assertions_for_object_property(self, object_property_iri: str) -> list[ClusterAssertion]:
        return [
            self.cluster_assertions[assertion_id]
            for assertion_id in self.object_property_to_assertion_ids.get(object_property_iri, [])
            if assertion_id in self.cluster_assertions
        ]

    # ----- Assertion helpers -----

    def assertion_assignments(self, assertion_id: str) -> list[RelationAssignment]:
        assertion = self.cluster_assertions[assertion_id]
        return [
            self.assignments[assignment_id]
            for assignment_id in assertion.support_assignment_ids
            if assignment_id in self.assignments
        ]

    def support_count(self, assertion_id: str) -> int:
        return len(self.cluster_assertions[assertion_id].support_assignment_ids)

    def mean_assertion_confidence(self, assertion_id: str) -> float:
        values = [
            self.confidence(assignment_id)
            for assignment_id in self.cluster_assertions[assertion_id].support_assignment_ids
            if assignment_id in self.assignments
        ]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, int]:
        return {
            "n_relation_assignments": len(self.assignments),
            "n_cluster_assertions": len(self.cluster_assertions),
            "n_indexed_predicate_tokens": len(self.predicate_token_to_assignment_ids),
            "n_indexed_source_clusters": len(self.source_cluster_to_assignment_ids),
            "n_indexed_target_clusters": len(self.target_cluster_to_assignment_ids),
            "n_indexed_assertion_source_clusters": len(self.source_cluster_to_assertion_ids),
            "n_indexed_assertion_target_clusters": len(self.target_cluster_to_assertion_ids),
            "n_indexed_object_properties": len(self.object_property_to_assertion_ids),
        }


def softmax_dict(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}

    max_logit = max(float(value) for value in logits.values())
    exps = {key: exp(float(value) - max_logit) for key, value in logits.items()}
    denominator = sum(exps.values())

    if denominator == 0.0:
        return {key: 0.0 for key in logits}

    return {key: value / denominator for key, value in exps.items()}


def make_relation_mention_id(
    *,
    predicate_token_i: int,
    source_mention_id: int,
    target_mention_id: int,
) -> str:
    return f"pred_{predicate_token_i}_srcm_{source_mention_id}_tgtm_{target_mention_id}"


def make_cluster_assertion_id(
    *,
    source_cluster_id: int,
    object_property_iri: str,
    target_cluster_id: int,
) -> str:
    prop_local = object_property_iri.rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    prop_local = "".join(ch if ch.isalnum() else "_" for ch in prop_local).strip("_")
    return f"src_{source_cluster_id}_prop_{prop_local}_tgt_{target_cluster_id}"


def register_spacy_relation_extension(*, force: bool = False) -> None:
    """Register doc._.relation_layer as the only spaCy relation extension."""

    if Doc.has_extension("relation_layer"):
        if force:
            Doc.set_extension("relation_layer", default=None, force=True)
        return

    Doc.set_extension("relation_layer", default=None)


def require_relation_layer(doc: Doc) -> RelationLayer:
    """Return doc._.relation_layer or fail early when absent."""

    if not Doc.has_extension("relation_layer") or doc._.relation_layer is None:
        raise ValueError("This Doc has no relation layer.")
    return doc._.relation_layer
