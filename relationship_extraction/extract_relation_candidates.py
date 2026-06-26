from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from coreference.coref_schema import require_coref_layer
from relationship_extraction.relation_schema import make_relation_mention_id

try:  # project-local dependency
    from cluster_typing.cluster_typing_schema import require_cluster_typing_layer
except ImportError:  # pragma: no cover
    require_cluster_typing_layer = None


SUBJECT_DEPS = {
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
}

PASSIVE_SUBJECT_DEPS = {
    "nsubjpass",
    "csubjpass",
}

OBJECT_DEPS = {
    "dobj",      # spaCy English models often use this
    "obj",       # Universal Dependencies-style label
    "iobj",
    "dative",
    "attr",      # copular / attributive complement
    "oprd",
}

PREP_DEPS = {
    "prep",
}

PREP_OBJECT_DEPS = {
    "pobj",
    "pcomp",
}

AUX_PASSIVE_DEPS = {
    "auxpass",
}

NEGATION_DEPS = {
    "neg",
}

PARTICLE_DEPS = {
    "prt",
}


@dataclass(frozen=True, slots=True)
class SyntaxRelationCandidate:
    """Raw dependency/coref relation candidate before ontology routing."""

    relation_mention_id: str

    source_cluster_id: int
    source_canonical_name: str
    source_class_iri: str

    predicate: str
    predicate_surface: str
    predicate_token_i: int
    predicate_start: int
    predicate_end: int

    target_cluster_id: int
    target_canonical_name: str
    target_class_iri: str

    sentence_index: int
    sentence_start: int
    sentence_end: int
    sentence_text: str

    source_mention_id: int
    target_mention_id: int

    source_token_i: int
    target_token_i: int

    object_dependency: str
    preposition: str | None

    is_passive: bool
    is_negated: bool


@dataclass(frozen=True, slots=True)
class RoutedRelationCandidate:
    """Stage-1 row serialized to routed_relation_candidates.jsonl."""

    relation_mention_id: str

    source_mention_id: int
    predicate_token_i: int
    predicate_start: int
    predicate_end: int
    target_mention_id: int

    source_cluster_id: int
    source_canonical_name: str
    source_class_iri: str

    predicate: str
    predicate_surface: str

    target_cluster_id: int
    target_canonical_name: str
    target_class_iri: str

    sentence_index: int
    sentence_start: int
    sentence_end: int
    sentence_text: str

    premise_text: str
    candidate_properties: list[dict[str, Any]]

    source_token_i: int
    target_token_i: int
    object_dependency: str
    preposition: str | None
    is_passive: bool
    is_negated: bool


def _ensure_dependency_parse(doc: Any) -> None:
    if not doc.has_annotation("DEP"):
        raise ValueError(
            "This Doc has no dependency parse. "
            "You need a spaCy pipeline with the parser enabled."
        )

    if not doc.has_annotation("POS"):
        raise ValueError(
            "This Doc has no POS annotation. "
            "You need POS tags to reliably identify verbal predicates."
        )


def _require_cluster_typing_layer(doc: Any, cluster_typing_layer: Any | None) -> Any:
    if cluster_typing_layer is not None:
        return cluster_typing_layer

    if require_cluster_typing_layer is None:
        raise ValueError(
            "cluster_typing_layer was not provided and "
            "cluster_typing.cluster_typing_schema could not be imported."
        )

    return require_cluster_typing_layer(doc)


def _token_mentions(coref: Any, token: Any) -> list[Any]:
    """Resolve a syntactic token to coreference mentions.

    Prefer exact syntactic-head match when available. Fall back to mentions
    covering the token.
    """

    head_mentions = coref.mentions_from_head_token(token.i)
    if head_mentions:
        return head_mentions

    return coref.mentions_from_token(token.i)


def _cluster_records_for_token(coref: Any, token: Any) -> list[tuple[int, int]]:
    """Return deterministic (mention_id, cluster_id) pairs for a token."""

    pairs: list[tuple[int, int]] = []

    for mention in _token_mentions(coref, token):
        if mention.cluster_id not in coref.clusters:
            continue
        pairs.append((int(mention.mention_id), int(mention.cluster_id)))

    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for pair in pairs:
        if pair in seen:
            continue
        unique_pairs.append(pair)
        seen.add(pair)

    return unique_pairs


def _is_verbal_predicate(token: Any) -> bool:
    return (
        token.pos_ == "VERB"
        or (token.pos_ == "AUX" and token.dep_ in {"ROOT", "cop"})
    )


def _is_passive(predicate: Any) -> bool:
    if any(child.dep_ in AUX_PASSIVE_DEPS for child in predicate.children):
        return True

    if any(child.dep_ in PASSIVE_SUBJECT_DEPS for child in predicate.children):
        return True

    return False


def _is_negated(predicate: Any) -> bool:
    return any(child.dep_ in NEGATION_DEPS for child in predicate.children)


def _predicate_particles(predicate: Any) -> list[Any]:
    return [
        child
        for child in sorted(predicate.children, key=lambda t: t.i)
        if child.dep_ in PARTICLE_DEPS
    ]


def _predicate_label(predicate: Any, *, preposition: Any | None = None) -> str:
    """Build a compact normalized predicate label.

    Examples:
        give
        look_at
        go_to
        pick_up
    """

    base = (predicate.lemma_ or predicate.text).lower()

    particles = [child.text.lower() for child in _predicate_particles(predicate)]
    parts = [base, *particles]

    if preposition is not None:
        parts.append((preposition.lemma_ or preposition.text).lower())

    return "_".join(parts)


def _predicate_span_bounds(predicate: Any, *, preposition: Any | None = None) -> tuple[int, int]:
    """Return a compact token span for the textual predicate trigger."""

    trigger_tokens = [predicate, *_predicate_particles(predicate)]
    if preposition is not None:
        trigger_tokens.append(preposition)

    start = min(token.i for token in trigger_tokens)
    end = max(token.i for token in trigger_tokens) + 1
    return int(start), int(end)


def _subjects_of(predicate: Any) -> list[Any]:
    return [child for child in predicate.children if child.dep_ in SUBJECT_DEPS]


def _direct_objects_of(predicate: Any) -> list[tuple[Any, str, Any | None]]:
    """Return triples: (object_token, dependency_label, preposition_or_None)."""

    objects: list[tuple[Any, str, Any | None]] = []

    for child in predicate.children:
        if child.dep_ in OBJECT_DEPS:
            objects.append((child, child.dep_, None))

    return objects


def _prepositional_objects_of(predicate: Any) -> list[tuple[Any, str, Any]]:
    """Extract cases like went to Camelot / spoke with Arthur."""

    objects: list[tuple[Any, str, Any]] = []

    for prep in predicate.children:
        if prep.dep_ not in PREP_DEPS:
            continue

        for pobj in prep.children:
            if pobj.dep_ in PREP_OBJECT_DEPS:
                objects.append((pobj, pobj.dep_, prep))

    return objects


def _cluster_class_iri(cluster_typing_layer: Any, cluster_id: int) -> str:
    if not hasattr(cluster_typing_layer, "class_iri"):
        raise TypeError("cluster_typing_layer must expose .class_iri(cluster_id).")

    value = cluster_typing_layer.class_iri(cluster_id)
    if value is None or not str(value).strip():
        raise ValueError(f"Empty class_iri for cluster_id={cluster_id}.")

    return str(value).strip()


def _candidate_property_to_dict(spec: Any) -> dict[str, Any]:
    return {
        "iri": str(getattr(spec, "iri")),
        "local_name": str(getattr(spec, "local_name", "")),
        "label": str(getattr(spec, "label", getattr(spec, "local_name", ""))),
        "human_readable_label": str(getattr(spec, "human_readable_label", getattr(spec, "label", ""))),
        "description": str(getattr(spec, "description", "")),
        "domains": list(getattr(spec, "domains", ())),
        "ranges": list(getattr(spec, "ranges", ())),
    }


def _premise_text(candidate: SyntaxRelationCandidate) -> str:
    return (
        f"Context: {candidate.sentence_text}\n\n"
        "Extracted relation:\n"
        f"{candidate.source_canonical_name} -- {candidate.predicate_surface} -- "
        f"{candidate.target_canonical_name}."
    )


def _short_iri(value: str | None) -> str:
    """Return a readable local name for an ontology IRI or label.

    Examples:
        http://example.org/fantasy#Character -> Character
        http://example.org/fantasy/Place     -> Place
        Character                            -> Character
    """

    if value is None:
        return "UNKNOWN"

    text = str(value).strip()
    if not text:
        return "UNKNOWN"

    if "#" in text:
        return text.rsplit("#", 1)[-1] or text

    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1] or text

    return text


def _typed_argument_text(
    *,
    value: str,
    class_iri: str | None,
) -> str:
    """Render one relation endpoint with a readable ontology type."""

    return f"{value} (type: {_short_iri(class_iri)})"


def _no_relationship_reason(
    *,
    subject_class: str | None,
    object_class: str | None,
) -> str:
    """Render the specific type-pair routing failure reason."""

    subject_type = _short_iri(subject_class)
    object_type = _short_iri(object_class)
    return f"no relationship between {subject_type} and {object_type}"


def _relation_discard_message(
    *,
    subject: str,
    subject_class: str | None,
    predicate: str,
    obj: str,
    object_class: str | None,
    reason: str,
) -> str:
    """Format a discard message for a source-predicate-object candidate.

    Output shape:
        [subject (type: SubjectClass)][predicate][object (type: ObjectClass)]
        because [reason]
    """

    return (
        f"[{_typed_argument_text(value=subject, class_iri=subject_class)}]"
        f"[{predicate}]"
        f"[{_typed_argument_text(value=obj, class_iri=object_class)}] "
        f"because [{reason}]"
    )


def _print_discard(
    *,
    subject: str,
    predicate: str,
    obj: str,
    reason: str,
    subject_class: str | None = None,
    object_class: str | None = None,
) -> None:
    """Print a readable discard reason.

    This function is intentionally generic and still supports non-routing
    discard reasons such as missing source/target type. For type-pair routing
    failures, prefer _print_no_relationship_discard(...).
    """

    print(
        _relation_discard_message(
            subject=subject,
            subject_class=subject_class,
            predicate=predicate,
            obj=obj,
            object_class=object_class,
            reason=reason,
        )
    )


def _print_no_relationship_discard(
    *,
    subject: str,
    subject_class: str | None,
    predicate: str,
    obj: str,
    object_class: str | None,
) -> None:
    """Print the specific message used when relation_router has no candidates.

    Requested output shape:
        [subject (type: subject class)][predicate][object (type: object class)]
        because [no relationship between subject class and object class]
    """

    print(
        _relation_discard_message(
            subject=subject,
            subject_class=subject_class,
            predicate=predicate,
            obj=obj,
            object_class=object_class,
            reason=_no_relationship_reason(
                subject_class=subject_class,
                object_class=object_class,
            ),
        )
    )


def iter_syntax_relation_candidates(
    doc: Any,
    *,
    cluster_typing_layer: Any | None = None,
) -> Iterable[SyntaxRelationCandidate]:
    """Yield dependency/coref relation candidates before ontology routing."""

    _ensure_dependency_parse(doc)

    coref = require_coref_layer(doc)
    cluster_typing_layer = _require_cluster_typing_layer(doc, cluster_typing_layer)

    for sentence_index, sent in enumerate(doc.sents):
        for predicate in sent:
            if not _is_verbal_predicate(predicate):
                continue

            subjects = _subjects_of(predicate)
            objects = [
                *_direct_objects_of(predicate),
                *_prepositional_objects_of(predicate),
            ]

            if not subjects or not objects:
                continue

            passive = _is_passive(predicate)
            negated = _is_negated(predicate)

            for subject_token in subjects:
                source_pairs = _cluster_records_for_token(coref, subject_token)
                if not source_pairs:
                    continue

                for object_token, object_dep, prep_token in objects:
                    target_pairs = _cluster_records_for_token(coref, object_token)
                    if not target_pairs:
                        continue

                    predicate_label = _predicate_label(predicate, preposition=prep_token)
                    predicate_start, predicate_end = _predicate_span_bounds(
                        predicate,
                        preposition=prep_token,
                    )
                    predicate_surface = doc[predicate_start:predicate_end].text

                    for (source_mention_id, source_cluster_id), (
                        target_mention_id,
                        target_cluster_id,
                    ) in product(source_pairs, target_pairs):
                        if source_cluster_id == target_cluster_id:
                            continue

                        source_cluster = coref.clusters[source_cluster_id]
                        target_cluster = coref.clusters[target_cluster_id]

                        source_class_iri = _cluster_class_iri(cluster_typing_layer, source_cluster_id)
                        target_class_iri = _cluster_class_iri(cluster_typing_layer, target_cluster_id)

                        relation_mention_id = make_relation_mention_id(
                            predicate_token_i=predicate.i,
                            source_mention_id=source_mention_id,
                            target_mention_id=target_mention_id,
                        )

                        yield SyntaxRelationCandidate(
                            relation_mention_id=relation_mention_id,
                            source_cluster_id=source_cluster_id,
                            source_canonical_name=source_cluster.canonical_name,
                            source_class_iri=source_class_iri,
                            predicate=predicate_label,
                            predicate_surface=predicate_surface,
                            predicate_token_i=predicate.i,
                            predicate_start=predicate_start,
                            predicate_end=predicate_end,
                            target_cluster_id=target_cluster_id,
                            target_canonical_name=target_cluster.canonical_name,
                            target_class_iri=target_class_iri,
                            sentence_index=sentence_index,
                            sentence_start=sent.start,
                            sentence_end=sent.end,
                            sentence_text=sent.text,
                            source_mention_id=source_mention_id,
                            target_mention_id=target_mention_id,
                            source_token_i=subject_token.i,
                            target_token_i=object_token.i,
                            object_dependency=object_dep,
                            preposition=prep_token.text if prep_token is not None else None,
                            is_passive=passive,
                            is_negated=negated,
                        )


def extract_relation_candidates(
    doc: Any,
    *,
    cluster_typing_layer: Any | None = None,
) -> pd.DataFrame:
    """Compatibility helper returning raw syntax/coref candidates as DataFrame."""

    rows = [asdict(row) for row in iter_syntax_relation_candidates(doc, cluster_typing_layer=cluster_typing_layer)]
    columns = list(SyntaxRelationCandidate.__dataclass_fields__)
    return pd.DataFrame(rows, columns=columns)


def export_routed_relation_candidates_jsonl(
    *,
    doc: Any,
    relation_router: Any,
    output_path: str | Path,
    cluster_typing_layer: Any | None = None,
    print_discards: bool = True,
    overwrite: bool = False,
) -> Path:
    """Export routed relation candidates for neural alignment.

    The output is JSONL because candidate_properties is nested. Discarded rows
    are not saved; they are optionally printed in the requested format.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass overwrite=True to replace it.")

    n_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for candidate in iter_syntax_relation_candidates(doc, cluster_typing_layer=cluster_typing_layer):
            source_class_iri = candidate.source_class_iri
            target_class_iri = candidate.target_class_iri

            if not source_class_iri:
                if print_discards:
                    _print_discard(
                        subject=candidate.source_canonical_name,
                        subject_class=source_class_iri,
                        predicate=candidate.predicate,
                        obj=candidate.target_canonical_name,
                        object_class=target_class_iri,
                        reason="missing source type",
                    )
                continue

            if not target_class_iri:
                if print_discards:
                    _print_discard(
                        subject=candidate.source_canonical_name,
                        subject_class=source_class_iri,
                        predicate=candidate.predicate,
                        obj=candidate.target_canonical_name,
                        object_class=target_class_iri,
                        reason="missing object type",
                    )
                continue

            candidate_properties = tuple(
                relation_router.candidates_for(source_class_iri, target_class_iri)
            )

            if not candidate_properties:
                if print_discards:
                    _print_no_relationship_discard(
                        subject=candidate.source_canonical_name,
                        subject_class=source_class_iri,
                        predicate=candidate.predicate,
                        obj=candidate.target_canonical_name,
                        object_class=target_class_iri,
                    )
                continue

            routed = RoutedRelationCandidate(
                relation_mention_id=candidate.relation_mention_id,
                source_mention_id=candidate.source_mention_id,
                predicate_token_i=candidate.predicate_token_i,
                predicate_start=candidate.predicate_start,
                predicate_end=candidate.predicate_end,
                target_mention_id=candidate.target_mention_id,
                source_cluster_id=candidate.source_cluster_id,
                source_canonical_name=candidate.source_canonical_name,
                source_class_iri=source_class_iri,
                predicate=candidate.predicate,
                predicate_surface=candidate.predicate_surface,
                target_cluster_id=candidate.target_cluster_id,
                target_canonical_name=candidate.target_canonical_name,
                target_class_iri=target_class_iri,
                sentence_index=candidate.sentence_index,
                sentence_start=candidate.sentence_start,
                sentence_end=candidate.sentence_end,
                sentence_text=candidate.sentence_text,
                premise_text=_premise_text(candidate),
                candidate_properties=[_candidate_property_to_dict(spec) for spec in candidate_properties],
                source_token_i=candidate.source_token_i,
                target_token_i=candidate.target_token_i,
                object_dependency=candidate.object_dependency,
                preposition=candidate.preposition,
                is_passive=candidate.is_passive,
                is_negated=candidate.is_negated,
            )

            f.write(json.dumps(asdict(routed), ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[relation extraction] Wrote {n_written} routed candidates to {output_path}")
    return output_path


# Backwards-friendly alias with the shorter name used in notebook discussions.
export_routed_relation_candidates = export_routed_relation_candidates_jsonl
