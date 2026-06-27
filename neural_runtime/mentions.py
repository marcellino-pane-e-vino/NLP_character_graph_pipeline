from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from typing import Any


__all__ = [
    "ContextConfig",
    "MentionRenderingConfig",
    "MentionRecord",
    "validate_text",
    "normalize_context_for_dedup",
    "mention_ids_for_cluster",
    "canonical_name_for_cluster",
    "find_mention_by_id",
    "sentence_index_for_mention",
    "mention_context_text",
    "mention_replacement_for_rendering",
    "rendered_mention_context_text",
    "sampled_mention_pairs_for_cluster",
    "mention_records_for_cluster",
    "cluster_random_seed",
]


@dataclass(frozen=True)
class ContextConfig:
    """Controls context extraction around a mention."""

    n_sentences_before: int = 0
    n_sentences_after: int = 0
    mark_mention: bool = True
    deduplicate: bool = False


@dataclass(frozen=True)
class MentionRenderingConfig:
    """Simple rules for rendering the target mention before model scoring."""

    canonicalize_simple_mentions: bool = True
    keep_first_second_person: bool = True


FIRST_SECOND_PERSON_PRONOUNS: set[str] = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
}
THIRD_PERSON_SUBJECT_PRONOUNS: set[str] = {"he", "she", "they", "it"}
THIRD_PERSON_OBJECT_PRONOUNS: set[str] = {"him", "her", "them"}
THIRD_PERSON_POSSESSIVE_PRONOUNS: set[str] = {"his", "hers", "their", "theirs", "its"}
THIRD_PERSON_REFLEXIVE_PRONOUNS: set[str] = {
    "himself", "herself", "itself", "themself", "themselves",
}
AMBIGUOUS_OR_RELATIONAL_NOMINALS: set[str] = {
    "my dear", "your dear", "my friend", "your friend", "our friend",
    "this fellow", "that fellow", "the poor fellow", "the stranger", "the creature",
}


# =============================================================================
# Text normalization and entity helpers
# =============================================================================


# =============================================================================
# Text normalization and entity helpers
# =============================================================================


def validate_text(text: str, *, field_name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be str, got {type(text)!r}")
    text = text.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def normalize_context_for_dedup(text: str) -> str:
    text = validate_text(text)
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(
        r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*—\s*", "—", text)
    return text.strip()


def _require_entities(doc: Any) -> Any:
    from annotation_layer.spacy_extension import require_entities

    return require_entities(doc)


def mention_ids_for_cluster(doc: Any, cluster_id: int) -> list[int]:
    entities = _require_entities(doc)
    if cluster_id not in entities.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    return list(entities.clusters[cluster_id].mention_ids)


def canonical_name_for_cluster(doc: Any, cluster_id: int) -> str:
    entities = _require_entities(doc)
    if cluster_id not in entities.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    canonical_name = str(entities.clusters[cluster_id].canonical_name).strip()
    if not canonical_name:
        raise ValueError(f"cluster_id={cluster_id} has an empty canonical_name")
    return canonical_name


def find_mention_by_id(doc: Any, mention_id: int) -> Any:
    entities = _require_entities(doc)
    if hasattr(entities, "mentions") and mention_id in entities.mentions:
        return entities.mentions[mention_id]
    raise KeyError(f"mention_id={mention_id} not found in doc._.annotation_layer.entities")


def _sentences(doc: Any) -> list[Any]:
    try:
        sentences = list(doc.sents)
    except ValueError as exc:
        raise ValueError(
            "This doc has no sentence boundaries. Run a parser, sentencizer, "
            "or sentence-boundary component before using mention contexts."
        ) from exc
    if not sentences:
        raise ValueError("doc.sents is empty")
    return sentences


def sentence_index_for_mention(doc: Any, mention_id: int) -> int:
    mention = find_mention_by_id(doc, mention_id)
    for i, sent in enumerate(_sentences(doc)):
        if sent.start <= mention.start and mention.end <= sent.end:
            return i
    raise ValueError(
        f"Could not find sentence containing mention_id={mention_id} "
        f"with token span [{mention.start}:{mention.end}]."
    )


def mention_context_text(
    doc: Any,
    mention_id: int,
    *,
    config: ContextConfig | None = None,
) -> str:
    config = config or ContextConfig()
    mention = find_mention_by_id(doc, mention_id)
    sentences = _sentences(doc)
    center_i = sentence_index_for_mention(doc, mention_id)
    start_i = max(0, center_i - config.n_sentences_before)
    end_i = min(len(sentences), center_i + config.n_sentences_after + 1)

    parts: list[str] = []
    for sent_i in range(start_i, end_i):
        sent = sentences[sent_i]
        if sent_i != center_i or not config.mark_mention:
            parts.append(sent.text)
            continue
        before = doc[sent.start: mention.start].text
        target = doc[mention.start: mention.end].text
        after = doc[mention.end: sent.end].text
        parts.append(f"{before} [[{target}]] {after}".strip())
    return normalize_context_for_dedup(" ".join(parts))


# =============================================================================
# Mention rendering
# =============================================================================


def canonical_possessive(subject: str | None) -> str:
    subject = (subject or "the character").strip() or "the character"
    if subject.endswith("'"):
        return subject
    if subject.endswith("s"):
        return f"{subject}'"
    return f"{subject}'s"


def _single_token_for_mention(doc: Any, mention: Any) -> Any | None:
    start = int(getattr(mention, "start"))
    end = int(getattr(mention, "end"))
    if end - start != 1:
        return None
    try:
        return doc[start]
    except Exception:
        return None


def _token_is_possessive_pronoun(token: Any | None) -> bool:
    if token is None:
        return False
    tag = str(getattr(token, "tag_", "") or "")
    morph = str(getattr(token, "morph", "") or "")
    return tag == "PRP$" or "Poss=Yes" in morph


def _looks_like_possessive_her_without_tags(doc: Any, mention: Any) -> bool:
    end = int(getattr(mention, "end"))
    if end >= len(doc):
        return False
    next_token = doc[end]
    next_text = str(getattr(next_token, "text", "") or "").lower()
    if next_text in {",", ".", ";", ":", "!", "?", "and", "or", "but", "to", "of", "in", "on", "at", "by", "with", "from", "for", "as"}:
        return False
    pos = str(getattr(next_token, "pos_", "") or "")
    if pos in {"NOUN", "PROPN", "ADJ"}:
        return True
    return bool(re.match(r"^[a-zA-Z][a-zA-Z'-]*$", next_text))


def _is_simple_nominal_mention(mention_text: str) -> bool:
    text = normalize_context_for_dedup(mention_text).lower()
    if text in AMBIGUOUS_OR_RELATIONAL_NOMINALS:
        return False
    if not text.startswith("the "):
        return False
    return len(text.split()) <= 4


def _is_proper_name_like_mention(mention_text: str, subject: str | None) -> bool:
    text = normalize_context_for_dedup(mention_text)
    subject_text = normalize_context_for_dedup(subject or "") if subject else ""
    if subject_text and text.lower() == subject_text.lower():
        return True
    tokens = [token for token in re.split(r"\s+", text) if token]
    return any(token[:1].isupper() and token.lower() != "i" for token in tokens)


def mention_replacement_for_rendering(
    doc: Any,
    mention_id: int,
    *,
    subject: str | None,
    rendering_config: MentionRenderingConfig | None = None,
) -> tuple[str, str]:
    rendering_config = rendering_config or MentionRenderingConfig()
    mention = find_mention_by_id(doc, mention_id)
    mention_text = str(getattr(mention, "text", "") or doc[mention.start: mention.end].text)
    mention_clean = normalize_context_for_dedup(mention_text)
    mention_lower = mention_clean.lower()
    subject_text = (subject or "the character").strip() or "the character"

    if not rendering_config.canonicalize_simple_mentions:
        return mention_clean, "canonicalization_disabled_keep_original"
    if rendering_config.keep_first_second_person and mention_lower in FIRST_SECOND_PERSON_PRONOUNS:
        return mention_clean, "first_second_person_keep_original"
    if _is_proper_name_like_mention(mention_clean, subject_text):
        return subject_text, "proper_name_to_canonical"

    token = _single_token_for_mention(doc, mention)

    if mention_lower in THIRD_PERSON_SUBJECT_PRONOUNS:
        return subject_text, "third_person_subject_pronoun_to_canonical"
    if mention_lower == "her":
        if _token_is_possessive_pronoun(token) or _looks_like_possessive_her_without_tags(doc, mention):
            return canonical_possessive(subject_text), "third_person_possessive_pronoun_to_canonical_possessive"
        return subject_text, "third_person_object_pronoun_to_canonical"
    if mention_lower in THIRD_PERSON_OBJECT_PRONOUNS:
        return subject_text, "third_person_object_pronoun_to_canonical"
    if mention_lower in THIRD_PERSON_POSSESSIVE_PRONOUNS:
        return canonical_possessive(subject_text), "third_person_possessive_pronoun_to_canonical_possessive"
    if mention_lower in THIRD_PERSON_REFLEXIVE_PRONOUNS:
        return mention_clean, "third_person_reflexive_keep_original"
    if _is_simple_nominal_mention(mention_clean):
        return subject_text, "simple_nominal_to_canonical"
    return mention_clean, "ambiguous_or_unsupported_keep_original"


def rendered_mention_context_text(
    doc: Any,
    mention_id: int,
    *,
    subject: str | None,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
) -> tuple[str, str, bool]:
    context_config = context_config or ContextConfig()
    rendering_config = rendering_config or MentionRenderingConfig()
    mention = find_mention_by_id(doc, mention_id)
    sentences = _sentences(doc)
    center_i = sentence_index_for_mention(doc, mention_id)
    start_i = max(0, center_i - context_config.n_sentences_before)
    end_i = min(len(sentences), center_i + context_config.n_sentences_after + 1)

    replacement, render_rule = mention_replacement_for_rendering(
        doc,
        mention_id,
        subject=subject,
        rendering_config=rendering_config,
    )
    target_original = doc[mention.start: mention.end].text
    was_changed = normalize_context_for_dedup(target_original) != normalize_context_for_dedup(replacement)

    parts: list[str] = []
    for sent_i in range(start_i, end_i):
        sent = sentences[sent_i]
        if sent_i != center_i:
            parts.append(sent.text)
            continue
        before = doc[sent.start: mention.start].text
        after = doc[mention.end: sent.end].text
        parts.append(f"{before} {replacement} {after}".strip())
    return normalize_context_for_dedup(" ".join(parts)), render_rule, was_changed


# =============================================================================
# Mention records and sampling
# =============================================================================


@dataclass(frozen=True)
class MentionRecord:
    cluster_id: int
    subject: str | None
    mention_index_in_cluster: int
    mention_id: int
    mention_text: str
    mention_start: int
    mention_end: int
    sentence_index: int
    context_text: str
    normalized_context_text: str
    original_context_text: str = ""
    rendered_context_text: str = ""
    mention_render_rule: str = "legacy_context"
    mention_render_was_changed: bool = False


def cluster_random_seed(base_seed: int | None, cluster_id: int) -> int | None:
    if base_seed is None:
        return None
    digest = hashlib.sha256(f"{int(base_seed)}:{int(cluster_id)}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sampled_mention_pairs_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    n_mentions: int | None = None,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
) -> list[tuple[int, int]]:
    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")

    all_mention_ids = mention_ids_for_cluster(doc, cluster_id)
    total_mentions = len(all_mention_ids)

    if n_mentions is None:
        selected_indexes = list(range(total_mentions))
    else:
        if n_mentions > total_mentions:
            raise ValueError(
                f"cluster_id={cluster_id} contains only {total_mentions} mentions, "
                f"but n_mentions={n_mentions} was requested."
            )
        rng = random.Random(random_seed)
        selected_indexes = rng.sample(range(total_mentions), n_mentions)
        if sort_sample_by_cluster_order:
            selected_indexes.sort()

    return [(index, int(all_mention_ids[index])) for index in selected_indexes]


def mention_records_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    subject: str | None = None,
    n_mentions: int | None = None,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
) -> list[MentionRecord]:
    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    subject = subject or canonical_name_for_cluster(doc, cluster_id)

    selected_pairs = sampled_mention_pairs_for_cluster(
        doc,
        cluster_id,
        n_mentions=n_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
    )

    records: list[MentionRecord] = []
    for mention_index, mention_id in selected_pairs:
        mention = find_mention_by_id(doc, mention_id)
        sentence_index = sentence_index_for_mention(doc, mention_id)
        original_context_text = mention_context_text(doc, mention_id, config=context_config)
        rendered_context, render_rule, render_was_changed = rendered_mention_context_text(
            doc,
            mention_id,
            subject=subject,
            context_config=context_config,
            rendering_config=rendering_config,
        )
        records.append(
            MentionRecord(
                cluster_id=int(cluster_id),
                subject=subject,
                mention_index_in_cluster=int(mention_index),
                mention_id=int(mention_id),
                mention_text=str(getattr(mention, "text", "")),
                mention_start=int(getattr(mention, "start")),
                mention_end=int(getattr(mention, "end")),
                sentence_index=int(sentence_index),
                context_text=rendered_context,
                normalized_context_text=normalize_context_for_dedup(rendered_context),
                original_context_text=original_context_text,
                rendered_context_text=rendered_context,
                mention_render_rule=render_rule,
                mention_render_was_changed=bool(render_was_changed),
            )
        )
    return records



