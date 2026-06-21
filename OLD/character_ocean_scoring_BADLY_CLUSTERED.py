from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Any, Callable, Iterable, Optional

import torch
from transformers import pipeline


# =============================================================================
# 1. Configuration and label definitions
# =============================================================================

OCEAN_TRAITS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_HYPOTHESIS_TEMPLATE = "This text shows {}."
DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE = "{subject} shows {} in this text."

# Default destination for long-running OCEAN profile exports.
# This keeps the generated artifacts grouped under ./outputs/OCEAN_profiles
# instead of mixing them directly into ./outputs.
DEFAULT_OCEAN_PROFILES_OUTPUT_DIR = Path("./outputs/OCEAN_profiles")

# Bump this only when the cached probability payload format changes.
PROBABILITY_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CollapseConfig:
    """Controls how positive/negative/neutral probabilities become a 0-100 score."""

    neutral_score: float = 50.0
    evidence_margin_threshold: float = 0.15
    low_trait_threshold: float = 30.0
    high_trait_threshold: float = 70.0
    rounding_digits: int = 2


@dataclass(frozen=True)
class OceanScoringConfig:
    """
    Runtime configuration for zero-shot OCEAN scoring.

    subject_aware:
        If True and subject is provided, the hypothesis becomes subject-specific:
            "Dorothy shows high agreeableness in this text."
        Otherwise the generic template is used:
            "This text shows high agreeableness."

    batch_size:
        Hugging Face pipeline batch size.

    multi_label:
        Keep False for the current three-way positive/negative/neutral competition.
    """

    model_name: str = DEFAULT_MODEL_NAME
    generic_hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE
    subject_hypothesis_template: str = DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE
    subject_aware: bool = True
    batch_size: int = 16
    truncation: bool = True
    multi_label: bool = False
    collapse: CollapseConfig = field(default_factory=CollapseConfig)


@dataclass(frozen=True)
class ContextConfig:
    """Controls the amount of context extracted around a mention."""

    n_sentences_before: int = 0
    n_sentences_after: int = 0
    mark_mention: bool = True
    deduplicate: bool = True


@dataclass(frozen=True)
class MentionRenderingConfig:
    """
    Simple rules for rendering the target mention before OCEAN scoring.

    The scorer uses rendered_context_text, while the CSV also keeps the original
    mention-marked context for audit/debugging.

    canonicalize_simple_mentions:
        If True, replace safe mentions with the cluster canonical name:
            - proper-name mentions
            - third-person personal pronouns
            - simple nominal mentions such as "the girl" or "the Scarecrow"

    keep_first_second_person:
        If True, never canonicalize I/me/my/we/us/you/your pronouns, because
        direct replacement inside dialogue usually corrupts the sentence.
    """

    canonicalize_simple_mentions: bool = True
    keep_first_second_person: bool = True


FIRST_SECOND_PERSON_PRONOUNS: set[str] = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

THIRD_PERSON_SUBJECT_PRONOUNS: set[str] = {"he", "she", "they", "it"}
THIRD_PERSON_OBJECT_PRONOUNS: set[str] = {"him", "her", "them"}
THIRD_PERSON_POSSESSIVE_PRONOUNS: set[str] = {
    "his",
    "hers",
    "their",
    "theirs",
    "its",
}
THIRD_PERSON_REFLEXIVE_PRONOUNS: set[str] = {
    "himself",
    "herself",
    "itself",
    "themself",
    "themselves",
}

AMBIGUOUS_OR_RELATIONAL_NOMINALS: set[str] = {
    "my dear",
    "your dear",
    "my friend",
    "your friend",
    "our friend",
    "this fellow",
    "that fellow",
    "the poor fellow",
    "the stranger",
    "the creature",
}


def canonical_possessive(subject: str | None) -> str:
    """Return a simple English possessive form for the canonical subject."""

    subject = (subject or "the character").strip() or "the character"

    if subject.endswith("'"):
        return subject

    if subject.endswith("s"):
        return f"{subject}'"

    return f"{subject}'s"


# Each trait has competing labels: high-side, low-side, no-evidence.
# These can be edited without touching batching, caching, doc extraction, or aggregation.
OCEAN_LABELS: dict[str, dict[str, str]] = {
    "openness": {
        "positive": (
            "high openness: curiosity, imagination, exploration, intellectual interest, "
            "openness to new experiences"
        ),
        "negative": (
            "low openness: rigidity, lack of curiosity, conventionality, "
            "resistance to new experiences"
        ),
        "neutral": (
            "no clear evidence about openness, curiosity, imagination, or exploration"
        ),
    },
    "conscientiousness": {
        "positive": (
            "high conscientiousness: carefulness, planning, responsibility, "
            "discipline, persistence, duty"
        ),
        "negative": (
            "low conscientiousness: carelessness, impulsiveness, irresponsibility, "
            "disorganization, lack of planning"
        ),
        "neutral": (
            "no clear evidence about conscientiousness, carefulness, planning, or responsibility"
        ),
    },
    "extraversion": {
        "positive": (
            "high extraversion: sociability, friendliness, outgoing social engagement, "
            "enthusiasm, enjoyment of interaction"
        ),
        "negative": (
            "low extraversion: social withdrawal, reserve, avoidance of interaction, "
            "quietness, lack of social engagement"
        ),
        "neutral": (
            "no clear evidence about extraversion, sociability, assertiveness, or outgoing energy"
        ),
    },
    "agreeableness": {
        "positive": (
            "high agreeableness: kindness, compassion, cooperation, helpfulness, "
            "trust, concern for others"
        ),
        "negative": (
            "low agreeableness: hostility, selfishness, cruelty, refusal to help, "
            "uncooperative behavior"
        ),
        "neutral": (
            "no clear evidence about agreeableness, kindness, compassion, cooperation, or hostility"
        ),
    },
    "neuroticism": {
        "positive": (
            "high neuroticism: fear, anxiety, sadness, distress, emotional instability, worry"
        ),
        "negative": (
            "low neuroticism: calmness, confidence, emotional stability, composure, lack of distress"
        ),
        "neutral": (
            "no clear evidence about neuroticism, fear, anxiety, sadness, worry, or emotional stability"
        ),
    },
}


# Optional future direction: trait decomposition.
# Not used by default; included as a simple extension point, not as a second framework.
DECOMPOSED_OCEAN_LABELS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "agreeableness": {
        "positive": (
            "kind toward others",
            "helpful or cooperative toward others",
            "compassionate or concerned for others",
        ),
        "negative": (
            "hostile toward others",
            "selfish or uncooperative toward others",
            "cruel or indifferent to others",
        ),
        "neutral": "no clear evidence about agreeableness or hostility",
    },
}


# =============================================================================
# 2. Text normalization and deduplication
# =============================================================================


def validate_text(text: str, *, field_name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be str, got {type(text)!r}")

    text = text.strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text



def normalize_context_for_dedup(text: str) -> str:
    """
    Normalize mention-context text for deduplication.

    Handles:
        - newlines
        - repeated spaces
        - spaces before punctuation
        - tokenizer artifacts like child 's -> child's, do n't -> don't
        - em dash spacing
    """

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



def unique_preserving_order(items: Iterable[str]) -> list[str]:
    """Return unique strings while preserving first occurrence order."""

    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        key = normalize_context_for_dedup(item)
        if key not in seen:
            result.append(key)
            seen.add(key)

    return result


# =============================================================================
# 3. Coreference / mention-context extraction
# =============================================================================


def require_coref_layer(doc: Any) -> Any:
    """Return doc._.coref_layer or raise a clear error."""

    if not hasattr(doc, "_") or not hasattr(doc._, "coref_layer"):
        raise ValueError("doc has no doc._.coref_layer")

    coref_layer = doc._.coref_layer

    if coref_layer is None:
        raise ValueError("doc._.coref_layer is None")

    return coref_layer



def mention_ids_for_cluster(doc: Any, cluster_id: int) -> list[int]:
    """Return all mention ids for one cluster id."""

    coref_layer = require_coref_layer(doc)

    if not hasattr(coref_layer, "clusters"):
        raise ValueError("doc._.coref_layer has no .clusters attribute")

    if cluster_id not in coref_layer.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    cluster = coref_layer.clusters[cluster_id]

    if not hasattr(cluster, "mention_ids"):
        raise ValueError(f"cluster_id={cluster_id} has no .mention_ids attribute")

    return list(cluster.mention_ids)



def canonical_name_for_cluster(doc: Any, cluster_id: int) -> str:
    """
    Return the canonical name stored in doc._.coref_layer.clusters[cluster_id].

    This module treats the annotated Doc as the source of truth for cluster names.
    With the project CorefLayer schema, every Cluster has a required
    canonical_name: str, so notebook code should not maintain a manual
    cluster_id -> subject dictionary.
    """

    coref_layer = require_coref_layer(doc)

    if not hasattr(coref_layer, "clusters"):
        raise ValueError("doc._.coref_layer has no .clusters attribute")

    if cluster_id not in coref_layer.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    cluster = coref_layer.clusters[cluster_id]
    canonical_name = getattr(cluster, "canonical_name", None)

    if canonical_name is None:
        raise ValueError(
            f"cluster_id={cluster_id} has no canonical_name. "
            "Expected doc._.coref_layer.clusters[cluster_id].canonical_name."
        )

    canonical_name = str(canonical_name).strip()

    if not canonical_name:
        raise ValueError(f"cluster_id={cluster_id} has an empty canonical_name")

    return canonical_name



def find_mention_by_id(doc: Any, mention_id: int) -> Any:
    """
    Find one mention object by mention_id.

    Defensive on purpose: different CorefLayer implementations may expose mentions
    through a dict, a list, a global iterator, or only cluster iterators.
    """

    coref_layer = require_coref_layer(doc)

    if hasattr(coref_layer, "mentions"):
        mentions = coref_layer.mentions

        if isinstance(mentions, dict) and mention_id in mentions:
            return mentions[mention_id]

        if not isinstance(mentions, dict):
            for mention in mentions:
                if getattr(mention, "mention_id", None) == mention_id:
                    return mention

    if hasattr(coref_layer, "iter_mentions"):
        for mention in coref_layer.iter_mentions():
            if getattr(mention, "mention_id", None) == mention_id:
                return mention

    if hasattr(coref_layer, "clusters") and hasattr(coref_layer, "iter_cluster_mentions"):
        for cluster_id in coref_layer.clusters:
            for mention in coref_layer.iter_cluster_mentions(cluster_id):
                if getattr(mention, "mention_id", None) == mention_id:
                    return mention

    raise KeyError(f"mention_id={mention_id} not found in doc._.coref_layer")



def _sentences(doc: Any) -> list[Any]:
    """Return doc sentences with a useful error if sentence boundaries are missing."""

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
    """Return the index of the sentence containing mention_id."""

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
    """
    Return context around one mention:
        [sentences before] + [sentence containing mention] + [sentences after]

    If config.mark_mention is True, only the target mention is wrapped as [[...]].
    """

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

        before = doc[sent.start : mention.start].text
        target = doc[mention.start : mention.end].text
        after = doc[mention.end : sent.end].text
        marked = f"{before} [[{target}]] {after}".strip()
        parts.append(marked)

    return normalize_context_for_dedup(" ".join(parts))





def _single_token_for_mention(doc: Any, mention: Any) -> Any | None:
    """Return the single spaCy token covered by a one-token mention, if any."""

    start = int(getattr(mention, "start"))
    end = int(getattr(mention, "end"))

    if end - start != 1:
        return None

    try:
        return doc[start]
    except Exception:
        return None


def _token_is_possessive_pronoun(token: Any | None) -> bool:
    """Best-effort detection of English possessive pronouns such as her/my/your."""

    if token is None:
        return False

    tag = str(getattr(token, "tag_", "") or "")
    morph = str(getattr(token, "morph", "") or "")

    return tag == "PRP$" or "Poss=Yes" in morph


def _looks_like_possessive_her_without_tags(doc: Any, mention: Any) -> bool:
    """
    Conservative fallback for ambiguous 'her'.

    If POS tags are available, _token_is_possessive_pronoun() should decide.
    Without tags, treat 'her' as possessive only when it is immediately followed
    by a plausible noun/proper/adjective token inside the same sentence.
    """

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

    # Last-resort English heuristic: "her eyes", "her aunt", "her bed".
    return bool(re.match(r"^[a-zA-Z][a-zA-Z'-]*$", next_text))


def _is_simple_nominal_mention(mention_text: str) -> bool:
    """Simple, intentionally conservative nominal-mention rule."""

    text = normalize_context_for_dedup(mention_text).lower()

    if text in AMBIGUOUS_OR_RELATIONAL_NOMINALS:
        return False

    if not text.startswith("the "):
        return False

    # Keep it simple: replace short definite NPs, reject long descriptions.
    return len(text.split()) <= 4


def _is_proper_name_like_mention(mention_text: str, subject: str | None) -> bool:
    """Best-effort proper-name detection without depending on a specific schema."""

    text = normalize_context_for_dedup(mention_text)
    subject_text = normalize_context_for_dedup(subject or "") if subject else ""

    if subject_text and text.lower() == subject_text.lower():
        return True

    # Multi-token names/titles like "the Great Oz" should be canonicalized too.
    # Avoid classifying the pronoun "I" as a proper name.
    tokens = [token for token in re.split(r"\s+", text) if token]
    return any(token[:1].isupper() and token.lower() != "i" for token in tokens)


def mention_replacement_for_rendering(
    doc: Any,
    mention_id: int,
    *,
    subject: str | None,
    rendering_config: MentionRenderingConfig | None = None,
) -> tuple[str, str]:
    """
    Return (replacement_text, render_rule) for the target mention.

    Simple policy:
        1. first/second-person pronouns -> keep original
        2. proper-name mentions -> canonical subject
        3. third-person pronouns -> canonical subject or possessive subject
        4. simple nominal mentions -> canonical subject
        5. everything else -> keep original
    """

    rendering_config = rendering_config or MentionRenderingConfig()
    mention = find_mention_by_id(doc, mention_id)
    mention_text = str(getattr(mention, "text", "") or doc[mention.start : mention.end].text)
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
    """
    Render context around a mention using the simple canonicalization policy.

    Returns:
        (rendered_context_text, mention_render_rule, mention_render_was_changed)

    The returned text is unmarked and intended for model scoring.
    """

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

    target_original = doc[mention.start : mention.end].text
    was_changed = normalize_context_for_dedup(target_original) != normalize_context_for_dedup(replacement)

    parts: list[str] = []

    for sent_i in range(start_i, end_i):
        sent = sentences[sent_i]

        if sent_i != center_i:
            parts.append(sent.text)
            continue

        before = doc[sent.start : mention.start].text
        after = doc[mention.end : sent.end].text
        rendered = f"{before} {replacement} {after}".strip()
        parts.append(rendered)

    return normalize_context_for_dedup(" ".join(parts)), render_rule, was_changed


def mention_contexts_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    config: ContextConfig | None = None,
) -> list[str]:
    """Extract mention contexts for every mention in one cluster."""

    config = config or ContextConfig()
    contexts = [
        mention_context_text(doc, mention_id, config=config)
        for mention_id in mention_ids_for_cluster(doc, cluster_id)
    ]

    if config.deduplicate:
        return unique_preserving_order(contexts)

    return contexts


# =============================================================================
# 4. Zero-shot model backend
# =============================================================================


@lru_cache(maxsize=4)
def load_zero_shot_classifier(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load and cache the Hugging Face zero-shot classifier."""

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=device,
        framework="pt",
    )



def _hypothesis_template(config: OceanScoringConfig, subject: str | None) -> str:
    """
    Build the Hugging Face zero-shot hypothesis template.

    Important:
        Hugging Face still needs one literal positional placeholder: {}
        because the pipeline injects each candidate label there.

    Therefore, when making the template subject-aware, replace only the
    named {subject} placeholder and preserve the literal {} placeholder.

    Example:
        {subject} shows {} in this text.
        -> Dorothy shows {} in this text.
    """

    if config.subject_aware and subject:
        return config.subject_hypothesis_template.replace("{subject}", subject)

    return config.generic_hypothesis_template



def _classifier_outputs_to_list(model_outputs: Any) -> list[dict[str, Any]]:
    """Hugging Face returns dict for one input and list[dict] for many inputs."""

    if isinstance(model_outputs, dict):
        return [model_outputs]

    return list(model_outputs)



def scores_by_label(model_output: dict[str, Any]) -> dict[str, float]:
    """Convert {'labels': [...], 'scores': [...]} into {label: score}."""

    return {
        str(label): float(score)
        for label, score in zip(model_output["labels"], model_output["scores"])
    }


# =============================================================================
# 5. OCEAN scoring and score collapsing
# =============================================================================


def collapse_trait_scores(
    *,
    positive: float,
    negative: float,
    neutral: float,
    config: CollapseConfig | None = None,
) -> float:
    """
    Convert positive/negative/neutral evidence into a 0-100 score.

    Meaning:
        100 = strong high-trait evidence
         50 = neutral / no usable evidence
          0 = strong low-trait evidence
    """

    config = config or CollapseConfig()

    evidence_strength = max(positive, negative) - neutral
    if evidence_strength < config.evidence_margin_threshold:
        return config.neutral_score

    denominator = positive + negative
    if denominator <= 1e-9:
        return config.neutral_score

    bipolar_score = 100.0 * positive / denominator

    if config.low_trait_threshold < bipolar_score < config.high_trait_threshold:
        return config.neutral_score

    return round(float(bipolar_score), config.rounding_digits)



def _score_trait_batch(
    texts: list[str],
    *,
    trait_labels: dict[str, str],
    classifier: Any,
    hypothesis_template: str,
    config: OceanScoringConfig,
) -> list[float]:
    """Score one OCEAN trait for a batch of texts."""

    positive_label = trait_labels["positive"]
    negative_label = trait_labels["negative"]
    neutral_label = trait_labels["neutral"]

    model_outputs = classifier(
        sequences=texts,
        candidate_labels=[positive_label, negative_label, neutral_label],
        hypothesis_template=hypothesis_template,
        multi_label=config.multi_label,
        batch_size=config.batch_size,
        truncation=config.truncation,
    )

    scores: list[float] = []

    for model_output in _classifier_outputs_to_list(model_outputs):
        label_scores = scores_by_label(model_output)
        scores.append(
            collapse_trait_scores(
                positive=label_scores[positive_label],
                negative=label_scores[negative_label],
                neutral=label_scores[neutral_label],
                config=config.collapse,
            )
        )

    return scores



def score_ocean_texts(
    texts: list[str],
    *,
    subject: str | None = None,
    config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, float]]:
    """
    Score many texts for the five OCEAN traits.

    INPUT:
        texts: list[str]

    OUTPUT:
        [
            {
                "openness": float,
                "conscientiousness": float,
                "extraversion": float,
                "agreeableness": float,
                "neuroticism": float,
            },
            ...
        ]
    """

    if not isinstance(texts, list):
        raise TypeError(f"texts must be list[str], got {type(texts)!r}")

    texts = [validate_text(text, field_name="text") for text in texts]
    if not texts:
        return []

    config = config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS

    classifier = load_zero_shot_classifier(config.model_name)
    hypothesis_template = _hypothesis_template(config, subject)

    results: list[dict[str, float]] = [{trait: config.collapse.neutral_score for trait in OCEAN_TRAITS} for _ in texts]

    for trait, labels in trait_labels.items():
        trait_scores = _score_trait_batch(
            texts,
            trait_labels=labels,
            classifier=classifier,
            hypothesis_template=hypothesis_template,
            config=config,
        )

        for i, score in enumerate(trait_scores):
            results[i][trait] = score

    return results



def score_ocean_text(
    text: str,
    *,
    subject: str | None = None,
    config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
) -> dict[str, float]:
    """Score one text span for OCEAN."""

    return score_ocean_texts(
        [text],
        subject=subject,
        config=config,
        trait_labels=trait_labels,
    )[0]


# =============================================================================
# 6. Optional SQLite cache for resumable long runs
# =============================================================================


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    """Convert dataclasses, paths, tuples, and sets into stable JSON-ready data."""

    if is_dataclass(value):
        return _json_ready(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]

    return value


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file and then atomically replace the target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _cache_key(
    *,
    text: str,
    subject: str | None,
    config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
) -> str:
    payload = {
        "text": normalize_context_for_dedup(text),
        "subject": subject,
        "model_name": config.model_name,
        "subject_aware": config.subject_aware,
        "generic_hypothesis_template": config.generic_hypothesis_template,
        "subject_hypothesis_template": config.subject_hypothesis_template,
        "collapse": config.collapse.__dict__,
        "trait_labels": trait_labels,
    }

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


class SQLiteScoreCache:
    """
    Minimal SQLite cache for marathon scoring runs.

    One row = one scored normalized text under one scoring configuration.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ocean_score_cache (
                cache_key TEXT PRIMARY KEY,
                score_json TEXT NOT NULL,
                created_at_unix REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def get(self, key: str) -> Optional[dict[str, float]]:
        row = self.connection.execute(
            "SELECT score_json FROM ocean_score_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        return {k: float(v) for k, v in json.loads(row[0]).items()}

    def set(self, key: str, score: dict[str, float]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO ocean_score_cache(cache_key, score_json, created_at_unix)
            VALUES (?, ?, ?)
            """,
            (key, _stable_json(score), time.time()),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteScoreCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()



def score_ocean_texts_cached(
    texts: list[str],
    *,
    subject: str | None = None,
    config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    cache: SQLiteScoreCache | None = None,
) -> list[dict[str, float]]:
    """Score texts using cache where possible and batching only uncached texts."""

    config = config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    texts = [normalize_context_for_dedup(text) for text in texts]

    if cache is None:
        return score_ocean_texts(
            texts,
            subject=subject,
            config=config,
            trait_labels=trait_labels,
        )

    results: list[Optional[dict[str, float]]] = [None] * len(texts)
    misses: list[str] = []
    miss_indexes: list[int] = []
    miss_keys: list[str] = []

    for i, text in enumerate(texts):
        key = _cache_key(
            text=text,
            subject=subject,
            config=config,
            trait_labels=trait_labels,
        )
        cached_score = cache.get(key)

        if cached_score is not None:
            results[i] = cached_score
        else:
            misses.append(text)
            miss_indexes.append(i)
            miss_keys.append(key)

    if misses:
        miss_scores = score_ocean_texts(
            misses,
            subject=subject,
            config=config,
            trait_labels=trait_labels,
        )

        for i, key, score in zip(miss_indexes, miss_keys, miss_scores):
            cache.set(key, score)
            results[i] = score

    return [score for score in results if score is not None]


# =============================================================================
# 7. Aggregation from contexts to cluster-level profile
# =============================================================================


ImportanceFunction = Callable[[str, Optional[str]], float]


@dataclass(frozen=True)
class ScoredContext:
    text: str
    scores: dict[str, float]
    importance: float = 1.0


@dataclass(frozen=True)
class ClusterOceanResult:
    cluster_id: int
    subject: Optional[str]
    scores: dict[str, float]
    scored_contexts: list[ScoredContext]
    n_mentions: int
    n_contexts: int


@dataclass(frozen=True)
class MentionRecord:
    """One extracted mention-level scoring input.

    The model should score context_text. In this rendered-CSV version,
    context_text is the rendered context, not necessarily the literal original
    sentence. The original mention-marked context is retained separately for
    CSV audit/debugging.

    Note:
        ContextConfig.deduplicate is intentionally ignored by
        mention_records_for_cluster(), because mention-level tests must preserve
        one returned row per selected mention.
    """

    cluster_id: int
    subject: Optional[str]
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


@dataclass(frozen=True)
class ScoredMentionRecord:
    """One mention-level scoring output."""

    cluster_id: int
    subject: Optional[str]
    mention_index_in_cluster: int
    mention_id: int
    mention_text: str
    mention_start: int
    mention_end: int
    sentence_index: int
    context_text: str
    normalized_context_text: str
    scores: dict[str, float]
    chunk_index: int
    elapsed_seconds_at_completion: float


def constant_importance(_: str, __: Optional[str] = None) -> float:
    """Default placeholder for future emotional-profile-importance models."""

    return 1.0



def aggregate_ocean_scores(
    scored_contexts: list[ScoredContext],
    *,
    neutral_score: float = 50.0,
) -> dict[str, float]:
    """Weighted average aggregation from context-level scores to one OCEAN profile."""

    if not scored_contexts:
        return {trait: neutral_score for trait in OCEAN_TRAITS}

    total_weight = sum(max(0.0, context.importance) for context in scored_contexts)
    if total_weight <= 1e-9:
        return {trait: neutral_score for trait in OCEAN_TRAITS}

    aggregated: dict[str, float] = {}

    for trait in OCEAN_TRAITS:
        weighted_sum = sum(
            context.scores.get(trait, neutral_score) * max(0.0, context.importance)
            for context in scored_contexts
        )
        aggregated[trait] = round(weighted_sum / total_weight, 2)

    return aggregated



def score_ocean_contexts(
    contexts: list[str],
    *,
    subject: str | None = None,
    config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    cache: SQLiteScoreCache | None = None,
    importance_fn: ImportanceFunction = constant_importance,
) -> list[ScoredContext]:
    """Score already-extracted contexts."""

    config = config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    contexts = unique_preserving_order(contexts)

    scores = score_ocean_texts_cached(
        contexts,
        subject=subject,
        config=config,
        trait_labels=trait_labels,
        cache=cache,
    )

    return [
        ScoredContext(
            text=context,
            scores=score,
            importance=float(importance_fn(context, subject)),
        )
        for context, score in zip(contexts, scores)
    ]



def score_ocean_cluster(
    doc: Any,
    cluster_id: int,
    *,
    subject: str | None = None,
    context_config: ContextConfig | None = None,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    cache_path: str | Path | None = None,
    importance_fn: ImportanceFunction = constant_importance,
) -> ClusterOceanResult:
    """
    End-to-end cluster scoring.

    Steps:
        1. Get mention ids for cluster.
        2. Extract one context per mention.
        3. Deduplicate normalized contexts.
        4. Score each unique context.
        5. Aggregate context scores into one cluster-level OCEAN profile.

    subject:
        If omitted, this function tries to use cluster.canonical_name.
    """

    context_config = context_config or ContextConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS

    mention_ids = mention_ids_for_cluster(doc, cluster_id)
    subject = subject or canonical_name_for_cluster(doc, cluster_id)

    contexts = [
        mention_context_text(doc, mention_id, config=context_config)
        for mention_id in mention_ids
    ]

    if context_config.deduplicate:
        contexts = unique_preserving_order(contexts)

    if cache_path is None:
        scored_contexts = score_ocean_contexts(
            contexts,
            subject=subject,
            config=scoring_config,
            trait_labels=trait_labels,
            cache=None,
            importance_fn=importance_fn,
        )
    else:
        with SQLiteScoreCache(cache_path) as cache:
            scored_contexts = score_ocean_contexts(
                contexts,
                subject=subject,
                config=scoring_config,
                trait_labels=trait_labels,
                cache=cache,
                importance_fn=importance_fn,
            )

    cluster_scores = aggregate_ocean_scores(
        scored_contexts,
        neutral_score=scoring_config.collapse.neutral_score,
    )

    return ClusterOceanResult(
        cluster_id=cluster_id,
        subject=subject,
        scores=cluster_scores,
        scored_contexts=scored_contexts,
        n_mentions=len(mention_ids),
        n_contexts=len(contexts),
    )



# =============================================================================
# 8. Mention-level testing and benchmarking helpers
# =============================================================================


def _sync_cuda_if_available() -> None:
    """Synchronize CUDA timings only when CUDA is active."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()



def _device_name() -> str:
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"

    return "cpu"



def sampled_mention_pairs_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    n_mentions: int | None = None,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
) -> list[tuple[int, int]]:
    """
    Return sampled (mention_index_in_cluster, mention_id) pairs for a cluster.

    Contract:
        - n_mentions=None returns every mention in the cluster
        - n_mentions=int samples exactly n_mentions mentions without replacement
        - mention_index_in_cluster always refers to the original cluster order

    Sorting sampled indexes back into cluster order keeps the CSV easy to audit
    while still selecting rows by random sampling.
    """

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
                f"but n_mentions={n_mentions} was requested. "
                "Use n_mentions=None to score the full cluster, or request fewer mentions."
            )

        rng = random.Random(random_seed)
        selected_indexes = rng.sample(range(total_mentions), n_mentions)

        if sort_sample_by_cluster_order:
            selected_indexes.sort()

    return [(index, int(all_mention_ids[index])) for index in selected_indexes]



def _slice_mention_pairs_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    n_mentions: int | None = None,
    start_index: int = 0,
) -> list[tuple[int, int]]:
    """Return legacy contiguous mention pairs for backwards-compatible slicing."""

    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")

    all_mention_ids = mention_ids_for_cluster(doc, cluster_id)

    if n_mentions is None:
        selected = all_mention_ids[start_index:]
    else:
        selected = all_mention_ids[start_index : start_index + n_mentions]

    return [
        (mention_index, int(mention_id))
        for mention_index, mention_id in enumerate(selected, start=start_index)
    ]



def mention_records_for_cluster(
    doc: Any,
    cluster_id: int,
    *,
    subject: str | None = None,
    n_mentions: int | None = None,
    start_index: int = 0,
    sample_mentions: bool = True,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
) -> list[MentionRecord]:
    """
    Return mention-level records for one cluster.

    This function performs no model inference.

    Default selection policy:
        n_mentions=None
            keep every mention in the cluster.
        n_mentions=int
            sample exactly n_mentions mentions without replacement from the full
            cluster, rather than taking the first n_mentions.

    Backwards-compatible contiguous slicing is still available by setting
    sample_mentions=False. In that mode, start_index is respected.

    Subject policy:
        If subject is omitted, it is read from
        doc._.coref_layer.clusters[cluster_id].canonical_name.

    Important:
        ContextConfig.deduplicate is intentionally ignored here. The contract is
        one returned MentionRecord per selected mention, because benchmarks must
        measure N mentions, not N unique contexts.
    """

    if sample_mentions and start_index != 0:
        raise ValueError(
            "start_index is only meaningful when sample_mentions=False. "
            "Sampling is performed over the full cluster."
        )

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    subject = subject or canonical_name_for_cluster(doc, cluster_id)

    if sample_mentions:
        selected_pairs = sampled_mention_pairs_for_cluster(
            doc,
            cluster_id,
            n_mentions=n_mentions,
            random_seed=random_seed,
            sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        )
    else:
        selected_pairs = _slice_mention_pairs_for_cluster(
            doc,
            cluster_id,
            n_mentions=n_mentions,
            start_index=start_index,
        )

    records: list[MentionRecord] = []

    for mention_index, mention_id in selected_pairs:
        mention = find_mention_by_id(doc, mention_id)
        sentence_index = sentence_index_for_mention(doc, mention_id)
        original_context_text = mention_context_text(doc, mention_id, config=context_config)
        rendered_context_text, render_rule, render_was_changed = rendered_mention_context_text(
            doc,
            mention_id,
            subject=subject,
            context_config=context_config,
            rendering_config=rendering_config,
        )
        normalized_context_text = normalize_context_for_dedup(rendered_context_text)

        records.append(
            MentionRecord(
                cluster_id=cluster_id,
                subject=subject,
                mention_index_in_cluster=mention_index,
                mention_id=int(mention_id),
                mention_text=str(getattr(mention, "text", "")),
                mention_start=int(getattr(mention, "start")),
                mention_end=int(getattr(mention, "end")),
                sentence_index=sentence_index,
                context_text=rendered_context_text,
                normalized_context_text=normalized_context_text,
                original_context_text=original_context_text,
                rendered_context_text=rendered_context_text,
                mention_render_rule=render_rule,
                mention_render_was_changed=render_was_changed,
            )
        )

    return records



def _single_subject_for_records(records: list[MentionRecord]) -> str | None:
    """Return the common subject for a record list, or raise on mixed subjects."""

    subjects = {record.subject for record in records}

    if len(subjects) > 1:
        raise ValueError(
            "score_mention_records() expects records with one common subject. "
            f"Got subjects: {sorted(str(subject) for subject in subjects)}"
        )

    return next(iter(subjects)) if subjects else None



def score_mention_records(
    records: list[MentionRecord],
    *,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    chunk_size: int = 32,
    print_progress: bool = True,
) -> list[ScoredMentionRecord]:
    """
    Score mention records in chunks and return one scored row per mention.

    This function preserves mention-level granularity:
        N MentionRecord objects -> N ScoredMentionRecord objects.

    No aggregation is performed here.
    No deduplication is performed here.
    """

    if not isinstance(records, list):
        raise TypeError(f"records must be list[MentionRecord], got {type(records)!r}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    if not records:
        return []

    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    subject = _single_subject_for_records(records)

    n_records = len(records)
    n_chunks = (n_records + chunk_size - 1) // chunk_size

    if print_progress:
        print("=" * 100)
        print("Mention-level OCEAN scoring")
        print(f"records: {n_records}")
        print(f"subject: {subject!r}")
        print(f"device: {_device_name()}")
        print(f"batch_size: {scoring_config.batch_size}")
        print(f"chunk_size: {chunk_size}")
        print(f"chunks: {n_chunks}")
        print("=" * 100)

    scored_records: list[ScoredMentionRecord] = []
    scoring_start = time.perf_counter()

    for chunk_index, start in enumerate(range(0, n_records, chunk_size), start=1):
        end = min(start + chunk_size, n_records)
        chunk_records = records[start:end]
        chunk_contexts = [record.context_text for record in chunk_records]

        _sync_cuda_if_available()
        chunk_start = time.perf_counter()

        chunk_scores = score_ocean_texts(
            chunk_contexts,
            subject=subject,
            config=scoring_config,
            trait_labels=trait_labels,
        )

        _sync_cuda_if_available()
        chunk_elapsed = time.perf_counter() - chunk_start
        elapsed_so_far = time.perf_counter() - scoring_start

        for record, score in zip(chunk_records, chunk_scores):
            scored_records.append(
                ScoredMentionRecord(
                    cluster_id=record.cluster_id,
                    subject=record.subject,
                    mention_index_in_cluster=record.mention_index_in_cluster,
                    mention_id=record.mention_id,
                    mention_text=record.mention_text,
                    mention_start=record.mention_start,
                    mention_end=record.mention_end,
                    sentence_index=record.sentence_index,
                    context_text=record.context_text,
                    normalized_context_text=record.normalized_context_text,
                    scores=score,
                    chunk_index=chunk_index,
                    elapsed_seconds_at_completion=elapsed_so_far,
                )
            )

        done = len(scored_records)
        seconds_per_mention = elapsed_so_far / done if done else 0.0
        estimated_total = seconds_per_mention * n_records
        estimated_remaining = max(0.0, estimated_total - elapsed_so_far)

        if print_progress:
            print(
                f"[chunk {chunk_index}/{n_chunks}] "
                f"mentions={start}:{end} | "
                f"chunk_time={chunk_elapsed:.2f}s | "
                f"done={done}/{n_records} | "
                f"avg={seconds_per_mention:.3f}s/mention | "
                f"elapsed={elapsed_so_far / 60:.2f}min | "
                f"remaining≈{estimated_remaining / 60:.2f}min"
            )

    return scored_records



def scored_mention_records_to_dataframe(
    scored_records: list[ScoredMentionRecord],
) -> Any:
    """
    Convert scored mention records to a pandas DataFrame.

    pandas is imported lazily so the module can still be imported in environments
    where pandas is not installed, unless this function is called.
    """

    import pandas as pd

    rows: list[dict[str, Any]] = []

    for record in scored_records:
        row: dict[str, Any] = {
            "cluster_id": record.cluster_id,
            "subject": record.subject,
            "mention_index_in_cluster": record.mention_index_in_cluster,
            "mention_id": record.mention_id,
            "mention_text": record.mention_text,
            "mention_start": record.mention_start,
            "mention_end": record.mention_end,
            "sentence_index": record.sentence_index,
            "context_text": record.context_text,
            "normalized_context_text": record.normalized_context_text,
            "chunk_index": record.chunk_index,
            "elapsed_seconds_at_completion": record.elapsed_seconds_at_completion,
        }

        for trait in OCEAN_TRAITS:
            row[trait] = float(record.scores.get(trait, 50.0))

        rows.append(row)

    return pd.DataFrame(rows)



def benchmark_ocean_mention_records(
    doc: Any,
    cluster_id: int,
    *,
    subject: str | None = None,
    n_mentions: int | None = 100,
    start_index: int = 0,
    sample_mentions: bool = True,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    chunk_size: int = 32,
    print_progress: bool = True,
) -> Any:
    """
    Convenience benchmark for scoring mention records from one cluster.

    Default behavior samples n_mentions from the full cluster. Set
    n_mentions=None to score every mention. Set sample_mentions=False to recover
    the old contiguous-slice behavior based on start_index.
    """

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    subject = subject or canonical_name_for_cluster(doc, cluster_id)

    total_mentions_in_cluster = len(mention_ids_for_cluster(doc, cluster_id))

    if print_progress:
        print("=" * 100)
        print("Mention-level OCEAN benchmark")
        print(f"cluster_id: {cluster_id}")
        print(f"subject: {subject!r}")
        print(f"requested mentions: {n_mentions}")
        print(f"sample_mentions: {sample_mentions}")
        print(f"random_seed: {random_seed}")
        print(f"start_index: {start_index}")
        print(f"total mentions in cluster: {total_mentions_in_cluster}")
        print("=" * 100)

    extraction_start = time.perf_counter()

    records = mention_records_for_cluster(
        doc,
        cluster_id,
        subject=subject,
        n_mentions=n_mentions,
        start_index=start_index,
        sample_mentions=sample_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        context_config=context_config,
        rendering_config=rendering_config,
    )

    extraction_elapsed = time.perf_counter() - extraction_start

    if print_progress:
        print(f"extracted records: {len(records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")
        if records:
            print("first extracted contexts:")
            for record in records[:3]:
                print(f"- mention_id={record.mention_id} text={record.mention_text!r}: {record.context_text}")
        print()

    scoring_start = time.perf_counter()

    scored_records = score_mention_records(
        records,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
        chunk_size=chunk_size,
        print_progress=print_progress,
    )

    scoring_elapsed = time.perf_counter() - scoring_start
    total_elapsed = extraction_elapsed + scoring_elapsed

    df = scored_mention_records_to_dataframe(scored_records)

    seconds_per_mention = scoring_elapsed / len(scored_records) if scored_records else 0.0

    df.attrs["benchmark"] = {
        "cluster_id": cluster_id,
        "subject": subject,
        "requested_mentions": n_mentions,
        "scored_mentions": len(scored_records),
        "sample_mentions": sample_mentions,
        "random_seed": random_seed,
        "sort_sample_by_cluster_order": sort_sample_by_cluster_order,
        "start_index": start_index,
        "total_mentions_in_cluster": total_mentions_in_cluster,
        "context_extraction_seconds": extraction_elapsed,
        "scoring_seconds": scoring_elapsed,
        "total_seconds": total_elapsed,
        "seconds_per_mention_scoring_only": seconds_per_mention,
        "estimated_seconds_for_100_mentions": seconds_per_mention * 100,
        "estimated_seconds_for_1000_mentions": seconds_per_mention * 1000,
        "estimated_seconds_for_full_cluster": seconds_per_mention * total_mentions_in_cluster,
        "batch_size": scoring_config.batch_size,
        "chunk_size": chunk_size,
        "device": _device_name(),
    }

    if print_progress:
        print()
        print("=" * 100)
        print("BENCHMARK COMPLETE")
        print("=" * 100)
        print(f"scored mentions: {len(scored_records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")
        print(f"scoring time: {scoring_elapsed:.2f}s")
        print(f"total time: {total_elapsed:.2f}s")
        print(f"seconds per mention, scoring only: {seconds_per_mention:.3f}s")
        print(f"estimated 100 mentions: {(seconds_per_mention * 100) / 60:.2f} min")
        print(f"estimated 1000 mentions: {(seconds_per_mention * 1000) / 60:.2f} min")
        print(f"estimated full cluster: {(seconds_per_mention * total_mentions_in_cluster) / 60:.2f} min")

    return df


# =============================================================================
# 9. Small benchmark helper
# =============================================================================


def benchmark_ocean_scoring(
    texts: list[str],
    *,
    subject: str | None = None,
    scoring_config: OceanScoringConfig | None = None,
) -> None:
    """Simple timing utility for notebook experiments."""

    scoring_config = scoring_config or OceanScoringConfig()
    texts = [validate_text(text) for text in texts]

    print(f"Number of texts: {len(texts)}")
    print(f"Batch size: {scoring_config.batch_size}")
    print(f"Subject-aware: {scoring_config.subject_aware}")
    print(f"Subject: {subject!r}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()
    scores = score_ocean_texts(texts, subject=subject, config=scoring_config)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print()
    print(f"Elapsed time: {elapsed:.2f} seconds")
    if texts:
        seconds_per_text = elapsed / len(texts)
        print(f"Seconds per text: {seconds_per_text:.3f}")
        print(f"Estimated time for 10,000 contexts: {seconds_per_text * 10_000 / 60:.2f} minutes")

    print()
    for text, score in zip(texts, scores):
        print(f"text: {text}")
        print(score)
        print("-" * 100)


# =============================================================================
# 10. CSV-first probability-table scorer
# =============================================================================


OCEAN_WEIGHT_LABELS: dict[str, str] = {
    "high": (
        "strong personality evidence about the character: behavior, emotion, "
        "motivation, decision, social interaction, preference, fear, desire, "
        "moral choice, reaction, or stable tendency"
    ),
    "medium": (
        "weak or ambiguous personality evidence about the character: indirect, "
        "minor, or context-dependent evidence about behavior, emotion, or motivation"
    ),
    "low": (
        "no useful personality evidence about the character: plot movement, "
        "physical description, setting, object description, bookkeeping dialogue, "
        "or events that do not reveal personality"
    ),
}


@dataclass(frozen=True)
class OceanWeightConfig:
    """
    Optional OCEAN_weight configuration.

    enabled:
        If False, no OCEAN_weight hypotheses are scored. The resulting CSV still
        receives OCEAN_weight columns, but OCEAN_weight is constant.

    default_weight:
        Weight assigned when enabled=False. Keep this on a 0-100 scale.

    hypothesis_template:
        Template for OCEAN_weight labels. It must preserve one literal {}
        placeholder for the candidate label. {subject}, when present, is replaced
        manually.
    """

    enabled: bool = False
    hypothesis_template: str = (
        "In this text, {subject}'s behavior or inner state provides {}."
    )
    default_weight: float = 100.0
    rounding_digits: int = 2


@dataclass(frozen=True)
class DirectNLIConfig:
    """
    Configuration for the lower-level NLI scorer.

    pair_batch_size:
        Number of premise-hypothesis pairs processed per forward pass. On a 3 GB
        GTX 1050 Max-Q, start conservatively with 16 or 32.
    """

    pair_batch_size: int = 16
    truncation: bool = True
    max_length: Optional[int] = None


class SQLiteProbabilityCache:
    """
    SQLite cache for resumable direct-NLI probability rows.

    One row stores only the model-dependent probability payload for one normalized
    context under one scoring configuration. Mention metadata is reconstructed
    when the CSV row is written, so repeated contexts and crash restarts do not
    need another model forward pass.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ocean_probability_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at_unix REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        row = self.connection.execute(
            "SELECT payload_json FROM ocean_probability_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def set_many(self, items: Iterable[tuple[str, dict[str, Any]]]) -> None:
        now = time.time()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO ocean_probability_cache(
                cache_key, payload_json, created_at_unix
            )
            VALUES (?, ?, ?)
            """,
            [
                (key, _stable_json(_json_ready(payload)), now)
                for key, payload in items
            ],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteProbabilityCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _probability_cache_key(
    record: "MentionRecord",
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: "OceanWeightConfig",
    nli_config: DirectNLIConfig,
) -> str:
    """Return a stable cache key for the model-dependent probability payload."""

    payload = {
        "schema_version": PROBABILITY_CACHE_SCHEMA_VERSION,
        "normalized_context_text": normalize_context_for_dedup(record.context_text),
        "subject": subject,
        "scoring_config": {
            "model_name": scoring_config.model_name,
            "generic_hypothesis_template": scoring_config.generic_hypothesis_template,
            "subject_hypothesis_template": scoring_config.subject_hypothesis_template,
            "subject_aware": scoring_config.subject_aware,
            "truncation": scoring_config.truncation,
            "multi_label": scoring_config.multi_label,
            "collapse": scoring_config.collapse,
        },
        "trait_labels": trait_labels,
        "weight_config": {
            "enabled": weight_config.enabled,
            "hypothesis_template": weight_config.hypothesis_template,
            "default_weight": weight_config.default_weight,
            "rounding_digits": weight_config.rounding_digits,
            "labels": OCEAN_WEIGHT_LABELS if weight_config.enabled else None,
        },
        # pair_batch_size intentionally does not participate in the cache key: it
        # changes memory/performance, not the mathematical output.
        "nli_config": {
            "truncation": nli_config.truncation,
            "max_length": nli_config.max_length,
        },
    }

    return hashlib.sha256(_stable_json(_json_ready(payload)).encode("utf-8")).hexdigest()


def _probability_run_config_fingerprint(
    *,
    cluster_id: int,
    subject: str | None,
    selected_mention_ids: list[int],
    n_mentions: int | None,
    sample_mentions: bool,
    random_seed: int | None,
    sort_sample_by_cluster_order: bool,
    context_config: ContextConfig,
    rendering_config: MentionRenderingConfig,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: "OceanWeightConfig",
    nli_config: DirectNLIConfig,
) -> str:
    """Fingerprint the run shape so resume mode does not mix incompatible CSVs."""

    payload = {
        "schema_version": PROBABILITY_CACHE_SCHEMA_VERSION,
        "cluster_id": cluster_id,
        "subject": subject,
        "selected_mention_ids": selected_mention_ids,
        "n_mentions": n_mentions,
        "sample_mentions": sample_mentions,
        "random_seed": random_seed,
        "sort_sample_by_cluster_order": sort_sample_by_cluster_order,
        "context_config": context_config,
        "rendering_config": rendering_config,
        "scoring_config": scoring_config,
        "trait_labels": trait_labels,
        "weight_config": {
            "config": weight_config,
            "labels": OCEAN_WEIGHT_LABELS if weight_config.enabled else None,
        },
        "nli_config": {
            "truncation": nli_config.truncation,
            "max_length": nli_config.max_length,
        },
    }

    return hashlib.sha256(_stable_json(_json_ready(payload)).encode("utf-8")).hexdigest()


def _completed_mention_ids_from_csv(csv_path: str | Path) -> set[int]:
    """Read completed mention_id values from an existing partial CSV."""

    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    completed: set[int] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames or "mention_id" not in reader.fieldnames:
            raise ValueError(
                f"Cannot resume from {csv_path}: CSV has no mention_id column."
            )

        for row in reader:
            raw_mention_id = row.get("mention_id")
            if raw_mention_id in (None, ""):
                continue

            # Treat a row as completed only if all final collapsed OCEAN columns
            # are present. This avoids skipping a partially/corruptly written row.
            if any(row.get(trait) in (None, "") for trait in OCEAN_TRAITS):
                continue

            completed.add(int(float(raw_mention_id)))

    return completed


def _count_csv_data_rows(csv_path: str | Path) -> int:
    """Count data rows in a CSV without materializing it as a DataFrame."""

    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        # subtract one header row
        return max(0, sum(1 for _ in handle) - 1)


def _release_chunk_memory() -> None:
    """Best-effort cleanup after each chunk on RAM-constrained machines."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class ProbabilityTask:
    """One grouped softmax task: one text competes over this task's labels."""

    task_name: str
    label_texts: dict[str, str]
    hypothesis_template: str


@lru_cache(maxsize=4)
def _load_direct_nli_components(model_name: str) -> tuple[Any, Any, torch.device, int]:
    """
    Load tokenizer/model for direct NLI scoring.

    This bypasses the zero-shot pipeline and directly scores premise-hypothesis
    pairs. For multi_label=False zero-shot behavior, we use the entailment logit
    for each candidate hypothesis and softmax across the labels of one task.
    """

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    entailment_id = _find_entailment_id(model)

    return tokenizer, model, device, entailment_id


def _find_entailment_id(model: Any) -> int:
    """Best-effort extraction of the entailment class id from model config."""

    label2id = getattr(model.config, "label2id", {}) or {}

    for label, idx in label2id.items():
        if "entail" in str(label).lower():
            return int(idx)

    id2label = getattr(model.config, "id2label", {}) or {}

    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)

    num_labels = int(getattr(model.config, "num_labels", 3))
    if num_labels >= 3:
        return 2

    raise ValueError(
        "Could not identify entailment label id from model config. "
        f"label2id={label2id!r}, id2label={id2label!r}"
    )


def _replace_subject_in_template(template: str, subject: str | None) -> str:
    """Replace only {subject}; preserve the positional {} placeholder."""

    if "{subject}" in template:
        return template.replace("{subject}", subject or "the character")

    return template


def _format_candidate_hypothesis(template: str, label_text: str) -> str:
    """Inject one candidate label into a zero-shot hypothesis template."""

    try:
        return template.format(label_text)
    except Exception as exc:
        raise ValueError(
            "Hypothesis template must contain exactly one positional {} "
            f"placeholder after subject replacement. Got: {template!r}"
        ) from exc


def _trait_probability_tasks(
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
) -> list[ProbabilityTask]:
    """Build one grouped three-way softmax task per OCEAN trait."""

    hypothesis_template = _hypothesis_template(scoring_config, subject)

    return [
        ProbabilityTask(
            task_name=trait,
            label_texts={
                "positive": labels["positive"],
                "negative": labels["negative"],
                "neutral": labels["neutral"],
            },
            hypothesis_template=hypothesis_template,
        )
        for trait, labels in trait_labels.items()
    ]


def _weight_probability_task(
    *,
    subject: str | None,
    weight_config: OceanWeightConfig,
) -> ProbabilityTask:
    """Build the grouped high/medium/low softmax task for OCEAN_weight."""

    return ProbabilityTask(
        task_name="OCEAN_weight",
        label_texts={
            "high": OCEAN_WEIGHT_LABELS["high"],
            "medium": OCEAN_WEIGHT_LABELS["medium"],
            "low": OCEAN_WEIGHT_LABELS["low"],
        },
        hypothesis_template=_replace_subject_in_template(
            weight_config.hypothesis_template,
            subject,
        ),
    )


def _direct_entailment_logits_for_pairs(
    pairs: list[tuple[str, str]],
    *,
    model_name: str,
    nli_config: DirectNLIConfig,
) -> list[float]:
    """Return one entailment logit for each premise-hypothesis pair."""

    if nli_config.pair_batch_size <= 0:
        raise ValueError(
            f"pair_batch_size must be > 0, got {nli_config.pair_batch_size}"
        )

    if not pairs:
        return []

    tokenizer, model, device, entailment_id = _load_direct_nli_components(model_name)
    logits: list[float] = []

    for start in range(0, len(pairs), nli_config.pair_batch_size):
        batch = pairs[start : start + nli_config.pair_batch_size]
        premises = [premise for premise, _ in batch]
        hypotheses = [hypothesis for _, hypothesis in batch]

        tokenizer_kwargs: dict[str, Any] = {
            "text": premises,
            "text_pair": hypotheses,
            "return_tensors": "pt",
            "padding": True,
            "truncation": nli_config.truncation,
        }
        if nli_config.max_length is not None:
            tokenizer_kwargs["max_length"] = nli_config.max_length

        encoded = tokenizer(**tokenizer_kwargs)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            output = model(**encoded)
            batch_logits = output.logits[:, entailment_id].detach().float().cpu()

        logits.extend(float(value) for value in batch_logits.tolist())

        # Keep peak memory lower during long runs. The model stays loaded, but
        # temporary tensors, token batches, and CPU lists are released per batch.
        del batch, premises, hypotheses, encoded, output, batch_logits
        _release_chunk_memory()

    return logits


def _probability_payloads_from_logits(
    records: list[MentionRecord],
    *,
    grouped_logits: dict[tuple[int, str], dict[str, float]],
    scoring_config: OceanScoringConfig,
    weight_config: OceanWeightConfig,
) -> list[dict[str, Any]]:
    """Convert grouped direct-NLI logits into compact cacheable payloads."""

    payloads: list[dict[str, Any]] = []

    for record_index, _record in enumerate(records):
        payload: dict[str, Any] = {}

        for trait in OCEAN_TRAITS:
            label_logits = grouped_logits[(record_index, trait)]
            ordered_keys = ["positive", "negative", "neutral"]
            logits_tensor = torch.tensor(
                [label_logits[key] for key in ordered_keys],
                dtype=torch.float32,
            )
            probabilities = torch.softmax(logits_tensor, dim=0).tolist()
            probability_by_key = dict(zip(ordered_keys, probabilities))

            positive = float(probability_by_key["positive"])
            negative = float(probability_by_key["negative"])
            neutral = float(probability_by_key["neutral"])

            payload[f"{trait}_positive_probability"] = positive
            payload[f"{trait}_negative_probability"] = negative
            payload[f"{trait}_neutral_probability"] = neutral
            payload[trait] = collapse_trait_scores(
                positive=positive,
                negative=negative,
                neutral=neutral,
                config=scoring_config.collapse,
            )

            del logits_tensor

        if weight_config.enabled:
            label_logits = grouped_logits[(record_index, "OCEAN_weight")]
            ordered_keys = ["high", "medium", "low"]
            logits_tensor = torch.tensor(
                [label_logits[key] for key in ordered_keys],
                dtype=torch.float32,
            )
            probabilities = torch.softmax(logits_tensor, dim=0).tolist()
            probability_by_key = dict(zip(ordered_keys, probabilities))

            high = float(probability_by_key["high"])
            medium = float(probability_by_key["medium"])
            low = float(probability_by_key["low"])
            ocean_weight = round(
                100.0 * high + 50.0 * medium,
                weight_config.rounding_digits,
            )

            payload["OCEAN_weight_high_probability"] = high
            payload["OCEAN_weight_medium_probability"] = medium
            payload["OCEAN_weight_low_probability"] = low
            payload["OCEAN_weight"] = ocean_weight
            payload["OCEAN_weight_source"] = "direct_nli_zero_shot"

            del logits_tensor
        else:
            payload["OCEAN_weight_high_probability"] = None
            payload["OCEAN_weight_medium_probability"] = None
            payload["OCEAN_weight_low_probability"] = None
            payload["OCEAN_weight"] = float(weight_config.default_weight)
            payload["OCEAN_weight_source"] = "constant_disabled"

        payloads.append(payload)

    return payloads


def _score_probability_payloads_for_chunk(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: DirectNLIConfig,
) -> list[dict[str, Any]]:
    """Score one mention chunk and return cacheable probability payloads."""

    tasks = _trait_probability_tasks(
        subject=subject,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
    )

    if weight_config.enabled:
        tasks.append(_weight_probability_task(subject=subject, weight_config=weight_config))

    pair_metadata: list[tuple[int, str, str]] = []
    pairs: list[tuple[str, str]] = []

    for record_index, record in enumerate(records):
        for task in tasks:
            for label_key, label_text in task.label_texts.items():
                hypothesis = _format_candidate_hypothesis(
                    task.hypothesis_template,
                    label_text,
                )
                pairs.append((record.context_text, hypothesis))
                pair_metadata.append((record_index, task.task_name, label_key))

    entailment_logits = _direct_entailment_logits_for_pairs(
        pairs,
        model_name=scoring_config.model_name,
        nli_config=nli_config,
    )

    grouped_logits: dict[tuple[int, str], dict[str, float]] = {}

    for (record_index, task_name, label_key), logit in zip(pair_metadata, entailment_logits):
        grouped_logits.setdefault((record_index, task_name), {})[label_key] = logit

    payloads = _probability_payloads_from_logits(
        records,
        grouped_logits=grouped_logits,
        scoring_config=scoring_config,
        weight_config=weight_config,
    )

    del tasks, pair_metadata, pairs, entailment_logits, grouped_logits
    _release_chunk_memory()

    return payloads


def _probability_payload_to_row(
    record: MentionRecord,
    payload: dict[str, Any],
    *,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> dict[str, Any]:
    """Combine mention metadata with cached/scored probability payload."""

    row: dict[str, Any] = {
        "cluster_id": record.cluster_id,
        "subject": record.subject,
        "mention_index_in_cluster": record.mention_index_in_cluster,
        "mention_id": record.mention_id,
        "mention_text": record.mention_text,
        "mention_start": record.mention_start,
        "mention_end": record.mention_end,
        "sentence_index": record.sentence_index,
        "mention_render_rule": record.mention_render_rule,
        "mention_render_was_changed": record.mention_render_was_changed,
        "original_context_text": record.original_context_text,
        "rendered_context_text": record.rendered_context_text or record.context_text,
        "context_text": record.context_text,
        "normalized_context_text": record.normalized_context_text,
        "chunk_index": chunk_index,
        "elapsed_seconds_at_completion": elapsed_seconds_at_completion,
    }
    row.update(payload)
    return row


def _score_probability_rows_for_chunk_cached(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: DirectNLIConfig,
    cache: SQLiteProbabilityCache | None,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Return CSV rows plus (cache_hits, cache_misses).

    Only cache misses are sent through the model. Cached payloads are still
    rewritten as fresh CSV rows with the current mention metadata.
    """

    if cache is None:
        payloads = _score_probability_payloads_for_chunk(
            records,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        rows = [
            _probability_payload_to_row(
                record,
                payload,
                chunk_index=chunk_index,
                elapsed_seconds_at_completion=elapsed_seconds_at_completion,
            )
            for record, payload in zip(records, payloads)
        ]
        return rows, 0, len(records)

    payloads_by_index: list[dict[str, Any] | None] = [None] * len(records)
    miss_records: list[MentionRecord] = []
    miss_indexes: list[int] = []
    miss_keys: list[str] = []
    cache_hits = 0

    for record_index, record in enumerate(records):
        key = _probability_cache_key(
            record,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        cached_payload = cache.get(key)

        if cached_payload is None:
            miss_records.append(record)
            miss_indexes.append(record_index)
            miss_keys.append(key)
        else:
            payloads_by_index[record_index] = cached_payload
            cache_hits += 1

    if miss_records:
        miss_payloads = _score_probability_payloads_for_chunk(
            miss_records,
            subject=subject,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
        )
        cache.set_many(zip(miss_keys, miss_payloads))

        for record_index, payload in zip(miss_indexes, miss_payloads):
            payloads_by_index[record_index] = payload

    rows = [
        _probability_payload_to_row(
            record,
            payload,
            chunk_index=chunk_index,
            elapsed_seconds_at_completion=elapsed_seconds_at_completion,
        )
        for record, payload in zip(records, payloads_by_index)
        if payload is not None
    ]

    cache_misses = len(miss_records)
    del payloads_by_index, miss_records, miss_indexes, miss_keys
    _release_chunk_memory()

    return rows, cache_hits, cache_misses


def _score_probability_rows_for_chunk(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: DirectNLIConfig,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> list[dict[str, Any]]:
    """
    Score one mention chunk and return wide rows with raw probabilities.

    Kept for backwards compatibility. New marathon/resume code calls
    _score_probability_rows_for_chunk_cached(), which uses the same payload
    computation plus an optional SQLite cache.
    """

    rows, _cache_hits, _cache_misses = _score_probability_rows_for_chunk_cached(
        records,
        subject=subject,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
        weight_config=weight_config,
        nli_config=nli_config,
        cache=None,
        chunk_index=chunk_index,
        elapsed_seconds_at_completion=elapsed_seconds_at_completion,
    )
    return rows


def benchmark_ocean_mention_probability_csv(
    doc: Any,
    cluster_id: int,
    *,
    csv_path: str | Path,
    subject: str | None = None,
    n_mentions: int | None = 100,
    start_index: int = 0,
    sample_mentions: bool = True,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    weight_config: OceanWeightConfig | None = None,
    nli_config: DirectNLIConfig | None = None,
    chunk_size: int = 16,
    overwrite_csv: bool = False,
    resume_from_csv: bool = True,
    use_sqlite_cache: bool = True,
    cache_path: str | Path | None = None,
    keep_rows_in_memory: bool = False,
    return_dataframe: bool = True,
    print_progress: bool = True,
) -> Any:
    """
    CSV-first mention-level OCEAN benchmark using rendered mentions and direct NLI scoring.

    Marathon/resume behavior:
        - overwrite_csv=False by default, so an existing CSV is treated as partial work.
        - resume_from_csv=True skips mention_ids already completed in that CSV.
        - use_sqlite_cache=True stores model probability payloads in SQLite. If the
          PC crashes after a chunk was scored but before all CSV work is complete,
          reruns can reuse cached model outputs instead of recomputing them.
        - keep_rows_in_memory=False avoids accumulating all chunk rows in RAM.

    Contract:
        INPUT:
            doc + cluster_id + number of mentions N

        OUTPUT:
            pandas.DataFrame and a CSV on disk by default. For very large runs,
            set return_dataframe=False to avoid loading the final CSV into RAM.

    Default selection policy:
        - n_mentions=None scores every mention in the cluster
        - n_mentions=int samples that many mentions from the full cluster

    Subject policy:
        If subject is omitted, it is read from
        doc._.coref_layer.clusters[cluster_id].canonical_name.

    The CSV always includes raw probability columns:
        - *_positive_probability
        - *_negative_probability
        - *_neutral_probability
        - OCEAN_weight_*_probability

    No doc annotation is performed here.
    """

    import pandas as pd

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")

    if sample_mentions and start_index != 0:
        raise ValueError(
            "start_index is only meaningful when sample_mentions=False. "
            "Sampling is performed over the full cluster."
        )

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")

    if cache_path is None:
        cache_path = csv_path.with_suffix(csv_path.suffix + ".cache.sqlite3")
    cache_path = Path(cache_path)

    if overwrite_csv:
        if csv_path.exists():
            csv_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    weight_config = weight_config or OceanWeightConfig(enabled=False)
    nli_config = nli_config or DirectNLIConfig()
    subject = subject or canonical_name_for_cluster(doc, cluster_id)

    total_mentions_in_cluster = len(mention_ids_for_cluster(doc, cluster_id))

    if print_progress:
        print("=" * 100)
        print("CSV-first OCEAN probability benchmark")
        print(f"cluster_id: {cluster_id}")
        print(f"subject: {subject!r}")
        print(f"requested mentions: {n_mentions}")
        print(f"sample_mentions: {sample_mentions}")
        print(f"random_seed: {random_seed}")
        print(f"start_index: {start_index}")
        print(f"total mentions in cluster: {total_mentions_in_cluster}")
        print(f"device: {_device_name()}")
        print(f"chunk_size: {chunk_size}")
        print(f"pair_batch_size: {nli_config.pair_batch_size}")
        print(f"OCEAN_weight enabled: {weight_config.enabled}")
        print(f"canonicalize_simple_mentions: {rendering_config.canonicalize_simple_mentions}")
        print(f"keep_first_second_person: {rendering_config.keep_first_second_person}")
        print(f"overwrite_csv: {overwrite_csv}")
        print(f"resume_from_csv: {resume_from_csv}")
        print(f"use_sqlite_cache: {use_sqlite_cache}")
        print(f"keep_rows_in_memory: {keep_rows_in_memory}")
        print(f"return_dataframe: {return_dataframe}")
        print(f"csv_path: {csv_path}")
        print(f"cache_path: {cache_path if use_sqlite_cache else None}")
        print("=" * 100)

    extraction_start = time.perf_counter()
    records = mention_records_for_cluster(
        doc,
        cluster_id,
        subject=subject,
        n_mentions=n_mentions,
        start_index=start_index,
        sample_mentions=sample_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        context_config=context_config,
        rendering_config=rendering_config,
    )
    extraction_elapsed = time.perf_counter() - extraction_start

    selected_mention_ids = [record.mention_id for record in records]
    run_config_hash = _probability_run_config_fingerprint(
        cluster_id=cluster_id,
        subject=subject,
        selected_mention_ids=selected_mention_ids,
        n_mentions=n_mentions,
        sample_mentions=sample_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        context_config=context_config,
        rendering_config=rendering_config,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
        weight_config=weight_config,
        nli_config=nli_config,
    )

    if csv_path.exists() and csv_path.stat().st_size > 0 and not overwrite_csv:
        if metadata_path.exists():
            try:
                existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot safely resume because metadata is corrupt: {metadata_path}"
                ) from exc

            existing_hash = existing_metadata.get("run_config_hash")
            if existing_hash and existing_hash != run_config_hash:
                raise ValueError(
                    "Refusing to resume into an existing CSV produced by a different run configuration. "
                    f"CSV: {csv_path}. Metadata: {metadata_path}. "
                    "Use overwrite_csv=True or choose a different csv_path."
                )
        elif resume_from_csv and print_progress:
            print(
                "WARNING: existing CSV has no metadata sidecar. "
                "Resuming by mention_id only; verify manually that the CSV came from the same config."
            )

    completed_mention_ids: set[int] = set()
    if resume_from_csv and csv_path.exists() and not overwrite_csv:
        completed_mention_ids = _completed_mention_ids_from_csv(csv_path)

    if completed_mention_ids:
        before_resume_filter = len(records)
        records = [
            record for record in records
            if record.mention_id not in completed_mention_ids
        ]
        skipped_from_existing_csv = before_resume_filter - len(records)
    else:
        skipped_from_existing_csv = 0

    if print_progress:
        print(f"extracted records for this run: {len(selected_mention_ids)}")
        print(f"already completed in CSV: {skipped_from_existing_csv}")
        print(f"records left to score/write: {len(records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")
        if records:
            print("first records still pending:")
            for record in records[:3]:
                print(
                    f"- mention_id={record.mention_id} "
                    f"text={record.mention_text!r} "
                    f"rule={record.mention_render_rule}: {record.context_text}"
                )
        print()

    scoring_start = time.perf_counter()
    n_records_to_score = len(records)
    n_chunks = (n_records_to_score + chunk_size - 1) // chunk_size if n_records_to_score else 0
    csv_header_written = csv_path.exists() and csv_path.stat().st_size > 0 and not overwrite_csv
    kept_rows: list[dict[str, Any]] | None = [] if keep_rows_in_memory else None
    newly_written_rows = 0
    total_cache_hits = 0
    total_cache_misses = 0

    initial_metadata = {
        "status": "running",
        "cluster_id": cluster_id,
        "subject": subject,
        "requested_mentions": n_mentions,
        "selected_mentions": len(selected_mention_ids),
        "pending_mentions_at_start": n_records_to_score,
        "skipped_from_existing_csv": skipped_from_existing_csv,
        "run_config_hash": run_config_hash,
        "csv_path": str(csv_path),
        "cache_path": str(cache_path) if use_sqlite_cache else None,
        "output_is_resumable": True,
        "resume_from_csv": resume_from_csv,
        "use_sqlite_cache": use_sqlite_cache,
        "started_at_unix": time.time(),
    }
    _write_json_atomic(metadata_path, initial_metadata)

    cache_manager: SQLiteProbabilityCache | None
    cache_manager = SQLiteProbabilityCache(cache_path) if use_sqlite_cache else None

    try:
        for chunk_index, start in enumerate(range(0, n_records_to_score, chunk_size), start=1):
            end = min(start + chunk_size, n_records_to_score)
            chunk_records = records[start:end]

            _sync_cuda_if_available()
            chunk_start = time.perf_counter()

            elapsed_before_chunk_completion = time.perf_counter() - scoring_start
            chunk_rows, cache_hits, cache_misses = _score_probability_rows_for_chunk_cached(
                chunk_records,
                subject=subject,
                scoring_config=scoring_config,
                trait_labels=trait_labels,
                weight_config=weight_config,
                nli_config=nli_config,
                cache=cache_manager,
                chunk_index=chunk_index,
                elapsed_seconds_at_completion=elapsed_before_chunk_completion,
            )

            _sync_cuda_if_available()
            chunk_elapsed = time.perf_counter() - chunk_start
            elapsed_so_far = time.perf_counter() - scoring_start

            # Correct the elapsed timestamp after the chunk is actually complete.
            for row in chunk_rows:
                row["elapsed_seconds_at_completion"] = elapsed_so_far

            chunk_df = pd.DataFrame(chunk_rows)
            chunk_df.to_csv(
                csv_path,
                mode="a",
                header=not csv_header_written,
                index=False,
                encoding="utf-8",
            )
            csv_header_written = True

            if kept_rows is not None:
                kept_rows.extend(chunk_rows)

            newly_written_rows += len(chunk_rows)
            total_cache_hits += cache_hits
            total_cache_misses += cache_misses

            done_this_invocation = newly_written_rows
            done_total_selected = skipped_from_existing_csv + newly_written_rows
            seconds_per_mention = elapsed_so_far / done_this_invocation if done_this_invocation else 0.0
            estimated_total = seconds_per_mention * n_records_to_score
            estimated_remaining = max(0.0, estimated_total - elapsed_so_far)

            progress_metadata = dict(initial_metadata)
            progress_metadata.update(
                {
                    "status": "running",
                    "last_completed_chunk": chunk_index,
                    "newly_written_rows": newly_written_rows,
                    "completed_mentions_total_selected": done_total_selected,
                    "cache_hits": total_cache_hits,
                    "cache_misses": total_cache_misses,
                    "elapsed_seconds_scoring_this_invocation": elapsed_so_far,
                    "updated_at_unix": time.time(),
                }
            )
            _write_json_atomic(metadata_path, progress_metadata)

            if print_progress:
                n_tasks = len(OCEAN_TRAITS) + (1 if weight_config.enabled else 0)
                n_pairs = cache_misses * n_tasks * 3
                print(
                    f"[chunk {chunk_index}/{n_chunks}] "
                    f"mentions={start}:{end} | "
                    f"model_pairs={n_pairs} | "
                    f"cache_hits={cache_hits} | "
                    f"cache_misses={cache_misses} | "
                    f"chunk_time={chunk_elapsed:.2f}s | "
                    f"done_this_run={done_this_invocation}/{n_records_to_score} | "
                    f"done_total={done_total_selected}/{len(selected_mention_ids)} | "
                    f"avg={seconds_per_mention:.3f}s/mention | "
                    f"elapsed={elapsed_so_far / 60:.2f}min | "
                    f"remaining≈{estimated_remaining / 60:.2f}min | "
                    f"csv_saved=True"
                )

            del chunk_df, chunk_rows, chunk_records
            _release_chunk_memory()
    finally:
        if cache_manager is not None:
            cache_manager.close()

    scoring_elapsed = time.perf_counter() - scoring_start
    total_elapsed = extraction_elapsed + scoring_elapsed
    seconds_per_mention = scoring_elapsed / newly_written_rows if newly_written_rows else 0.0
    csv_rows_total = _count_csv_data_rows(csv_path)

    benchmark_metadata = {
        "status": "complete",
        "cluster_id": cluster_id,
        "subject": subject,
        "requested_mentions": n_mentions,
        "selected_mentions": len(selected_mention_ids),
        "newly_scored_or_written_mentions": newly_written_rows,
        "scored_mentions_in_csv": csv_rows_total,
        "skipped_from_existing_csv": skipped_from_existing_csv,
        "sample_mentions": sample_mentions,
        "random_seed": random_seed,
        "sort_sample_by_cluster_order": sort_sample_by_cluster_order,
        "start_index": start_index,
        "total_mentions_in_cluster": total_mentions_in_cluster,
        "context_extraction_seconds": extraction_elapsed,
        "scoring_seconds_this_invocation": scoring_elapsed,
        "total_seconds_this_invocation": total_elapsed,
        "seconds_per_mention_scoring_only_this_invocation": seconds_per_mention,
        "estimated_seconds_for_100_mentions": seconds_per_mention * 100,
        "estimated_seconds_for_1000_mentions": seconds_per_mention * 1000,
        "estimated_seconds_for_full_cluster": seconds_per_mention * total_mentions_in_cluster,
        "chunk_size": chunk_size,
        "pair_batch_size": nli_config.pair_batch_size,
        "device": _device_name(),
        "model_name": scoring_config.model_name,
        "ocean_weight_enabled": weight_config.enabled,
        "mention_rendering": {
            "canonicalize_simple_mentions": rendering_config.canonicalize_simple_mentions,
            "keep_first_second_person": rendering_config.keep_first_second_person,
        },
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "cache_path": str(cache_path) if use_sqlite_cache else None,
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "run_config_hash": run_config_hash,
        "raw_probabilities_in_csv": True,
        "doc_annotation_performed": False,
        "output_is_resumable": True,
        "resume_from_csv": resume_from_csv,
        "use_sqlite_cache": use_sqlite_cache,
    }

    _write_json_atomic(metadata_path, benchmark_metadata)

    if keep_rows_in_memory:
        df = pd.DataFrame(kept_rows or [])
    elif return_dataframe and csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    df.attrs["benchmark"] = benchmark_metadata

    # Release the extraction records after the CSV/DataFrame has been produced.
    del records
    if kept_rows is not None:
        del kept_rows
    _release_chunk_memory()

    if print_progress:
        print()
        print("=" * 100)
        print("CSV-FIRST BENCHMARK COMPLETE")
        print("=" * 100)
        print(f"new rows written this invocation: {newly_written_rows}")
        print(f"rows currently in CSV: {csv_rows_total}")
        print(f"skipped from existing CSV: {skipped_from_existing_csv}")
        print(f"cache hits: {total_cache_hits}")
        print(f"cache misses: {total_cache_misses}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")
        print(f"scoring/writing time this invocation: {scoring_elapsed:.2f}s")
        print(f"total time this invocation: {total_elapsed:.2f}s")
        print(f"seconds per newly written mention: {seconds_per_mention:.3f}s")
        print(f"estimated 100 new mentions: {(seconds_per_mention * 100) / 60:.2f} min")
        print(f"estimated 1000 new mentions: {(seconds_per_mention * 1000) / 60:.2f} min")
        print(f"estimated full cluster: {(seconds_per_mention * total_mentions_in_cluster) / 60:.2f} min")
        print(f"csv saved to: {csv_path}")
        print(f"sqlite cache saved to: {cache_path if use_sqlite_cache else None}")
        print(f"metadata saved to: {metadata_path}")

    return df



def _safe_filename_component(value: Any, *, default: str = "unknown") -> str:
    """Return a filesystem-safe filename component."""

    text = str(value if value is not None else default).strip() or default
    text = normalize_context_for_dedup(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default



def _cluster_random_seed(base_seed: int | None, cluster_id: int) -> int | None:
    """
    Derive a stable per-cluster seed from a base seed.

    This avoids accidentally sampling the same relative mention positions across
    clusters when the same random seed is reused for a multi-cluster run.
    """

    if base_seed is None:
        return None

    payload = f"{int(base_seed)}:{int(cluster_id)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)



def default_cluster_csv_path(
    output_dir: str | Path | None = None,
    *,
    cluster_id: int,
    subject: str,
    n_mentions: int | None,
) -> Path:
    """Build the default output CSV path for one cluster."""

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OCEAN_PROFILES_OUTPUT_DIR
    subject_part = _safe_filename_component(subject, default="unknown_subject")
    n_part = "all" if n_mentions is None else str(n_mentions)
    return output_dir / f"OCEAN_scores_cluster_{cluster_id}_{subject_part}_{n_part}.csv"



def benchmark_ocean_clusters_probability_csvs(
    doc: Any,
    cluster_ids: list[int],
    n_mentions: int | None,
    *,
    output_dir: str | Path | None = None,
    context_config: ContextConfig | None = None,
    rendering_config: MentionRenderingConfig | None = None,
    scoring_config: OceanScoringConfig | None = None,
    trait_labels: dict[str, dict[str, str]] | None = None,
    weight_config: OceanWeightConfig | None = None,
    nli_config: DirectNLIConfig | None = None,
    chunk_size: int = 16,
    overwrite_csv: bool = False,
    resume_from_csv: bool = True,
    use_sqlite_cache: bool = True,
    keep_rows_in_memory: bool = False,
    return_dataframes: bool = False,
    random_seed: int | None = None,
    sort_sample_by_cluster_order: bool = True,
    print_progress: bool = True,
) -> dict[int, Any]:
    """
    CSV-first multi-cluster OCEAN scoring with sampled mentions.

    Marathon defaults:
        - output_dir defaults to ./outputs/OCEAN_profiles
        - overwrite_csv=False
        - resume_from_csv=True
        - use_sqlite_cache=True
        - return_dataframes=False, so large multi-cluster runs do not keep every
          cluster DataFrame in RAM. Set return_dataframes=True for notebook
          inspection on smaller runs.

    Contract:
        INPUT:
            doc: spaCy-like document with doc._.coref_layer
            cluster_ids: list[int]
            n_mentions: int | None

        OUTPUT:
            One CSV per cluster_id, saved under output_dir.
            The return value is {cluster_id: pandas.DataFrame}. When
            return_dataframes=False, each DataFrame is intentionally empty but
            still carries run metadata in df.attrs["benchmark"].

    Name/subject policy:
        The subject for each cluster is read directly from
        doc._.coref_layer.clusters[cluster_id].canonical_name. No manual
        cluster_id -> subject dictionary is required or accepted.

    Selection policy:
        - n_mentions=None: score every mention in each cluster
        - n_mentions=int: sample exactly n_mentions mentions without replacement
          from all mentions in each cluster

    CSV schema:
        The per-cluster CSVs use the same columns as
        benchmark_ocean_mention_probability_csv(). Sampling metadata is written
        to the sidecar *.metadata.json files, not added as extra CSV columns.
    """

    if not isinstance(cluster_ids, list):
        raise TypeError(f"cluster_ids must be list[int], got {type(cluster_ids)!r}")

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OCEAN_PROFILES_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    weight_config = weight_config or OceanWeightConfig(enabled=False)
    nli_config = nli_config or DirectNLIConfig()

    if print_progress:
        print("=" * 100)
        print("CSV-first sampled multi-cluster OCEAN run")
        print(f"cluster_ids: {cluster_ids}")
        print(f"requested mentions per cluster: {n_mentions}")
        print(f"output_dir: {output_dir}")
        print(f"overwrite_csv: {overwrite_csv}")
        print(f"resume_from_csv: {resume_from_csv}")
        print(f"use_sqlite_cache: {use_sqlite_cache}")
        print(f"return_dataframes: {return_dataframes}")
        print(f"random_seed: {random_seed}")
        print(f"sort_sample_by_cluster_order: {sort_sample_by_cluster_order}")
        print(f"device: {_device_name()}")
        print("=" * 100)

    dataframes_by_cluster_id: dict[int, Any] = {}
    csv_paths_by_cluster_id: dict[str, str] = {}
    run_start = time.perf_counter()

    for cluster_position, cluster_id in enumerate(cluster_ids, start=1):
        subject = canonical_name_for_cluster(doc, cluster_id)
        csv_path = default_cluster_csv_path(
            output_dir,
            cluster_id=cluster_id,
            subject=subject,
            n_mentions=n_mentions,
        )
        per_cluster_seed = _cluster_random_seed(random_seed, cluster_id)

        if print_progress:
            print()
            print("-" * 100)
            print(f"cluster {cluster_position}/{len(cluster_ids)}")
            print(f"cluster_id: {cluster_id}")
            print(f"subject from doc._.coref_layer: {subject!r}")
            print(f"csv_path: {csv_path}")
            print(f"per_cluster_seed: {per_cluster_seed}")
            print("-" * 100)

        df = benchmark_ocean_mention_probability_csv(
            doc,
            cluster_id,
            csv_path=csv_path,
            subject=subject,
            n_mentions=n_mentions,
            start_index=0,
            sample_mentions=True,
            random_seed=per_cluster_seed,
            sort_sample_by_cluster_order=sort_sample_by_cluster_order,
            context_config=context_config,
            rendering_config=rendering_config,
            scoring_config=scoring_config,
            trait_labels=trait_labels,
            weight_config=weight_config,
            nli_config=nli_config,
            chunk_size=chunk_size,
            overwrite_csv=overwrite_csv,
            resume_from_csv=resume_from_csv,
            use_sqlite_cache=use_sqlite_cache,
            keep_rows_in_memory=keep_rows_in_memory,
            return_dataframe=return_dataframes,
            print_progress=print_progress,
        )
        dataframes_by_cluster_id[cluster_id] = df
        csv_paths_by_cluster_id[str(cluster_id)] = str(csv_path)

        # In the memory-safe mode, make sure the previous cluster cannot keep a
        # large DataFrame alive beyond the returned lightweight object.
        if not return_dataframes:
            _release_chunk_memory()

    total_elapsed = time.perf_counter() - run_start

    summary = {
        "cluster_ids": cluster_ids,
        "n_mentions": n_mentions,
        "output_dir": str(output_dir),
        "random_seed": random_seed,
        "sort_sample_by_cluster_order": sort_sample_by_cluster_order,
        "subject_source": "doc._.coref_layer.clusters[cluster_id].canonical_name",
        "csv_paths": csv_paths_by_cluster_id,
        "total_seconds": total_elapsed,
        "device": _device_name(),
        "model_name": scoring_config.model_name,
        "ocean_weight_enabled": weight_config.enabled,
        "overwrite_csv": overwrite_csv,
        "resume_from_csv": resume_from_csv,
        "use_sqlite_cache": use_sqlite_cache,
        "return_dataframes": return_dataframes,
    }

    summary_path = output_dir / "OCEAN_sampled_clusters_run.metadata.json"
    _write_json_atomic(summary_path, summary)

    if print_progress:
        print()
        print("=" * 100)
        print("MULTI-CLUSTER CSV RUN COMPLETE")
        print("=" * 100)
        print(f"clusters processed: {len(dataframes_by_cluster_id)}")
        print(f"total time: {total_elapsed:.2f}s")
        print(f"summary metadata saved to: {summary_path}")

    return dataframes_by_cluster_id



def aggregate_ocean_probability_dataframe(
    df: Any,
    *,
    neutral_score: float = 50.0,
    weight_column: str = "OCEAN_weight",
) -> dict[str, float]:
    """Compute a weighted cluster-level OCEAN profile from the CSV/DataFrame."""

    if df.empty:
        return {trait: neutral_score for trait in OCEAN_TRAITS}

    weights = df[weight_column].astype(float).clip(lower=0.0)
    total_weight = float(weights.sum())

    if total_weight <= 1e-9:
        return {trait: neutral_score for trait in OCEAN_TRAITS}

    return {
        trait: round(float((df[trait].astype(float) * weights).sum() / total_weight), 2)
        for trait in OCEAN_TRAITS
    }
