from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
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



def canonical_name_for_cluster(doc: Any, cluster_id: int) -> Optional[str]:
    """Best-effort canonical name extraction for a cluster."""

    coref_layer = require_coref_layer(doc)
    cluster = coref_layer.clusters[cluster_id]
    canonical_name = getattr(cluster, "canonical_name", None)

    if canonical_name is None:
        return None

    canonical_name = str(canonical_name).strip()
    return canonical_name or None



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
# 8. Small benchmark helper
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
