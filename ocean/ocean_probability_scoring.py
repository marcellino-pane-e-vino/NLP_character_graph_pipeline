"""CSV-first OCEAN probability scoring.

This module is the model-inference/export stage of the OCEAN staging pipeline.
It consumes ``doc._.coref_layer`` and an explicit list of ``cluster_ids``.
It does not choose clusters by semantic type, does not annotate the Doc, and
never mutates ``doc._.coref_layer``.

Output contract:
    ``./outputs/OCEAN_profiles/{n_mentions}/OCEAN_scores_cluster_*.csv``

The CSV raw probability columns are the source of truth. Final collapsed OCEAN
scores are intentionally computed later by ``ocean_annotator.py``.
"""

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
from typing import Any, Iterable, Optional

import torch

from ocean.ocean_schema import OCEAN_TRAITS


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_HYPOTHESIS_TEMPLATE",
    "DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE",
    "OCEAN_LABELS",
    "OCEAN_WEIGHT_LABELS",
    "ContextConfig",
    "MentionRenderingConfig",
    "OceanScoringConfig",
    "OceanWeightConfig",
    "DirectNLIConfig",
    "OceanProbabilityExportConfig",
    "MentionRecord",
    "normalize_context_for_dedup",
    "mention_ids_for_cluster",
    "canonical_name_for_cluster",
    "mention_records_for_cluster",
    "ocean_profiles_output_dir",
    "default_cluster_csv_path",
    "export_ocean_probability_csv_for_cluster",
    "export_ocean_probability_csvs",
]


# =============================================================================
# Configuration and labels
# =============================================================================

DEFAULT_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_HYPOTHESIS_TEMPLATE = "This text shows {}."
DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE = "{subject} shows {} in this text."
PROBABILITY_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class OceanScoringConfig:
    """Runtime configuration for direct-NLI OCEAN probability scoring."""

    model_name: str = DEFAULT_MODEL_NAME
    generic_hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE
    subject_hypothesis_template: str = DEFAULT_SUBJECT_HYPOTHESIS_TEMPLATE
    subject_aware: bool = True
    truncation: bool = True
    multi_label: bool = False  # retained for config readability; direct scorer uses grouped softmax


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


@dataclass(frozen=True)
class OceanWeightConfig:
    """Configuration for raw OCEAN_weight probability scoring.

    OCEAN_weight is always scored in the refactored pipeline because
    ``ocean_annotator.py`` requires the three raw weight probability columns.
    """

    hypothesis_template: str = (
        "In this text, {subject}'s behavior or inner state provides {}."
    )


@dataclass(frozen=True)
class DirectNLIConfig:
    """Configuration for lower-level NLI pair scoring."""

    pair_batch_size: int = 16
    truncation: bool = True
    max_length: Optional[int] = None


@dataclass(frozen=True)
class OceanProbabilityExportConfig:
    """Configuration for multi-cluster OCEAN probability export.

    ``cluster_ids`` is intentionally explicit. The caller decides how that list
    is produced. This module does not know or care whether the list came from
    manual testing, semantic typing, ontology fitting, or some future selector.
    """

    cluster_ids: list[int]
    n_mentions_per_cluster: int | None

    output_root: str | Path = "./outputs"
    random_seed: int | None = None
    sort_sample_by_cluster_order: bool = True

    overwrite_csv: bool = False
    resume_from_csv: bool = True
    use_sqlite_cache: bool = True

    chunk_size: int = 16

    context_config: ContextConfig = field(default_factory=ContextConfig)
    rendering_config: MentionRenderingConfig = field(default_factory=MentionRenderingConfig)
    scoring_config: OceanScoringConfig = field(default_factory=OceanScoringConfig)
    weight_config: OceanWeightConfig = field(default_factory=OceanWeightConfig)
    nli_config: DirectNLIConfig = field(default_factory=DirectNLIConfig)
    return_dataframes: bool = False
    print_progress: bool = True


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
        "neutral": "no clear evidence about openness, curiosity, imagination, or exploration",
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
        "neutral": "no clear evidence about conscientiousness, carefulness, planning, or responsibility",
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
        "neutral": "no clear evidence about extraversion, sociability, assertiveness, or outgoing energy",
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
        "neutral": "no clear evidence about agreeableness, kindness, compassion, cooperation, or hostility",
    },
    "neuroticism": {
        "positive": "high neuroticism: fear, anxiety, sadness, distress, emotional instability, worry",
        "negative": "low neuroticism: calmness, confidence, emotional stability, composure, lack of distress",
        "neutral": "no clear evidence about neuroticism, fear, anxiety, sadness, worry, or emotional stability",
    },
}


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
# Text normalization and coreference helpers
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


def _require_coref_layer(doc: Any) -> Any:
    if not hasattr(doc, "_") or not hasattr(doc._, "coref_layer"):
        raise ValueError("doc has no doc._.coref_layer")
    coref_layer = doc._.coref_layer
    if coref_layer is None:
        raise ValueError("doc._.coref_layer is None")
    return coref_layer


def mention_ids_for_cluster(doc: Any, cluster_id: int) -> list[int]:
    coref_layer = _require_coref_layer(doc)
    if cluster_id not in coref_layer.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    return list(coref_layer.clusters[cluster_id].mention_ids)


def canonical_name_for_cluster(doc: Any, cluster_id: int) -> str:
    coref_layer = _require_coref_layer(doc)
    if cluster_id not in coref_layer.clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    canonical_name = str(coref_layer.clusters[cluster_id].canonical_name).strip()
    if not canonical_name:
        raise ValueError(f"cluster_id={cluster_id} has an empty canonical_name")
    return canonical_name


def find_mention_by_id(doc: Any, mention_id: int) -> Any:
    coref_layer = _require_coref_layer(doc)
    if hasattr(coref_layer, "mentions") and mention_id in coref_layer.mentions:
        return coref_layer.mentions[mention_id]
    raise KeyError(f"mention_id={mention_id} not found in doc._.coref_layer")


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


def _cluster_random_seed(base_seed: int | None, cluster_id: int) -> int | None:
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


# =============================================================================
# Direct NLI model backend
# =============================================================================


def _hypothesis_template(config: OceanScoringConfig, subject: str | None) -> str:
    if config.subject_aware and subject:
        return config.subject_hypothesis_template.replace("{subject}", subject)
    return config.generic_hypothesis_template


def _replace_subject_in_template(template: str, subject: str | None) -> str:
    if "{subject}" in template:
        return template.replace("{subject}", subject or "the character")
    return template


def _format_candidate_hypothesis(template: str, label_text: str) -> str:
    try:
        return template.format(label_text)
    except Exception as exc:
        raise ValueError(
            "Hypothesis template must contain exactly one positional {} placeholder "
            f"after subject replacement. Got: {template!r}"
        ) from exc


@dataclass(frozen=True)
class ProbabilityTask:
    task_name: str
    label_texts: dict[str, str]
    hypothesis_template: str


def _trait_probability_tasks(
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
) -> list[ProbabilityTask]:
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


def _weight_probability_task(*, subject: str | None, weight_config: OceanWeightConfig) -> ProbabilityTask:
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


@lru_cache(maxsize=4)
def _load_direct_nli_components(model_name: str) -> tuple[Any, Any, torch.device, int]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    entailment_id = _find_entailment_id(model)
    return tokenizer, model, device, entailment_id


def _find_entailment_id(model: Any) -> int:
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
    raise ValueError("Could not identify entailment label id from model config.")


def _release_chunk_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_name() -> str:
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return "cpu"


def _direct_entailment_logits_for_pairs(
    pairs: list[tuple[str, str]],
    *,
    model_name: str,
    nli_config: DirectNLIConfig,
) -> list[float]:
    if nli_config.pair_batch_size <= 0:
        raise ValueError(f"pair_batch_size must be > 0, got {nli_config.pair_batch_size}")
    if not pairs:
        return []

    tokenizer, model, device, entailment_id = _load_direct_nli_components(model_name)
    logits: list[float] = []

    for start in range(0, len(pairs), nli_config.pair_batch_size):
        batch = pairs[start: start + nli_config.pair_batch_size]
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
        del batch, premises, hypotheses, encoded, output, batch_logits
        _release_chunk_memory()

    return logits


def _score_probability_payloads_for_chunk(
    records: list[MentionRecord],
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: DirectNLIConfig,
) -> list[dict[str, Any]]:
    tasks = _trait_probability_tasks(
        subject=subject,
        scoring_config=scoring_config,
        trait_labels=trait_labels,
    )
    tasks.append(_weight_probability_task(subject=subject, weight_config=weight_config))

    pair_metadata: list[tuple[int, str, str]] = []
    pairs: list[tuple[str, str]] = []

    for record_index, record in enumerate(records):
        for task in tasks:
            for label_key, label_text in task.label_texts.items():
                hypothesis = _format_candidate_hypothesis(task.hypothesis_template, label_text)
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

    payloads: list[dict[str, Any]] = []
    for record_index, _record in enumerate(records):
        payload: dict[str, Any] = {}
        for trait in OCEAN_TRAITS:
            label_logits = grouped_logits[(record_index, trait)]
            ordered_keys = ["positive", "negative", "neutral"]
            logits_tensor = torch.tensor([label_logits[key] for key in ordered_keys], dtype=torch.float32)
            probabilities = torch.softmax(logits_tensor, dim=0).tolist()
            probability_by_key = dict(zip(ordered_keys, probabilities))
            payload[f"{trait}_positive_probability"] = float(probability_by_key["positive"])
            payload[f"{trait}_negative_probability"] = float(probability_by_key["negative"])
            payload[f"{trait}_neutral_probability"] = float(probability_by_key["neutral"])
            del logits_tensor

        label_logits = grouped_logits[(record_index, "OCEAN_weight")]
        ordered_keys = ["high", "medium", "low"]
        logits_tensor = torch.tensor([label_logits[key] for key in ordered_keys], dtype=torch.float32)
        probabilities = torch.softmax(logits_tensor, dim=0).tolist()
        probability_by_key = dict(zip(ordered_keys, probabilities))
        payload["OCEAN_weight_high_probability"] = float(probability_by_key["high"])
        payload["OCEAN_weight_medium_probability"] = float(probability_by_key["medium"])
        payload["OCEAN_weight_low_probability"] = float(probability_by_key["low"])
        del logits_tensor

        payloads.append(payload)

    del tasks, pair_metadata, pairs, entailment_logits, grouped_logits
    _release_chunk_memory()
    return payloads


# =============================================================================
# SQLite cache and CSV export
# =============================================================================


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


class SQLiteProbabilityCache:
    """SQLite cache for resumable direct-NLI probability payloads."""

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
            INSERT OR REPLACE INTO ocean_probability_cache(cache_key, payload_json, created_at_unix)
            VALUES (?, ?, ?)
            """,
            [(key, _stable_json(_json_ready(payload)), now) for key, payload in items],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteProbabilityCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _probability_cache_key(
    record: MentionRecord,
    *,
    subject: str | None,
    scoring_config: OceanScoringConfig,
    trait_labels: dict[str, dict[str, str]],
    weight_config: OceanWeightConfig,
    nli_config: DirectNLIConfig,
) -> str:
    payload = {
        "schema_version": PROBABILITY_CACHE_SCHEMA_VERSION,
        "normalized_context_text": normalize_context_for_dedup(record.context_text),
        "subject": subject,
        "scoring_config": scoring_config,
        "trait_labels": trait_labels,
        "weight_config": {"config": weight_config, "labels": OCEAN_WEIGHT_LABELS},
        "nli_config": {
            "truncation": nli_config.truncation,
            "max_length": nli_config.max_length,
        },
    }
    return hashlib.sha256(_stable_json(_json_ready(payload)).encode("utf-8")).hexdigest()


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
    _release_chunk_memory()
    return rows, cache_hits, cache_misses


def _probability_payload_to_row(
    record: MentionRecord,
    payload: dict[str, Any],
    *,
    chunk_index: int,
    elapsed_seconds_at_completion: float,
) -> dict[str, Any]:
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


REQUIRED_RAW_COMPLETION_COLUMNS: tuple[str, ...] = tuple(
    f"{trait}_{polarity}_probability"
    for trait in OCEAN_TRAITS
    for polarity in ("positive", "negative", "neutral")
) + (
    "OCEAN_weight_high_probability",
    "OCEAN_weight_medium_probability",
    "OCEAN_weight_low_probability",
)

DEPRECATED_FINAL_COLUMNS: set[str] = set(OCEAN_TRAITS) | {"OCEAN_weight"}


def _completed_mention_ids_from_csv(csv_path: str | Path) -> set[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    completed: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mention_id" not in reader.fieldnames:
            raise ValueError(f"Cannot resume from {csv_path}: CSV has no mention_id column.")
        missing_columns = set(REQUIRED_RAW_COMPLETION_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Cannot resume from {csv_path}: CSV is missing raw probability columns: "
                f"{sorted(missing_columns)}. Use overwrite_csv=True."
            )
        if DEPRECATED_FINAL_COLUMNS & set(reader.fieldnames):
            raise ValueError(
                f"Cannot resume into legacy CSV with deprecated final columns: {csv_path}. "
                "Use overwrite_csv=True to create the refactored raw-only CSV."
            )
        for row in reader:
            raw_mention_id = row.get("mention_id")
            if raw_mention_id in (None, ""):
                continue
            if any(row.get(column) in (None, "") for column in REQUIRED_RAW_COMPLETION_COLUMNS):
                continue
            completed.add(int(float(raw_mention_id)))
    return completed


def _safe_filename_component(value: Any, *, default: str = "unknown") -> str:
    text = str(value if value is not None else default).strip() or default
    text = normalize_context_for_dedup(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def ocean_profiles_output_dir(
    *,
    output_root: str | Path,
    n_mentions_per_cluster: int | None,
) -> Path:
    n_part = "all" if n_mentions_per_cluster is None else str(n_mentions_per_cluster)
    return Path(output_root) / "OCEAN_profiles" / n_part


def default_cluster_csv_path(
    output_dir: str | Path,
    *,
    cluster_id: int,
    subject: str,
    n_mentions: int | None,
) -> Path:
    subject_part = _safe_filename_component(subject, default="unknown_subject")
    n_part = "all" if n_mentions is None else str(n_mentions)
    return Path(output_dir) / f"OCEAN_scores_cluster_{int(cluster_id)}_{subject_part}_{n_part}.csv"


def export_ocean_probability_csv_for_cluster(
    *,
    doc: Any,
    cluster_id: int,
    csv_path: str | Path,
    n_mentions: int | None,
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
    return_dataframe: bool = False,
    print_progress: bool = True,
) -> Any:
    import pandas as pd

    if n_mentions is not None and n_mentions < 0:
        raise ValueError(f"n_mentions must be >= 0 or None, got {n_mentions}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    _require_coref_layer(doc)
    if cluster_id not in _require_coref_layer(doc).clusters:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = csv_path.with_suffix(csv_path.suffix + ".cache.sqlite3")
    cache_path = Path(cache_path)

    if overwrite_csv and csv_path.exists():
        csv_path.unlink()

    context_config = context_config or ContextConfig(deduplicate=False)
    rendering_config = rendering_config or MentionRenderingConfig()
    scoring_config = scoring_config or OceanScoringConfig()
    trait_labels = trait_labels or OCEAN_LABELS
    weight_config = weight_config or OceanWeightConfig()
    nli_config = nli_config or DirectNLIConfig()

    subject = canonical_name_for_cluster(doc, cluster_id)
    total_mentions_in_cluster = len(mention_ids_for_cluster(doc, cluster_id))

    if print_progress:
        print("=" * 100)
        print("OCEAN probability CSV export")
        print(f"cluster_id: {cluster_id}")
        print(f"subject: {subject!r}")
        print(f"requested mentions: {n_mentions}")
        print(f"total mentions in cluster: {total_mentions_in_cluster}")
        print(f"csv_path: {csv_path}")
        print(f"cache_path: {cache_path if use_sqlite_cache else None}")
        print(f"device: {_device_name()}")
        print("=" * 100)

    extraction_start = time.perf_counter()
    records = mention_records_for_cluster(
        doc,
        cluster_id,
        subject=subject,
        n_mentions=n_mentions,
        random_seed=random_seed,
        sort_sample_by_cluster_order=sort_sample_by_cluster_order,
        context_config=context_config,
        rendering_config=rendering_config,
    )
    extraction_elapsed = time.perf_counter() - extraction_start

    completed_mention_ids: set[int] = set()
    if resume_from_csv and csv_path.exists() and not overwrite_csv:
        completed_mention_ids = _completed_mention_ids_from_csv(csv_path)

    if completed_mention_ids:
        before = len(records)
        records = [record for record in records if record.mention_id not in completed_mention_ids]
        skipped_from_existing_csv = before - len(records)
    else:
        skipped_from_existing_csv = 0

    if print_progress:
        print(f"extracted records: {len(records) + skipped_from_existing_csv}")
        print(f"already completed in CSV: {skipped_from_existing_csv}")
        print(f"records left to score/write: {len(records)}")
        print(f"context extraction time: {extraction_elapsed:.2f}s")

    n_records_to_score = len(records)
    n_chunks = (n_records_to_score + chunk_size - 1) // chunk_size if n_records_to_score else 0
    csv_header_written = csv_path.exists() and csv_path.stat().st_size > 0 and not overwrite_csv
    newly_written_rows = 0
    total_cache_hits = 0
    total_cache_misses = 0
    kept_rows: list[dict[str, Any]] = []
    scoring_start = time.perf_counter()

    cache_manager = SQLiteProbabilityCache(cache_path) if use_sqlite_cache else None
    try:
        for chunk_index, start in enumerate(range(0, n_records_to_score, chunk_size), start=1):
            end = min(start + chunk_size, n_records_to_score)
            chunk_records = records[start:end]
            _sync_cuda_if_available()
            chunk_start = time.perf_counter()
            elapsed_before = time.perf_counter() - scoring_start
            chunk_rows, cache_hits, cache_misses = _score_probability_rows_for_chunk_cached(
                chunk_records,
                subject=subject,
                scoring_config=scoring_config,
                trait_labels=trait_labels,
                weight_config=weight_config,
                nli_config=nli_config,
                cache=cache_manager,
                chunk_index=chunk_index,
                elapsed_seconds_at_completion=elapsed_before,
            )
            _sync_cuda_if_available()
            chunk_elapsed = time.perf_counter() - chunk_start
            elapsed_so_far = time.perf_counter() - scoring_start
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

            if return_dataframe:
                kept_rows.extend(chunk_rows)

            newly_written_rows += len(chunk_rows)
            total_cache_hits += cache_hits
            total_cache_misses += cache_misses

            if print_progress:
                n_tasks = len(OCEAN_TRAITS) + 1
                n_pairs = cache_misses * n_tasks * 3
                done_total = skipped_from_existing_csv + newly_written_rows
                print(
                    f"[chunk {chunk_index}/{n_chunks}] "
                    f"mentions={start}:{end} | "
                    f"model_pairs={n_pairs} | "
                    f"cache_hits={cache_hits} | "
                    f"cache_misses={cache_misses} | "
                    f"chunk_time={chunk_elapsed:.2f}s | "
                    f"done_total={done_total}/{n_records_to_score + skipped_from_existing_csv}"
                )

            del chunk_df, chunk_rows, chunk_records
            _release_chunk_memory()
    finally:
        if cache_manager is not None:
            cache_manager.close()

    if print_progress:
        print("=" * 100)
        print("OCEAN PROBABILITY CSV EXPORT COMPLETE")
        print(f"new rows written: {newly_written_rows}")
        print(f"skipped from existing CSV: {skipped_from_existing_csv}")
        print(f"cache hits: {total_cache_hits}")
        print(f"cache misses: {total_cache_misses}")
        print(f"csv saved to: {csv_path}")
        print("=" * 100)

    if return_dataframe:
        return pd.DataFrame(kept_rows)
    return pd.DataFrame()


def export_ocean_probability_csvs(
    doc: Any,
    config: OceanProbabilityExportConfig,
) -> dict[int, Path]:
    coref_layer = _require_coref_layer(doc)

    if not config.cluster_ids:
        raise ValueError("cluster_ids cannot be empty.")

    output_dir = ocean_profiles_output_dir(
        output_root=config.output_root,
        n_mentions_per_cluster=config.n_mentions_per_cluster,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.print_progress:
        print("=" * 100)
        print("Multi-cluster OCEAN probability export")
        print(f"cluster_ids: {config.cluster_ids}")
        print(f"n_mentions_per_cluster: {config.n_mentions_per_cluster}")
        print(f"output_dir: {output_dir}")
        print("=" * 100)

    csv_paths: dict[int, Path] = {}

    for cluster_position, cluster_id in enumerate(config.cluster_ids, start=1):
        if cluster_id not in coref_layer.clusters:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")

        subject = canonical_name_for_cluster(doc, cluster_id)
        csv_path = default_cluster_csv_path(
            output_dir,
            cluster_id=cluster_id,
            subject=subject,
            n_mentions=config.n_mentions_per_cluster,
        )
        per_cluster_seed = _cluster_random_seed(config.random_seed, cluster_id)

        if config.print_progress:
            print()
            print("-" * 100)
            print(f"cluster {cluster_position}/{len(config.cluster_ids)}")
            print(f"cluster_id: {cluster_id}")
            print(f"subject: {subject!r}")
            print(f"csv_path: {csv_path}")
            print(f"per_cluster_seed: {per_cluster_seed}")
            print("-" * 100)

        export_ocean_probability_csv_for_cluster(
            doc=doc,
            cluster_id=cluster_id,
            csv_path=csv_path,
            n_mentions=config.n_mentions_per_cluster,
            random_seed=per_cluster_seed,
            sort_sample_by_cluster_order=config.sort_sample_by_cluster_order,
            context_config=config.context_config,
            rendering_config=config.rendering_config,
            scoring_config=config.scoring_config,
            trait_labels=OCEAN_LABELS,
            weight_config=config.weight_config,
            nli_config=config.nli_config,
            chunk_size=config.chunk_size,
            overwrite_csv=config.overwrite_csv,
            resume_from_csv=config.resume_from_csv,
            use_sqlite_cache=config.use_sqlite_cache,
            return_dataframe=config.return_dataframes,
            print_progress=config.print_progress,
        )
        csv_paths[cluster_id] = csv_path

    return csv_paths
