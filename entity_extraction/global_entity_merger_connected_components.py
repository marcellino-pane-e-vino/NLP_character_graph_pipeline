from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

MentionKey = tuple[int, int, str]


# ---------------------------------------------------------------------------
# Final global cluster salience filter config
# ---------------------------------------------------------------------------

# Intentionally local to this module: callers keep using the same public
# merge_local_coreference_clusters(...) contract, while this merger exports
# filtered global artifacts plus an audit CSV for rejected clusters.
GLOBAL_CLUSTER_SALIENCE_FILTER_ENABLED = True
GLOBAL_CLUSTER_SALIENCE_PERCENTILE = 0.95
MIN_GLOBAL_CLUSTER_MENTIONS: int | None = None
MIN_KEPT_GLOBAL_CLUSTERS: int | None = 10
REJECTED_GLOBAL_CLUSTERS_FILENAME = "rejected_global_clusters.csv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalMention:
    """One mention observation from mentions.csv."""

    local_mention_uid: str
    local_cluster_uid: str

    chunk_id: str
    chunk_index: int

    local_cluster_id: int
    local_mention_id: int

    local_start: int
    local_end: int
    global_start: int
    global_end: int

    text: str
    normalized_text: str

    head_local_i: int
    head_global_i: int

    zone: str
    overlap_exact_span_key: Optional[str]

    @property
    def mention_key(self) -> MentionKey:
        """Exact global mention key used for overlap-based matching."""
        return (self.global_start, self.global_end, self.normalized_text)

    @property
    def is_overlap(self) -> bool:
        """Whether this mention belongs to a left/right overlap region."""
        return self.zone in {"left_overlap", "right_overlap"}

    @property
    def is_pronominal(self) -> bool:
        """Whether this mention is a standalone pronoun/demonstrative."""
        return looks_like_pronoun(self.normalized_text)

    @property
    def is_relevant_for_overlap_coefficient(self) -> bool:
        """Whether this mention should be used as lexical overlap evidence.

        The global merge sieves should compare nominal/proper mentions, not
        standalone pronominal mentions. This keeps mentions such as "the girl",
        "her aunt", "Dorothy", and "the Tin Woodman", while excluding standalone
        mentions such as "she", "her", "they", and "you".
        """
        if self.is_pronominal:
            return False
        return bool(canonical_content_tokens(self.normalized_text))


@dataclass
class LocalCorefCluster:
    """One original Maverick local cluster from one chunk."""

    local_cluster_uid: str

    chunk_id: str
    chunk_index: int
    local_cluster_id: int

    canonical_name: str
    normalized_canonical_name: str

    mentions: list[LocalMention]


@dataclass
class MergeComponent:
    """Current mergeable component.

    Initially, one MergeComponent wraps one LocalCorefCluster. After aggregative
    merging, one MergeComponent can wrap multiple original local clusters.
    """

    merge_component_uid: str

    local_cluster_uids: set[str]
    mentions: list[LocalMention]

    canonical_name_counts: Counter[str]
    original_canonical_name_counts: Counter[str]

    canonical_names: set[str] = field(default_factory=set)
    mention_keys: set[MentionKey] = field(default_factory=set)
    relevant_mention_keys: set[MentionKey] = field(default_factory=set)
    overlap_mention_keys: set[MentionKey] = field(default_factory=set)
    overlap_relevant_mention_keys: set[MentionKey] = field(default_factory=set)
    proper_name_anchors: set[str] = field(default_factory=set)
    chunk_indices: set[int] = field(default_factory=set)

    def recompute(self) -> None:
        """Recompute all derived fields after initialization or merging."""
        self.canonical_names = {
            canonical for canonical in self.canonical_name_counts if canonical
        }

        self.mention_keys = {mention.mention_key for mention in self.mentions}

        self.relevant_mention_keys = {
            mention.mention_key
            for mention in self.mentions
            if mention.is_relevant_for_overlap_coefficient
        }

        self.overlap_mention_keys = {
            mention.mention_key for mention in self.mentions if mention.is_overlap
        }

        self.overlap_relevant_mention_keys = {
            mention.mention_key
            for mention in self.mentions
            if mention.is_overlap and mention.is_relevant_for_overlap_coefficient
        }

        self.proper_name_anchors = extract_proper_name_anchors(self)

        self.chunk_indices = {mention.chunk_index for mention in self.mentions}


@dataclass(frozen=True)
class AcceptedMergeEdge:
    """Accepted pairwise merge relation proposed by a positive sieve."""

    sieve_name: str
    left_component_uid: str
    right_component_uid: str
    reason: str
    metrics: dict[str, int | float | str]


@dataclass(frozen=True)
class SieveEvaluation:
    """Result of evaluating one positive sieve once."""

    sieve_name: str
    candidate_pairs: int
    accepted_edges: list[AcceptedMergeEdge]


@dataclass
class GlobalCorefMergeResult:
    """Returned result for notebook inspection."""

    components: dict[str, MergeComponent]
    global_cluster_uid_by_component_uid: dict[str, str]
    output_dir: Optional[Path]


@dataclass(frozen=True)
class CurrentComponentIndexes:
    """Sparse indexes used to generate candidate pairs for the current state."""

    by_canonical_and_overlap_key: dict[tuple[str, MentionKey], set[str]]
    by_mention_key: dict[MentionKey, set[str]]
    by_relevant_mention_key: dict[MentionKey, set[str]]
    by_overlap_relevant_mention_key: dict[MentionKey, set[str]]
    by_proper_name_anchor: dict[str, set[str]]
    by_canonical_head: dict[str, set[str]]


# ---------------------------------------------------------------------------
# Generic normalization helpers
# ---------------------------------------------------------------------------


CANONICAL_STOP_TOKENS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "his",
    "her",
    "their",
    "my",
    "your",
    "our",
}

PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "we",
    "us",
    "our",
    "ours",
    "they",
    "them",
    "their",
    "theirs",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
}

# Intentionally minimal. These heads are broad placeholders, not novel-specific
# entities. Do not add story-specific nouns here.
GENERIC_ENTITY_HEADS = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "child",
    "children",
    "thing",
    "things",
    "one",
    "ones",
    "other",
    "others",
    "someone",
    "somebody",
    "anyone",
    "anybody",
    "nobody",
}

GENERIC_MENTION_TEXTS = {
    "someone",
    "somebody",
    "anyone",
    "anybody",
    "nobody",
    "no one",
    "the others",
    "the other",
    "another",
    "others",
    "a man",
    "the man",
    "a woman",
    "the woman",
    "a person",
    "the person",
}

GENERIC_ENTITY_HEADS_NORMALIZED = set()
for _generic_head in GENERIC_ENTITY_HEADS:
    if _generic_head.endswith("ies") and len(_generic_head) > 4:
        GENERIC_ENTITY_HEADS_NORMALIZED.add(_generic_head[:-3] + "y")
    elif (
        _generic_head.endswith("s")
        and len(_generic_head) > 3
        and not _generic_head.endswith("ss")
    ):
        GENERIC_ENTITY_HEADS_NORMALIZED.add(_generic_head[:-1])
    else:
        GENERIC_ENTITY_HEADS_NORMALIZED.add(_generic_head)


def normalize_text(text: object) -> str:
    """Lightweight, deterministic text normalization."""
    value = "" if text is None else str(text)
    value = value.strip().lower()
    value = " ".join(value.split())
    return value


def none_if_nan(value: object) -> Optional[str]:
    """Convert pandas/CSV missing values to None, otherwise string."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    value_as_text = str(value)
    if value_as_text.lower() == "nan":
        return None
    return value_as_text


def ordered_content_tokens(text: str) -> tuple[str, ...]:
    """Return ordered content tokens for phrase-level heuristics."""
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    return tuple(token for token in raw_tokens if token not in CANONICAL_STOP_TOKENS)


def canonical_content_tokens(text: str) -> frozenset[str]:
    """Return simple content-token signature for a canonical/mention string."""
    return frozenset(ordered_content_tokens(text))


def normalize_entity_phrase(text: str) -> str:
    """Normalize a phrase into ordered content tokens joined by spaces."""
    return " ".join(ordered_content_tokens(text))


def simple_singularize(token: str) -> str:
    """Very small deterministic singularizer for heuristic head comparison."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def canonical_token_set(text: str) -> set[str]:
    """Return singularized content tokens for phrase-similarity heuristics."""
    return {simple_singularize(token) for token in ordered_content_tokens(text)}


def canonical_head(text: str) -> str:
    """Return the final content token, lightly singularized."""
    tokens = ordered_content_tokens(text)
    if not tokens:
        return ""
    return simple_singularize(tokens[-1])


def is_generic_head(head: str) -> bool:
    """Whether a head is too generic to be strong entity evidence."""
    normalized_head = simple_singularize(normalize_text(head))
    if not normalized_head:
        return True
    return normalized_head in GENERIC_ENTITY_HEADS_NORMALIZED


def is_generic_nominal_text(text: str) -> bool:
    """Whether a mention/canonical string is generic and weak as merge evidence."""
    normalized = normalize_text(text)
    if not normalized:
        return True
    if normalized in GENERIC_MENTION_TEXTS:
        return True
    tokens = ordered_content_tokens(normalized)
    if not tokens:
        return True
    head = canonical_head(normalized)
    return is_generic_head(head)


def looks_like_pronoun(text: str) -> bool:
    """Whether a string is a simple English pronoun/demonstrative."""
    return normalize_text(text) in PRONOUNS


def looks_like_proper_name(text: str) -> bool:
    """Conservative capitalization-based proper-name heuristic.

    This is intentionally shallow. It is not entity typing and does not depend on
    spaCy/NER. It only supports the strict proper-name anchor sieve.
    """
    if not text:
        return False

    if looks_like_pronoun(text):
        return False

    tokens = re.findall(r"[A-Za-z]+", text)
    if not tokens:
        return False

    return any(token[:1].isupper() for token in tokens)


def extract_proper_name_anchors(component: MergeComponent) -> set[str]:
    """Extract conservative proper-name anchors from original canonical names.

    The anchor sieve intentionally uses canonical names only, not mention texts,
    to avoid noisy mention-level proper-name candidates.
    """
    anchors = set()

    for original_name in component.original_canonical_name_counts:
        if looks_like_proper_name(original_name):
            anchors.add(normalize_text(original_name))

    return anchors


# ---------------------------------------------------------------------------
# Generic lexical compatibility helpers
# ---------------------------------------------------------------------------


def token_jaccard(left_tokens: set[str], right_tokens: set[str]) -> float:
    """Return set Jaccard similarity with empty-set protection."""
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def canonical_surface_similarity(
    left_phrase: str,
    right_phrase: str,
    *,
    token_jaccard_threshold: float,
) -> tuple[bool, float, str]:
    """Compare two canonical-like phrases using generic surface structure.

    A match requires:
    - exact normalized phrase equality, or
    - same non-generic head plus token containment, or
    - same non-generic head plus Jaccard above the supplied threshold.

    No novel-specific aliases are used here.
    """
    left_norm = normalize_entity_phrase(left_phrase)
    right_norm = normalize_entity_phrase(right_phrase)

    if not left_norm or not right_norm:
        return False, 0.0, ""

    if left_norm == right_norm:
        head = canonical_head(left_norm)
        return True, 1.0, head

    left_head = canonical_head(left_norm)
    right_head = canonical_head(right_norm)

    if not left_head or left_head != right_head or is_generic_head(left_head):
        return False, 0.0, left_head if left_head == right_head else ""

    left_tokens = canonical_token_set(left_norm)
    right_tokens = canonical_token_set(right_norm)

    if not left_tokens or not right_tokens:
        return False, 0.0, left_head

    score = token_jaccard(left_tokens, right_tokens)

    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True, score, left_head

    if score >= token_jaccard_threshold:
        return True, score, left_head

    return False, score, left_head


def canonical_head_alias_match(
    left_phrase: str,
    right_phrase: str,
) -> tuple[bool, float, str]:
    """Positive-sieve canonical-head alias predicate."""
    return canonical_surface_similarity(
        left_phrase,
        right_phrase,
        token_jaccard_threshold=0.50,
    )


def are_canonical_surfaces_compatible(left_phrase: str, right_phrase: str) -> bool:
    """Generic canonical surface compatibility used only by the veto layer.

    This function does not authorize a merge. It only prevents the veto from
    blocking lexically compatible variants after a positive sieve has already
    proposed an edge.
    """
    is_match, _score, _head = canonical_surface_similarity(
        left_phrase,
        right_phrase,
        token_jaccard_threshold=0.67,
    )
    return is_match


# ---------------------------------------------------------------------------
# Validation and loading
# ---------------------------------------------------------------------------


REQUIRED_CLUSTER_COLUMNS = {
    "local_cluster_uid",
    "chunk_id",
    "chunk_index",
    "local_cluster_id",
    "canonical_name",
    "n_mentions",
}

REQUIRED_MENTION_COLUMNS = {
    "local_mention_uid",
    "local_cluster_uid",
    "chunk_id",
    "chunk_index",
    "local_cluster_id",
    "local_mention_id",
    "local_start",
    "local_end",
    "global_start",
    "global_end",
    "text",
    "head_local_i",
    "head_global_i",
    "zone",
    "overlap_exact_span_key",
}


def validate_required_columns(
    *, clusters_rows: list[dict], mentions_rows: list[dict]
) -> None:
    """Fail early if required input columns are missing."""
    if not clusters_rows:
        raise ValueError("clusters_rows is empty.")
    if not mentions_rows:
        raise ValueError("mentions_rows is empty.")

    cluster_columns = set(clusters_rows[0].keys())
    mention_columns = set(mentions_rows[0].keys())

    missing_cluster_columns = sorted(REQUIRED_CLUSTER_COLUMNS - cluster_columns)
    missing_mention_columns = sorted(REQUIRED_MENTION_COLUMNS - mention_columns)

    if missing_cluster_columns:
        raise ValueError(
            "clusters_rows is missing required columns: "
            + ", ".join(missing_cluster_columns)
        )

    if missing_mention_columns:
        raise ValueError(
            "mentions_rows is missing required columns: "
            + ", ".join(missing_mention_columns)
        )


def build_local_clusters(
    *, clusters_rows: list[dict], mentions_rows: list[dict]
) -> list[LocalCorefCluster]:
    """Join cluster rows and mention rows into LocalCorefCluster objects."""
    mentions_by_cluster_uid: dict[str, list[LocalMention]] = defaultdict(list)

    for row in mentions_rows:
        mention = LocalMention(
            local_mention_uid=str(row["local_mention_uid"]),
            local_cluster_uid=str(row["local_cluster_uid"]),
            chunk_id=str(row["chunk_id"]),
            chunk_index=int(row["chunk_index"]),
            local_cluster_id=int(row["local_cluster_id"]),
            local_mention_id=int(row["local_mention_id"]),
            local_start=int(row["local_start"]),
            local_end=int(row["local_end"]),
            global_start=int(row["global_start"]),
            global_end=int(row["global_end"]),
            text=str(row["text"]),
            normalized_text=normalize_text(row["text"]),
            head_local_i=int(row["head_local_i"]),
            head_global_i=int(row["head_global_i"]),
            zone=str(row["zone"]),
            overlap_exact_span_key=none_if_nan(row.get("overlap_exact_span_key")),
        )
        mentions_by_cluster_uid[mention.local_cluster_uid].append(mention)

    local_clusters: list[LocalCorefCluster] = []

    for row in clusters_rows:
        local_cluster_uid = str(row["local_cluster_uid"])
        canonical_name = str(row["canonical_name"])

        cluster = LocalCorefCluster(
            local_cluster_uid=local_cluster_uid,
            chunk_id=str(row["chunk_id"]),
            chunk_index=int(row["chunk_index"]),
            local_cluster_id=int(row["local_cluster_id"]),
            canonical_name=canonical_name,
            normalized_canonical_name=normalize_text(canonical_name),
            mentions=mentions_by_cluster_uid.get(local_cluster_uid, []),
        )
        local_clusters.append(cluster)

    return local_clusters


def merge_local_coreference_clusters_from_csv(
    *,
    clusters_csv: str | Path,
    mentions_csv: str | Path,
    output_dir: str | Path,
    verbose: bool = True,
    enable_global_merge_veto: bool = True,
) -> GlobalCorefMergeResult:
    """CSV-path wrapper around merge_local_coreference_clusters."""
    clusters_rows = pd.read_csv(clusters_csv).to_dict("records")
    mentions_rows = pd.read_csv(mentions_csv).to_dict("records")

    return merge_local_coreference_clusters(
        clusters_rows=clusters_rows,
        mentions_rows=mentions_rows,
        output_dir=output_dir,
        verbose=verbose,
        enable_global_merge_veto=enable_global_merge_veto,
    )


# ---------------------------------------------------------------------------
# Component initialization and indexes
# ---------------------------------------------------------------------------


def initialize_merge_components(
    local_clusters: list[LocalCorefCluster],
) -> dict[str, MergeComponent]:
    """Create one MergeComponent for each LocalCorefCluster."""
    components: dict[str, MergeComponent] = {}

    for index, cluster in enumerate(local_clusters):
        component_uid = f"merge_component_{index:06d}"

        component = MergeComponent(
            merge_component_uid=component_uid,
            local_cluster_uids={cluster.local_cluster_uid},
            mentions=list(cluster.mentions),
            canonical_name_counts=Counter({cluster.normalized_canonical_name: 1}),
            original_canonical_name_counts=Counter({cluster.canonical_name: 1}),
        )
        component.recompute()
        components[component_uid] = component

    return components


def build_current_component_indexes(
    components: dict[str, MergeComponent],
) -> CurrentComponentIndexes:
    """Build sparse indexes for the current active components."""
    by_canonical_and_overlap_key: dict[tuple[str, MentionKey], set[str]] = defaultdict(set)
    by_mention_key: dict[MentionKey, set[str]] = defaultdict(set)
    by_relevant_mention_key: dict[MentionKey, set[str]] = defaultdict(set)
    by_overlap_relevant_mention_key: dict[MentionKey, set[str]] = defaultdict(set)
    by_proper_name_anchor: dict[str, set[str]] = defaultdict(set)
    by_canonical_head: dict[str, set[str]] = defaultdict(set)

    for component_uid, component in components.items():
        for mention_key in component.mention_keys:
            by_mention_key[mention_key].add(component_uid)

        for mention_key in component.relevant_mention_keys:
            by_relevant_mention_key[mention_key].add(component_uid)

        for mention_key in component.overlap_relevant_mention_keys:
            by_overlap_relevant_mention_key[mention_key].add(component_uid)

        for overlap_key in component.overlap_mention_keys:
            for canonical_name in component.canonical_names:
                by_canonical_and_overlap_key[(canonical_name, overlap_key)].add(
                    component_uid
                )

        for anchor in component.proper_name_anchors:
            by_proper_name_anchor[anchor].add(component_uid)

        for canonical_name in component.canonical_names:
            head = canonical_head(canonical_name)
            if head and not is_generic_head(head):
                by_canonical_head[head].add(component_uid)

    return CurrentComponentIndexes(
        by_canonical_and_overlap_key=dict(by_canonical_and_overlap_key),
        by_mention_key=dict(by_mention_key),
        by_relevant_mention_key=dict(by_relevant_mention_key),
        by_overlap_relevant_mention_key=dict(by_overlap_relevant_mention_key),
        by_proper_name_anchor=dict(by_proper_name_anchor),
        by_canonical_head=dict(by_canonical_head),
    )


def unique_pairs_from_buckets(buckets: Iterable[set[str]]) -> set[tuple[str, str]]:
    """Generate unique sorted pairs from index buckets."""
    pairs: set[tuple[str, str]] = set()

    for bucket in buckets:
        ids = sorted(bucket)
        if len(ids) < 2:
            continue

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))

    return pairs


# ---------------------------------------------------------------------------
# Positive sieve functions
# ---------------------------------------------------------------------------


def sieve_exactCanonicalName_stitching(
    *, components: dict[str, MergeComponent], indexes: CurrentComponentIndexes
) -> SieveEvaluation:
    """Sieve 1: shared normalized canonical name + shared exact overlap mention."""
    sieve_name = "exactMention_stitching"

    candidate_pairs = unique_pairs_from_buckets(
        indexes.by_canonical_and_overlap_key.values()
    )

    accepted_edges: list[AcceptedMergeEdge] = []

    for left_uid, right_uid in candidate_pairs:
        left = components[left_uid]
        right = components[right_uid]

        shared_canonicals = left.canonical_names & right.canonical_names
        shared_overlap_keys = left.overlap_mention_keys & right.overlap_mention_keys

        if not shared_canonicals or not shared_overlap_keys:
            continue

        accepted_edges.append(
            AcceptedMergeEdge(
                sieve_name=sieve_name,
                left_component_uid=left_uid,
                right_component_uid=right_uid,
                reason="shared normalized canonical name and shared exact overlap mention",
                metrics={
                    "shared_canonical_count": len(shared_canonicals),
                    "shared_overlap_mention_count": len(shared_overlap_keys),
                },
            )
        )

    return SieveEvaluation(
        sieve_name=sieve_name,
        candidate_pairs=len(candidate_pairs),
        accepted_edges=accepted_edges,
    )


def are_adjacent_components(left: MergeComponent, right: MergeComponent) -> bool:
    """Whether two components touch on adjacent chunk indices."""
    if not left.chunk_indices or not right.chunk_indices:
        return False

    left_min = min(left.chunk_indices)
    left_max = max(left.chunk_indices)
    right_min = min(right.chunk_indices)
    right_max = max(right.chunk_indices)

    return left_max + 1 == right_min or right_max + 1 == left_min


def sieve_adjacent_overlap_exact_mention_without_canonical_equality(
    *,
    components: dict[str, MergeComponent],
    indexes: CurrentComponentIndexes,
    min_shared_mentions: int = 2,
    overlap_coefficient_threshold: float = 0.80,
) -> SieveEvaluation:
    """Sieve 2: adjacent chunks with strong exact overlap evidence.

    This sieve does not require canonical-name equality. It uses only overlap-zone
    relevant mentions, so pronominal evidence is excluded.
    """
    sieve_name = "adjacent_overlap_exact_mention_without_canonical_equality"

    candidate_pairs = unique_pairs_from_buckets(
        indexes.by_overlap_relevant_mention_key.values()
    )
    accepted_edges: list[AcceptedMergeEdge] = []

    for left_uid, right_uid in candidate_pairs:
        left = components[left_uid]
        right = components[right_uid]

        if not are_adjacent_components(left, right):
            continue

        shared_keys = left.overlap_relevant_mention_keys & right.overlap_relevant_mention_keys
        shared_count = len(shared_keys)

        if shared_count < min_shared_mentions:
            continue

        smaller_size = min(
            len(left.overlap_relevant_mention_keys),
            len(right.overlap_relevant_mention_keys),
        )
        if smaller_size == 0:
            continue

        overlap_coefficient = shared_count / smaller_size
        if overlap_coefficient < overlap_coefficient_threshold:
            continue

        accepted_edges.append(
            AcceptedMergeEdge(
                sieve_name=sieve_name,
                left_component_uid=left_uid,
                right_component_uid=right_uid,
                reason="adjacent components share strong exact overlap relevant mentions",
                metrics={
                    "shared_overlap_relevant_mentions": shared_count,
                    "left_overlap_relevant_mentions": len(left.overlap_relevant_mention_keys),
                    "right_overlap_relevant_mentions": len(right.overlap_relevant_mention_keys),
                    "overlap_coefficient": overlap_coefficient,
                },
            )
        )

    return SieveEvaluation(
        sieve_name=sieve_name,
        candidate_pairs=len(candidate_pairs),
        accepted_edges=accepted_edges,
    )


def sieve_exact_mention_overlap_coefficient(
    *,
    components: dict[str, MergeComponent],
    indexes: CurrentComponentIndexes,
    min_shared_mentions: int = 2,
    overlap_coefficient_threshold: float = 0.75,
) -> SieveEvaluation:
    """Sieve 3: relevant exact mention-key overlap coefficient.

    The coefficient is computed only on relevant nominal/proper mentions.
    Standalone pronominal mentions are ignored for both candidate generation and
    overlap-coefficient computation.
    """
    sieve_name = "exact_mention_overlap_coefficient"

    candidate_pairs = unique_pairs_from_buckets(
        indexes.by_relevant_mention_key.values()
    )
    accepted_edges: list[AcceptedMergeEdge] = []

    for left_uid, right_uid in candidate_pairs:
        left = components[left_uid]
        right = components[right_uid]

        shared_keys = left.relevant_mention_keys & right.relevant_mention_keys
        shared_count = len(shared_keys)

        if shared_count < min_shared_mentions:
            continue

        smaller_size = min(
            len(left.relevant_mention_keys),
            len(right.relevant_mention_keys),
        )
        if smaller_size == 0:
            continue

        overlap_coefficient = shared_count / smaller_size

        if overlap_coefficient < overlap_coefficient_threshold:
            continue

        accepted_edges.append(
            AcceptedMergeEdge(
                sieve_name=sieve_name,
                left_component_uid=left_uid,
                right_component_uid=right_uid,
                reason="exact mention overlap coefficient reached required threshold",
                metrics={
                    "shared_relevant_mentions": shared_count,
                    "left_relevant_mentions": len(left.relevant_mention_keys),
                    "right_relevant_mentions": len(right.relevant_mention_keys),
                    "overlap_coefficient": overlap_coefficient,
                },
            )
        )

    return SieveEvaluation(
        sieve_name=sieve_name,
        candidate_pairs=len(candidate_pairs),
        accepted_edges=accepted_edges,
    )


def sieve_same_clean_proper_name_anchor(
    *, components: dict[str, MergeComponent], indexes: CurrentComponentIndexes
) -> SieveEvaluation:
    """Sieve 4: both components have exactly the same single proper-name anchor."""
    sieve_name = "same_clean_proper_name_anchor"

    candidate_pairs = unique_pairs_from_buckets(indexes.by_proper_name_anchor.values())
    accepted_edges: list[AcceptedMergeEdge] = []

    for left_uid, right_uid in candidate_pairs:
        left = components[left_uid]
        right = components[right_uid]

        if len(left.proper_name_anchors) != 1:
            continue
        if len(right.proper_name_anchors) != 1:
            continue
        if left.proper_name_anchors != right.proper_name_anchors:
            continue

        shared_anchor = next(iter(left.proper_name_anchors))

        accepted_edges.append(
            AcceptedMergeEdge(
                sieve_name=sieve_name,
                left_component_uid=left_uid,
                right_component_uid=right_uid,
                reason="both components have the same single proper-name anchor",
                metrics={"proper_name_anchor": shared_anchor},
            )
        )

    return SieveEvaluation(
        sieve_name=sieve_name,
        candidate_pairs=len(candidate_pairs),
        accepted_edges=accepted_edges,
    )


def sieve_canonical_head_alias_match(
    *, components: dict[str, MergeComponent], indexes: CurrentComponentIndexes
) -> SieveEvaluation:
    """Sieve 5: generic canonical-head alias matching.

    This replaces hardcoded alias groups. It compares canonical names only and
    accepts pairs with the same non-generic head plus token containment or token
    Jaccard >= 0.50.
    """
    sieve_name = "canonical_head_alias_match"

    candidate_pairs = unique_pairs_from_buckets(indexes.by_canonical_head.values())
    accepted_edges: list[AcceptedMergeEdge] = []

    for left_uid, right_uid in candidate_pairs:
        left = components[left_uid]
        right = components[right_uid]

        best_score = 0.0
        best_head = ""
        is_accepted = False

        for left_canonical in left.canonical_names:
            for right_canonical in right.canonical_names:
                is_match, score, head = canonical_head_alias_match(
                    left_canonical,
                    right_canonical,
                )
                if score > best_score:
                    best_score = score
                    best_head = head
                if is_match:
                    is_accepted = True

        if not is_accepted:
            continue

        accepted_edges.append(
            AcceptedMergeEdge(
                sieve_name=sieve_name,
                left_component_uid=left_uid,
                right_component_uid=right_uid,
                reason="canonical names share non-generic head and sufficient surface similarity",
                metrics={
                    "token_jaccard": best_score,
                    "shared_head": best_head,
                },
            )
        )

    return SieveEvaluation(
        sieve_name=sieve_name,
        candidate_pairs=len(candidate_pairs),
        accepted_edges=accepted_edges,
    )


# ---------------------------------------------------------------------------
# Global merge veto layer
# ---------------------------------------------------------------------------


def component_relevant_mention_text_counts(component: MergeComponent) -> Counter[str]:
    """Count normalized non-pronominal mention texts in a component."""
    counts: Counter[str] = Counter()
    for mention in component.mentions:
        if mention.is_relevant_for_overlap_coefficient:
            counts[mention.normalized_text] += 1
    return counts


def component_contains_phrase(component: MergeComponent, phrase: str) -> bool:
    """Whether a component contains a compatible mention/canonical phrase."""
    normalized_phrase = normalize_entity_phrase(phrase)
    if not normalized_phrase:
        return False

    for canonical_name in component.canonical_names:
        if are_canonical_surfaces_compatible(canonical_name, normalized_phrase):
            return True

    for mention_text in component_relevant_mention_text_counts(component):
        if are_canonical_surfaces_compatible(mention_text, normalized_phrase):
            return True

    return False


def is_strong_entity_anchor(text: str) -> bool:
    """Whether a canonical-like phrase is strong enough for negative evidence."""
    normalized_phrase = normalize_entity_phrase(text)
    if not normalized_phrase:
        return False
    if looks_like_pronoun(normalized_phrase):
        return False
    if normalized_phrase in GENERIC_MENTION_TEXTS:
        return False

    tokens = ordered_content_tokens(normalized_phrase)
    if not tokens:
        return False

    head = canonical_head(normalized_phrase)
    if not is_generic_head(head):
        return True

    return len(tokens) >= 2 and any(
        not is_generic_head(token) for token in tokens[:-1]
    )


def component_strong_canonical_anchors(component: MergeComponent) -> set[str]:
    """Return strong canonical anchors for a component."""
    anchors = set()
    for canonical_name in component.canonical_names:
        normalized_phrase = normalize_entity_phrase(canonical_name)
        if is_strong_entity_anchor(normalized_phrase):
            anchors.add(normalized_phrase)
    return anchors


def component_dominant_strong_canonical_anchor(component: MergeComponent) -> Optional[str]:
    """Return the most common strong canonical anchor, if any."""
    for canonical_name, _count in component.canonical_name_counts.most_common():
        normalized_phrase = normalize_entity_phrase(canonical_name)
        if is_strong_entity_anchor(normalized_phrase):
            return normalized_phrase
    return None


def canonical_sets_are_compatible(left: MergeComponent, right: MergeComponent) -> bool:
    """Whether the two components have at least one compatible canonical anchor."""
    left_anchors = component_strong_canonical_anchors(left)
    right_anchors = component_strong_canonical_anchors(right)

    if not left_anchors or not right_anchors:
        return False

    return any(
        are_canonical_surfaces_compatible(left_anchor, right_anchor)
        for left_anchor in left_anchors
        for right_anchor in right_anchors
    )


def edge_shared_relevant_mention_texts(
    left: MergeComponent,
    right: MergeComponent,
) -> set[str]:
    """Return mention texts for exact shared relevant mention keys."""
    shared_keys = left.relevant_mention_keys & right.relevant_mention_keys
    return {mention_text for _start, _end, mention_text in shared_keys}


def _metric_as_float(
    metrics: dict[str, int | float | str],
    key: str,
    default: float = 0.0,
) -> float:
    """Read a numeric metric safely from an AcceptedMergeEdge metrics dict."""
    value = metrics.get(key, default)

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def edge_has_strong_overlap_evidence(edge: AcceptedMergeEdge) -> bool:
    """Whether an edge has strong structural overlap evidence.

    This is not a merge generator. It only prevents the veto layer from blocking
    high-confidence overlap-stitching edges solely because the local canonical
    names differ.
    """
    if edge.sieve_name in {
        "exactMention_stitching",
        "adjacent_overlap_exact_mention_without_canonical_equality",
    }:
        return True

    overlap_coefficient = _metric_as_float(edge.metrics, "overlap_coefficient", 0.0)
    shared_relevant = _metric_as_float(edge.metrics, "shared_relevant_mentions", 0.0)
    shared_overlap = _metric_as_float(edge.metrics, "shared_overlap_mention_count", 0.0)
    shared_overlap_relevant = _metric_as_float(
        edge.metrics,
        "shared_overlap_relevant_mentions",
        0.0,
    )

    return (
        (overlap_coefficient >= 0.95 and shared_relevant >= 3)
        or shared_overlap >= 3
        or shared_overlap_relevant >= 3
    )


def veto_incompatible_dominant_anchors(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Block merges whose dominant strong canonical identities conflict."""
    left_anchor = component_dominant_strong_canonical_anchor(left)
    right_anchor = component_dominant_strong_canonical_anchor(right)

    if not left_anchor or not right_anchor:
        return False

    if are_canonical_surfaces_compatible(left_anchor, right_anchor):
        return False

    if edge_has_strong_overlap_evidence(edge):
        return False

    return True


def veto_disjoint_strong_anchor_sets(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Block merges between components with strong but incompatible anchor sets."""
    left_anchors = component_strong_canonical_anchors(left)
    right_anchors = component_strong_canonical_anchors(right)

    if not left_anchors or not right_anchors:
        return False

    if any(
        are_canonical_surfaces_compatible(left_anchor, right_anchor)
        for left_anchor in left_anchors
        for right_anchor in right_anchors
    ):
        return False

    if edge_has_strong_overlap_evidence(edge):
        return False

    return True


def veto_generic_only_shared_evidence(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Block edges supported only by generic mention text overlap."""
    if canonical_sets_are_compatible(left, right):
        return False

    shared_texts = edge_shared_relevant_mention_texts(left, right)
    if not shared_texts:
        return False

    generic_shared_texts = {
        text for text in shared_texts if is_generic_nominal_text(text)
    }

    return generic_shared_texts == shared_texts


def veto_asymmetric_pollution(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Block one-way evidence that looks like polluted mention leakage."""
    if edge_has_strong_overlap_evidence(edge):
        return False

    left_anchor = component_dominant_strong_canonical_anchor(left)
    right_anchor = component_dominant_strong_canonical_anchor(right)

    if not left_anchor or not right_anchor:
        return False

    if are_canonical_surfaces_compatible(left_anchor, right_anchor):
        return False

    left_contains_right = component_contains_phrase(left, right_anchor)
    right_contains_left = component_contains_phrase(right, left_anchor)

    return left_contains_right != right_contains_left


def _compatible_anchor_group_count(anchors: set[str]) -> int:
    """Count generic surface-compatible anchor groups without alias tables."""
    groups: list[list[str]] = []

    for anchor in sorted(anchors):
        placed = False
        for group in groups:
            if any(are_canonical_surfaces_compatible(anchor, existing) for existing in group):
                group.append(anchor)
                placed = True
                break
        if not placed:
            groups.append([anchor])

    return len(groups)


def veto_canonical_entropy_explosion(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Block weak edges that would introduce too many strong identities."""
    left_anchors = component_strong_canonical_anchors(left)
    right_anchors = component_strong_canonical_anchors(right)
    merged_anchors = left_anchors | right_anchors

    merged_group_count = _compatible_anchor_group_count(merged_anchors)
    if merged_group_count <= 2:
        return False

    strongest_side = max(
        _compatible_anchor_group_count(left_anchors),
        _compatible_anchor_group_count(right_anchors),
    )
    if merged_group_count <= strongest_side:
        return False

    if edge_has_strong_overlap_evidence(edge):
        return False

    return True


def should_veto_global_merge(
    *,
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> bool:
    """Return True when a positive merge edge must be blocked.

    The veto layer is a negative safety layer. It never proposes merges, never
    mutates components, and never emits diagnostics.
    """
    if veto_incompatible_dominant_anchors(left, right, edge):
        return True

    if veto_disjoint_strong_anchor_sets(left, right, edge):
        return True

    if veto_generic_only_shared_evidence(left, right, edge):
        return True

    if veto_asymmetric_pollution(left, right, edge):
        return True

    if veto_canonical_entropy_explosion(left, right, edge):
        return True

    return False


# ---------------------------------------------------------------------------
# Connected-components merging
# ---------------------------------------------------------------------------


def component_document_order_key(component: MergeComponent) -> tuple[int, int, str]:
    """Stable deterministic component order used for tie-breaking and export IDs."""
    chunk_min = min(component.chunk_indices) if component.chunk_indices else 10**12
    mention_start_min = (
        min(mention.global_start for mention in component.mentions)
        if component.mentions
        else 10**12
    )
    return (chunk_min, mention_start_min, component.merge_component_uid)


def accepted_merge_edge_document_order_key(
    edge: AcceptedMergeEdge,
    components: dict[str, MergeComponent],
) -> tuple:
    """Deterministic edge ordering used only for logging/debug stability."""
    left = components[edge.left_component_uid]
    right = components[edge.right_component_uid]

    left_key = component_document_order_key(left)
    right_key = component_document_order_key(right)

    if right_key < left_key:
        left_key, right_key = right_key, left_key

    return (
        left_key,
        right_key,
        edge.sieve_name,
        edge.left_component_uid,
        edge.right_component_uid,
    )


def non_vetoed_accepted_edges(
    *,
    components: dict[str, MergeComponent],
    accepted_edges: list[AcceptedMergeEdge],
    enable_global_merge_veto: bool,
) -> list[AcceptedMergeEdge]:
    """Return currently valid accepted edges after applying the global veto layer."""
    kept_edges: list[AcceptedMergeEdge] = []

    for edge in accepted_edges:
        if edge.left_component_uid not in components:
            continue
        if edge.right_component_uid not in components:
            continue
        if edge.left_component_uid == edge.right_component_uid:
            continue

        if enable_global_merge_veto:
            left = components[edge.left_component_uid]
            right = components[edge.right_component_uid]
            if should_veto_global_merge(left=left, right=right, edge=edge):
                continue

        kept_edges.append(edge)

    return sorted(
        kept_edges,
        key=lambda edge: accepted_merge_edge_document_order_key(edge, components),
    )


def connected_component_uid_groups_from_edges(
    edges: list[AcceptedMergeEdge],
) -> list[tuple[str, ...]]:
    """Compute connected component groups from accepted merge edges.

    Isolated nodes are intentionally omitted, because they do not require a merge.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)

    for edge in edges:
        left_uid = edge.left_component_uid
        right_uid = edge.right_component_uid
        adjacency[left_uid].add(right_uid)
        adjacency[right_uid].add(left_uid)

    groups: list[tuple[str, ...]] = []
    visited: set[str] = set()

    for start_uid in sorted(adjacency):
        if start_uid in visited:
            continue

        stack = [start_uid]
        component_group: set[str] = set()

        while stack:
            uid = stack.pop()
            if uid in visited:
                continue

            visited.add(uid)
            component_group.add(uid)

            for neighbor_uid in sorted(adjacency.get(uid, ())):
                if neighbor_uid not in visited:
                    stack.append(neighbor_uid)

        if len(component_group) > 1:
            groups.append(tuple(sorted(component_group)))

    return groups


def merge_component_uid_group(
    *,
    components: dict[str, MergeComponent],
    component_uids: tuple[str, ...],
    next_component_index: int,
) -> tuple[MergeComponent, int]:
    """Merge one connected-component group into a new MergeComponent."""
    if len(component_uids) < 2:
        raise ValueError("A connected component merge group must contain at least two components.")

    missing = [uid for uid in component_uids if uid not in components]
    if missing:
        raise KeyError(f"Unknown component uid(s): {missing[:10]}")

    ordered_members = sorted(
        (components[uid] for uid in component_uids),
        key=component_document_order_key,
    )

    new_uid = f"merge_component_{next_component_index:06d}"
    next_component_index += 1

    local_cluster_uids: set[str] = set()
    mentions: list[LocalMention] = []
    canonical_name_counts: Counter[str] = Counter()
    original_canonical_name_counts: Counter[str] = Counter()

    for member in ordered_members:
        local_cluster_uids.update(member.local_cluster_uids)
        mentions.extend(member.mentions)
        canonical_name_counts.update(member.canonical_name_counts)
        original_canonical_name_counts.update(member.original_canonical_name_counts)

    merged = MergeComponent(
        merge_component_uid=new_uid,
        local_cluster_uids=local_cluster_uids,
        mentions=mentions,
        canonical_name_counts=canonical_name_counts,
        original_canonical_name_counts=original_canonical_name_counts,
    )
    merged.recompute()

    return merged, next_component_index


def merge_connected_component_groups(
    *,
    components: dict[str, MergeComponent],
    component_uid_groups: list[tuple[str, ...]],
    next_component_index: int,
) -> tuple[dict[str, MergeComponent], int, list[str]]:
    """Merge all connected-component groups produced by one sieve iteration."""
    if not component_uid_groups:
        return components, next_component_index, []

    consumed_uids: set[str] = set()
    new_components = dict(components)
    new_component_uids: list[str] = []

    ordered_groups = sorted(
        component_uid_groups,
        key=lambda group: min(component_document_order_key(components[uid]) for uid in group),
    )

    for group in ordered_groups:
        overlap = consumed_uids & set(group)
        if overlap:
            raise ValueError(
                "Connected component groups must be disjoint, but overlap was found: "
                f"{sorted(overlap)[:10]}"
            )

        merged, next_component_index = merge_component_uid_group(
            components=components,
            component_uids=group,
            next_component_index=next_component_index,
        )

        for uid in group:
            del new_components[uid]
            consumed_uids.add(uid)

        new_components[merged.merge_component_uid] = merged
        new_component_uids.append(merged.merge_component_uid)

    return new_components, next_component_index, new_component_uids


def run_sieve_connected_components_until_stability(
    *,
    components: dict[str, MergeComponent],
    sieve_fn: Callable[..., SieveEvaluation],
    next_component_index: int,
    verbose: bool,
    enable_global_merge_veto: bool = True,
) -> tuple[dict[str, MergeComponent], int]:
    """Run one positive sieve using graph connected components until stable.

    This replaces the previous greedy aggregative strategy. Each iteration:
      1. evaluates the sieve on the current components;
      2. applies the same global veto layer to pairwise edges;
      3. builds an undirected graph from the surviving accepted edges;
      4. merges every non-trivial connected component in one deterministic pass.
    """
    merge_step = 0

    while True:
        components_before = len(components)

        indexes = build_current_component_indexes(components)
        evaluation = sieve_fn(components=components, indexes=indexes)

        accepted_edges = non_vetoed_accepted_edges(
            components=components,
            accepted_edges=evaluation.accepted_edges,
            enable_global_merge_veto=enable_global_merge_veto,
        )

        component_uid_groups = connected_component_uid_groups_from_edges(accepted_edges)

        if not component_uid_groups:
            if verbose:
                print(
                    f"[global-coref][{evaluation.sieve_name}] "
                    f"stable after {merge_step} connected-component merge steps; "
                    f"candidate_pairs={evaluation.candidate_pairs}; "
                    f"accepted_edges={len(evaluation.accepted_edges)}; "
                    f"non_vetoed_edges={len(accepted_edges)}"
                )
            break

        components, next_component_index, new_component_uids = merge_connected_component_groups(
            components=components,
            component_uid_groups=component_uid_groups,
            next_component_index=next_component_index,
        )

        merge_step += 1
        components_after = len(components)

        if verbose:
            group_sizes = sorted((len(group) for group in component_uid_groups), reverse=True)
            print(
                f"[global-coref][{evaluation.sieve_name}][cc-step {merge_step}] "
                f"components {components_before} -> {components_after}; "
                f"candidate_pairs={evaluation.candidate_pairs}; "
                f"accepted_edges={len(evaluation.accepted_edges)}; "
                f"non_vetoed_edges={len(accepted_edges)}; "
                f"connected_components={len(component_uid_groups)}; "
                f"group_sizes={group_sizes[:20]}; "
                f"new_components={new_component_uids[:20]}"
            )

    return components, next_component_index


# Backwards-compatible internal alias. No external caller should rely on this,
# but keeping the name prevents breakage in ad-hoc notebooks that imported it.
run_sieve_aggregative_until_stability = run_sieve_connected_components_until_stability


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def assign_global_cluster_uids(
    components: dict[str, MergeComponent],
) -> dict[str, str]:
    """Assign stable document-order global cluster IDs."""
    ordered_components = sorted(components.values(), key=component_document_order_key)

    return {
        component.merge_component_uid: f"global_cluster_{index:06d}"
        for index, component in enumerate(ordered_components)
    }


def choose_representative_canonical_name(component: MergeComponent) -> str:
    """Choose a human-readable canonical name for the exported global cluster."""
    if not component.original_canonical_name_counts:
        return ""
    return component.original_canonical_name_counts.most_common(1)[0][0]


def split_components_by_mention_percentile(
    components: dict[str, MergeComponent],
    *,
    percentile: float,
    min_mentions: int | None = None,
    min_kept_clusters: int | None = None,
) -> tuple[dict[str, MergeComponent], dict[str, MergeComponent], int]:
    """Split components into kept/rejected by final mention-count salience.

    Policy:
        - compute the requested quantile over final global component sizes;
        - use ceil(quantile) as the preferred cutoff;
        - keep components with len(component.mentions) >= cutoff;
        - if fewer than min_kept_clusters survive, keep the top-N components
          by mention count instead;
        - reject every component not included in the final kept set.

    Returns:
        kept_components,
        rejected_components,
        cutoff
    """

    if not 0 <= percentile <= 1:
        raise ValueError(f"percentile must be between 0 and 1, got {percentile}")

    if min_kept_clusters is not None and min_kept_clusters < 0:
        raise ValueError(
            f"min_kept_clusters must be non-negative or None, got {min_kept_clusters}"
        )

    if not components:
        return {}, {}, 0

    mention_counts = pd.Series(
        [len(component.mentions) for component in components.values()],
        dtype="float64",
    )

    cutoff = int(math.ceil(float(mention_counts.quantile(percentile))))

    if min_mentions is not None:
        cutoff = max(cutoff, int(min_mentions))

    kept_components = {
        component_uid: component
        for component_uid, component in components.items()
        if len(component.mentions) >= cutoff
    }

    if min_kept_clusters is not None and len(kept_components) < min_kept_clusters:
        target_kept_count = min(int(min_kept_clusters), len(components))

        ordered_components = sorted(
            components.items(),
            key=lambda item: (
                -len(item[1].mentions),
                component_document_order_key(item[1]),
                item[0],
            ),
        )

        kept_component_uids = {
            component_uid
            for component_uid, _component in ordered_components[:target_kept_count]
        }

        kept_components = {
            component_uid: component
            for component_uid, component in components.items()
            if component_uid in kept_component_uids
        }

    rejected_components = {
        component_uid: component
        for component_uid, component in components.items()
        if component_uid not in kept_components
    }

    return kept_components, rejected_components, cutoff


def export_global_clusters_csv(
    *,
    components: dict[str, MergeComponent],
    global_cluster_uid_by_component_uid: dict[str, str],
    output_path: Path,
) -> None:
    """Write global_clusters.csv."""
    rows = []

    for component_uid, component in components.items():
        global_cluster_uid = global_cluster_uid_by_component_uid[component_uid]

        chunk_min = min(component.chunk_indices) if component.chunk_indices else None
        chunk_max = max(component.chunk_indices) if component.chunk_indices else None

        rows.append(
            {
                "global_cluster_uid": global_cluster_uid,
                "n_local_clusters": len(component.local_cluster_uids),
                "n_mentions": len(component.mentions),
                "canonical_name": choose_representative_canonical_name(component),
                "canonical_names": "|".join(sorted(component.canonical_names)),
                "local_cluster_uids": "|".join(sorted(component.local_cluster_uids)),
                "chunk_min": chunk_min,
                "chunk_max": chunk_max,
            }
        )

    pd.DataFrame(rows).sort_values("global_cluster_uid").to_csv(
        output_path, index=False
    )


def export_global_mentions_csv(
    *,
    components: dict[str, MergeComponent],
    global_cluster_uid_by_component_uid: dict[str, str],
    output_path: Path,
) -> None:
    """Write global_mentions.csv with one row per local mention observation."""
    rows = []

    for component_uid, component in components.items():
        global_cluster_uid = global_cluster_uid_by_component_uid[component_uid]

        for mention in component.mentions:
            rows.append(
                {
                    "global_cluster_uid": global_cluster_uid,
                    "local_cluster_uid": mention.local_cluster_uid,
                    "local_mention_uid": mention.local_mention_uid,
                    "chunk_id": mention.chunk_id,
                    "chunk_index": mention.chunk_index,
                    "local_cluster_id": mention.local_cluster_id,
                    "local_mention_id": mention.local_mention_id,
                    "local_start": mention.local_start,
                    "local_end": mention.local_end,
                    "global_start": mention.global_start,
                    "global_end": mention.global_end,
                    "text": mention.text,
                    "head_local_i": mention.head_local_i,
                    "head_global_i": mention.head_global_i,
                    "zone": mention.zone,
                    "overlap_exact_span_key": mention.overlap_exact_span_key,
                }
            )

    pd.DataFrame(rows).sort_values(
        ["global_cluster_uid", "chunk_index", "global_start", "global_end"]
    ).to_csv(output_path, index=False)



def export_rejected_global_clusters_csv(
    *,
    rejected_components: dict[str, MergeComponent],
    pre_filter_global_cluster_uid_by_component_uid: dict[str, str],
    output_path: Path,
    salience_percentile: float,
    salience_cutoff: int,
) -> None:
    """Write rejected_global_clusters.csv as an audit artifact.

    This file is deliberately not consumed by annotator.py. It preserves enough
    information to inspect which final merge components were removed by the
    salience filter while keeping global_clusters.csv/global_mentions.csv fully
    compatible with the existing downstream contract.
    """

    columns = [
        "rejected_global_cluster_uid",
        "would_be_global_cluster_uid",
        "merge_component_uid",
        "rejection_reason",
        "salience_percentile",
        "salience_cutoff",
        "n_local_clusters",
        "n_mentions",
        "canonical_name",
        "canonical_names",
        "local_cluster_uids",
        "chunk_min",
        "chunk_max",
    ]

    rows = []

    ordered_rejected = sorted(
        rejected_components.items(),
        key=lambda item: component_document_order_key(item[1]),
    )

    for rejected_index, (component_uid, component) in enumerate(ordered_rejected):
        chunk_min = min(component.chunk_indices) if component.chunk_indices else None
        chunk_max = max(component.chunk_indices) if component.chunk_indices else None

        rows.append(
            {
                "rejected_global_cluster_uid": f"rejected_global_cluster_{rejected_index:06d}",
                "would_be_global_cluster_uid": pre_filter_global_cluster_uid_by_component_uid[
                    component_uid
                ],
                "merge_component_uid": component_uid,
                "rejection_reason": "below_global_cluster_salience_policy",
                "salience_percentile": salience_percentile,
                "salience_cutoff": salience_cutoff,
                "n_local_clusters": len(component.local_cluster_uids),
                "n_mentions": len(component.mentions),
                "canonical_name": choose_representative_canonical_name(component),
                "canonical_names": "|".join(sorted(component.canonical_names)),
                "local_cluster_uids": "|".join(sorted(component.local_cluster_uids)),
                "chunk_min": chunk_min,
                "chunk_max": chunk_max,
            }
        )

    pd.DataFrame(rows, columns=columns).sort_values(
        ["n_mentions", "would_be_global_cluster_uid"],
        ascending=[False, True],
    ).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_local_coreference_clusters(
    *,
    clusters_rows: list[dict],
    mentions_rows: list[dict],
    output_dir: str | Path,
    verbose: bool = True,
    enable_global_merge_veto: bool = True,
) -> GlobalCorefMergeResult:
    """Merge local chunk-level clusters into global clusters and export CSVs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validate_required_columns(clusters_rows=clusters_rows, mentions_rows=mentions_rows)

    local_clusters = build_local_clusters(
        clusters_rows=clusters_rows,
        mentions_rows=mentions_rows,
    )

    components = initialize_merge_components(local_clusters)
    next_component_index = len(components)

    if verbose:
        print(
            f"[global-coref] initialized {len(components)} merge components "
            f"from {len(local_clusters)} local clusters"
        )

    sieve_functions: list[Callable[..., SieveEvaluation]] = [
        sieve_exactCanonicalName_stitching,
        sieve_adjacent_overlap_exact_mention_without_canonical_equality,
        sieve_exact_mention_overlap_coefficient,
        sieve_same_clean_proper_name_anchor,
        sieve_canonical_head_alias_match,
    ]

    for sieve_fn in sieve_functions:
        components, next_component_index = run_sieve_connected_components_until_stability(
            components=components,
            sieve_fn=sieve_fn,
            next_component_index=next_component_index,
            verbose=verbose,
            enable_global_merge_veto=enable_global_merge_veto,
        )

    pre_filter_global_cluster_uid_by_component_uid = assign_global_cluster_uids(components)

    rejected_components: dict[str, MergeComponent] = {}
    salience_cutoff = 0

    if GLOBAL_CLUSTER_SALIENCE_FILTER_ENABLED:
        components, rejected_components, salience_cutoff = split_components_by_mention_percentile(
            components,
            percentile=GLOBAL_CLUSTER_SALIENCE_PERCENTILE,
            min_mentions=MIN_GLOBAL_CLUSTER_MENTIONS,
            min_kept_clusters=MIN_KEPT_GLOBAL_CLUSTERS,
        )

        if verbose:
            print(
                "[global-coref][salience-filter] "
                f"percentile={GLOBAL_CLUSTER_SALIENCE_PERCENTILE:.2f}; "
                f"cutoff={salience_cutoff}; "
                f"min_kept_clusters={MIN_KEPT_GLOBAL_CLUSTERS}; "
                f"kept={len(components)}; "
                f"rejected={len(rejected_components)}"
            )

    # Assign exported global IDs after filtering to preserve the existing
    # downstream expectation that global_clusters.csv contains compact,
    # document-order global_cluster_XXXXXX identifiers.
    global_cluster_uid_by_component_uid = assign_global_cluster_uids(components)

    export_global_clusters_csv(
        components=components,
        global_cluster_uid_by_component_uid=global_cluster_uid_by_component_uid,
        output_path=output_path / "global_clusters.csv",
    )

    export_global_mentions_csv(
        components=components,
        global_cluster_uid_by_component_uid=global_cluster_uid_by_component_uid,
        output_path=output_path / "global_mentions.csv",
    )

    export_rejected_global_clusters_csv(
        rejected_components=rejected_components,
        pre_filter_global_cluster_uid_by_component_uid=pre_filter_global_cluster_uid_by_component_uid,
        output_path=output_path / REJECTED_GLOBAL_CLUSTERS_FILENAME,
        salience_percentile=GLOBAL_CLUSTER_SALIENCE_PERCENTILE
        if GLOBAL_CLUSTER_SALIENCE_FILTER_ENABLED
        else 0.0,
        salience_cutoff=salience_cutoff,
    )

    if verbose:
        print(f"[global-coref] exported {output_path / 'global_clusters.csv'}")
        print(f"[global-coref] exported {output_path / 'global_mentions.csv'}")
        print(f"[global-coref] exported {output_path / REJECTED_GLOBAL_CLUSTERS_FILENAME}")
        print(f"[global-coref] final global clusters: {len(components)}")

    return GlobalCorefMergeResult(
        components=components,
        global_cluster_uid_by_component_uid=global_cluster_uid_by_component_uid,
        output_dir=output_path,
    )
