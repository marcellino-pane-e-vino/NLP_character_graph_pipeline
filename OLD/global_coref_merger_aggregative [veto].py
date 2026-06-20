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
        """Whether this mention should be used by Sieve 2.

        Sieve 2 should compare nominal/proper mentions, not standalone
        pronominal mentions. This keeps mentions such as "the girl",
        "her aunt", "Dorothy", and "the Tin Woodman", while excluding
        standalone mentions such as "she", "her", "they", and "you".
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

    Initially, one MergeComponent wraps one LocalCorefCluster.
    After aggregative merging, one MergeComponent can wrap multiple
    original local clusters.
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

        self.proper_name_anchors = extract_proper_name_anchors(self)

        self.chunk_indices = {mention.chunk_index for mention in self.mentions}


@dataclass(frozen=True)
class AcceptedMergeEdge:
    """Accepted pairwise merge relation proposed by a sieve."""

    sieve_name: str
    left_component_uid: str
    right_component_uid: str
    reason: str
    metrics: dict[str, int | float | str]


@dataclass(frozen=True)
class SieveEvaluation:
    """Result of evaluating one sieve once."""

    sieve_name: str
    candidate_pairs: int
    accepted_edges: list[AcceptedMergeEdge]


@dataclass(frozen=True)
class VetoDecision:
    """Decision returned by the global merge veto layer."""

    is_vetoed: bool
    reason: str
    metrics: dict[str, int | float | str]


@dataclass(frozen=True)
class RejectedMergeEdge:
    """Accepted merge edge rejected by the global veto layer."""

    sieve_name: str
    left_component_uid: str
    right_component_uid: str
    positive_reason: str
    positive_metrics: dict[str, int | float | str]
    veto_reason: str
    veto_metrics: dict[str, int | float | str]


@dataclass
class GlobalCorefMergeResult:
    """Returned result for notebook inspection."""

    components: dict[str, MergeComponent]
    global_cluster_uid_by_component_uid: dict[str, str]
    output_dir: Optional[Path]
    rejected_merge_edges: list[RejectedMergeEdge] = field(default_factory=list)


@dataclass(frozen=True)
class CurrentComponentIndexes:
    """Sparse indexes used to generate candidate pairs for the current state."""

    by_canonical_and_overlap_key: dict[tuple[str, MentionKey], set[str]]
    by_mention_key: dict[MentionKey, set[str]]
    by_relevant_mention_key: dict[MentionKey, set[str]]
    by_proper_name_anchor: dict[str, set[str]]


# ---------------------------------------------------------------------------
# Normalization helpers
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

# Heads that are too generic to function as strong entity evidence.
# These are intentionally lowercase and language-specific because the upstream
# Maverick output used here is English.
GENERIC_ENTITY_HEADS = {
    "boy",
    "boys",
    "child",
    "children",
    "city",
    "cities",
    "country",
    "countries",
    "father",
    "fathers",
    "friend",
    "friends",
    "girl",
    "girls",
    "home",
    "house",
    "land",
    "lands",
    "man",
    "men",
    "mother",
    "mothers",
    "one",
    "ones",
    "other",
    "others",
    "people",
    "person",
    "place",
    "places",
    "queen",
    "queens",
    "road",
    "roads",
    "thing",
    "things",
    "traveler",
    "travelers",
    "traveller",
    "travellers",
    "uncle",
    "uncles",
    "aunt",
    "aunts",
    "way",
    "ways",
    "woman",
    "women",
}

GENERIC_MENTION_TEXTS = {
    "a man",
    "the man",
    "a woman",
    "the woman",
    "a girl",
    "the girl",
    "a boy",
    "the boy",
    "the people",
    "the others",
    "the other",
    "his friends",
    "her friends",
    "their friends",
    "the road",
    "the country",
    "this country",
    "the land",
    "this land",
    "the city",
    "the house",
    "home",
    "no one",
    "someone",
    "somebody",
    "anyone",
    "anybody",
}

# Conservative, editable alias groups. These prevent the veto from blocking
# obvious alias pairs while still allowing the positive sieve to decide whether
# there is enough merge evidence.
CANONICAL_ALIAS_GROUPS = (
    frozenset({"tin woodman", "woodman"}),
    frozenset({"cowardly lion", "lion"}),
    frozenset({"wicked witch", "witch"}),
    frozenset({"wonderful wizard", "wizard", "oz"}),
    frozenset({"aunt em", "em"}),
    frozenset({"uncle henry", "henry"}),
)

ALIAS_GROUP_BY_PHRASE = {
    phrase: "|".join(sorted(group))
    for group in CANONICAL_ALIAS_GROUPS
    for phrase in group
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


def canonical_content_tokens(text: str) -> frozenset[str]:
    """Return simple content-token signature for a canonical/mention string."""
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    tokens = [token for token in raw_tokens if token not in CANONICAL_STOP_TOKENS]
    return frozenset(tokens)


def ordered_content_tokens(text: str) -> tuple[str, ...]:
    """Return ordered content tokens for phrase-level heuristics."""
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    return tuple(token for token in raw_tokens if token not in CANONICAL_STOP_TOKENS)


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


def canonical_head(text: str) -> str:
    """Return the final content token, lightly singularized."""
    tokens = ordered_content_tokens(text)
    if not tokens:
        return ""
    return simple_singularize(tokens[-1])


def is_generic_head(head: str) -> bool:
    """Whether a head is too generic to be strong entity evidence."""
    if not head:
        return True
    return head in GENERIC_ENTITY_HEADS_NORMALIZED


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


def alias_group_key(phrase: str) -> str:
    """Return a stable alias-group key for explicit aliases, else the phrase."""
    normalized_phrase = normalize_entity_phrase(phrase)
    return ALIAS_GROUP_BY_PHRASE.get(normalized_phrase, normalized_phrase)


def are_alias_compatible(left_phrase: str, right_phrase: str) -> bool:
    """Conservative phrase compatibility used by the veto layer.

    This function does not itself authorize a merge. It only prevents the veto
    from blocking obvious alias variants once a positive sieve has already found
    merge evidence.
    """
    left_norm = normalize_entity_phrase(left_phrase)
    right_norm = normalize_entity_phrase(right_phrase)

    if not left_norm or not right_norm:
        return False

    if left_norm == right_norm:
        return True

    if alias_group_key(left_norm) == alias_group_key(right_norm):
        return True

    left_tokens = set(ordered_content_tokens(left_norm))
    right_tokens = set(ordered_content_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False

    left_head = canonical_head(left_norm)
    right_head = canonical_head(right_norm)

    if left_head and left_head == right_head and not is_generic_head(left_head):
        if left_tokens <= right_tokens or right_tokens <= left_tokens:
            return True
        token_jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        return token_jaccard >= 0.67

    return False


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

    V1 intentionally uses canonical names only, not mention texts, to avoid noisy
    mention-level proper-name candidates.
    """
    anchors = set()

    for original_name in component.original_canonical_name_counts:
        if looks_like_proper_name(original_name):
            anchors.add(normalize_text(original_name))

    return anchors


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
    by_proper_name_anchor: dict[str, set[str]] = defaultdict(set)

    for component_uid, component in components.items():
        for mention_key in component.mention_keys:
            by_mention_key[mention_key].add(component_uid)

        for mention_key in component.relevant_mention_keys:
            by_relevant_mention_key[mention_key].add(component_uid)

        for overlap_key in component.overlap_mention_keys:
            for canonical_name in component.canonical_names:
                by_canonical_and_overlap_key[(canonical_name, overlap_key)].add(
                    component_uid
                )

        for anchor in component.proper_name_anchors:
            by_proper_name_anchor[anchor].add(component_uid)

    return CurrentComponentIndexes(
        by_canonical_and_overlap_key=dict(by_canonical_and_overlap_key),
        by_mention_key=dict(by_mention_key),
        by_relevant_mention_key=dict(by_relevant_mention_key),
        by_proper_name_anchor=dict(by_proper_name_anchor),
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
# Sieve functions
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


def sieve_exact_mention_overlap_coefficient(
    *,
    components: dict[str, MergeComponent],
    indexes: CurrentComponentIndexes,
    min_shared_mentions: int = 2,
    overlap_coefficient_threshold: float = 0.75,
) -> SieveEvaluation:
    """Sieve 2: relevant exact mention-key overlap coefficient.

    The coefficient is computed only on relevant nominal/proper mentions.
    Standalone pronominal mentions are ignored for both candidate generation
    and overlap-coefficient computation.
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
    """Sieve 3: both components have exactly the same single proper-name anchor."""
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



# ---------------------------------------------------------------------------
# Global merge veto layer
# ---------------------------------------------------------------------------


def _format_set(values: Iterable[str]) -> str:
    """Serialize a small set/list of strings for metrics and logs."""
    return "|".join(sorted(str(value) for value in values if str(value)))


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
        if are_alias_compatible(canonical_name, normalized_phrase):
            return True

    for mention_text in component_relevant_mention_text_counts(component):
        if are_alias_compatible(mention_text, normalized_phrase):
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

    # A generic head may still be strong if there is a distinctive modifier,
    # for example "emerald city" even though "city" alone is generic.
    return len(tokens) >= 2 and any(token not in GENERIC_ENTITY_HEADS for token in tokens[:-1])


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
        are_alias_compatible(left_anchor, right_anchor)
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


def veto_incompatible_dominant_anchors(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Block merges whose dominant strong canonical identities conflict."""
    left_anchor = component_dominant_strong_canonical_anchor(left)
    right_anchor = component_dominant_strong_canonical_anchor(right)

    if not left_anchor or not right_anchor:
        return VetoDecision(False, "dominant_anchor_check_not_applicable", {})

    if are_alias_compatible(left_anchor, right_anchor):
        return VetoDecision(False, "dominant_anchors_compatible", {})

    # Exact same canonical-name stitching is intentionally allowed to pass this
    # veto because the positive evidence is conservative and local to overlap.
    if edge.sieve_name == "exactMention_stitching":
        return VetoDecision(False, "exact_canonical_stitching_override", {})

    return VetoDecision(
        True,
        "incompatible_dominant_canonical_anchors",
        {
            "left_dominant_anchor": left_anchor,
            "right_dominant_anchor": right_anchor,
            "left_canonical_names": _format_set(left.canonical_names),
            "right_canonical_names": _format_set(right.canonical_names),
        },
    )


def veto_disjoint_strong_anchor_sets(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Block merges between components with strong but incompatible anchor sets."""
    left_anchors = component_strong_canonical_anchors(left)
    right_anchors = component_strong_canonical_anchors(right)

    if not left_anchors or not right_anchors:
        return VetoDecision(False, "strong_anchor_set_check_not_applicable", {})

    if any(
        are_alias_compatible(left_anchor, right_anchor)
        for left_anchor in left_anchors
        for right_anchor in right_anchors
    ):
        return VetoDecision(False, "strong_anchor_sets_have_compatible_pair", {})

    if edge.sieve_name == "exactMention_stitching":
        return VetoDecision(False, "exact_canonical_stitching_override", {})

    return VetoDecision(
        True,
        "disjoint_incompatible_strong_anchor_sets",
        {
            "left_strong_anchors": _format_set(left_anchors),
            "right_strong_anchors": _format_set(right_anchors),
        },
    )


def veto_generic_only_shared_evidence(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Block edges supported only by generic mention text overlap."""
    if canonical_sets_are_compatible(left, right):
        return VetoDecision(False, "canonical_compatibility_override", {})

    shared_texts = edge_shared_relevant_mention_texts(left, right)
    if not shared_texts:
        return VetoDecision(False, "no_shared_relevant_mentions", {})

    generic_shared_texts = {
        text for text in shared_texts if is_generic_nominal_text(text)
    }

    if generic_shared_texts == shared_texts:
        return VetoDecision(
            True,
            "generic_only_shared_mention_evidence",
            {
                "shared_texts": _format_set(shared_texts),
                "generic_shared_texts": _format_set(generic_shared_texts),
            },
        )

    return VetoDecision(False, "non_generic_shared_evidence_present", {})


def veto_asymmetric_pollution(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Block one-way evidence that looks like polluted mention leakage."""
    left_anchor = component_dominant_strong_canonical_anchor(left)
    right_anchor = component_dominant_strong_canonical_anchor(right)

    if not left_anchor or not right_anchor:
        return VetoDecision(False, "asymmetric_pollution_check_not_applicable", {})

    if are_alias_compatible(left_anchor, right_anchor):
        return VetoDecision(False, "dominant_anchors_compatible", {})

    left_contains_right = component_contains_phrase(left, right_anchor)
    right_contains_left = component_contains_phrase(right, left_anchor)

    if left_contains_right != right_contains_left:
        return VetoDecision(
            True,
            "asymmetric_pollution_evidence",
            {
                "left_dominant_anchor": left_anchor,
                "right_dominant_anchor": right_anchor,
                "left_contains_right_anchor": int(left_contains_right),
                "right_contains_left_anchor": int(right_contains_left),
            },
        )

    return VetoDecision(False, "asymmetric_pollution_not_detected", {})


def veto_canonical_entropy_explosion(
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Block weak edges that would introduce too many strong identities."""
    left_groups = {alias_group_key(anchor) for anchor in component_strong_canonical_anchors(left)}
    right_groups = {alias_group_key(anchor) for anchor in component_strong_canonical_anchors(right)}
    merged_groups = left_groups | right_groups

    if len(merged_groups) <= 2:
        return VetoDecision(False, "canonical_entropy_within_limit", {})

    strongest_side = max(len(left_groups), len(right_groups))
    increases_identity_space = len(merged_groups) > strongest_side
    if not increases_identity_space:
        return VetoDecision(False, "canonical_entropy_not_increased", {})

    overlap_coefficient = _metric_as_float(edge.metrics, "overlap_coefficient", 0.0)
    shared_relevant = _metric_as_float(edge.metrics, "shared_relevant_mentions", 0.0)
    shared_overlap = _metric_as_float(edge.metrics, "shared_overlap_mention_count", 0.0)

    very_strong_edge = (
        edge.sieve_name == "exactMention_stitching"
        or (overlap_coefficient >= 0.95 and shared_relevant >= 3)
        or shared_overlap >= 3
    )

    if very_strong_edge:
        return VetoDecision(False, "strong_edge_overrides_entropy_warning", {})

    return VetoDecision(
        True,
        "canonical_entropy_explosion",
        {
            "left_anchor_groups": _format_set(left_groups),
            "right_anchor_groups": _format_set(right_groups),
            "merged_anchor_group_count": len(merged_groups),
            "overlap_coefficient": overlap_coefficient,
            "shared_relevant_mentions": shared_relevant,
            "shared_overlap_mentions": shared_overlap,
        },
    )


def evaluate_global_merge_veto(
    *,
    left: MergeComponent,
    right: MergeComponent,
    edge: AcceptedMergeEdge,
) -> VetoDecision:
    """Return a veto decision for one positive merge edge.

    The veto is a negative safety layer. It never proposes merges. It can only
    reject an edge that a positive sieve already accepted.
    """
    veto_checks = [
        veto_incompatible_dominant_anchors,
        veto_disjoint_strong_anchor_sets,
        veto_generic_only_shared_evidence,
        veto_asymmetric_pollution,
        veto_canonical_entropy_explosion,
    ]

    for veto_check in veto_checks:
        decision = veto_check(left, right, edge)
        if decision.is_vetoed:
            return decision

    return VetoDecision(False, "merge_allowed", {})


def apply_global_merge_veto(
    *,
    components: dict[str, MergeComponent],
    accepted_edges: list[AcceptedMergeEdge],
    enable_global_merge_veto: bool,
) -> tuple[list[AcceptedMergeEdge], list[RejectedMergeEdge]]:
    """Filter accepted edges through the global merge veto layer."""
    safe_edges: list[AcceptedMergeEdge] = []
    rejected_edges: list[RejectedMergeEdge] = []

    for edge in accepted_edges:
        if (
            edge.left_component_uid not in components
            or edge.right_component_uid not in components
            or edge.left_component_uid == edge.right_component_uid
        ):
            continue

        if not enable_global_merge_veto:
            safe_edges.append(edge)
            continue

        left = components[edge.left_component_uid]
        right = components[edge.right_component_uid]
        decision = evaluate_global_merge_veto(left=left, right=right, edge=edge)

        if decision.is_vetoed:
            rejected_edges.append(
                RejectedMergeEdge(
                    sieve_name=edge.sieve_name,
                    left_component_uid=edge.left_component_uid,
                    right_component_uid=edge.right_component_uid,
                    positive_reason=edge.reason,
                    positive_metrics=edge.metrics,
                    veto_reason=decision.reason,
                    veto_metrics=decision.metrics,
                )
            )
            continue

        safe_edges.append(edge)

    return safe_edges, rejected_edges


def choose_best_non_vetoed_accepted_merge_edge(
    *,
    components: dict[str, MergeComponent],
    accepted_edges: list[AcceptedMergeEdge],
    enable_global_merge_veto: bool,
) -> tuple[Optional[AcceptedMergeEdge], list[RejectedMergeEdge], int]:
    """Choose the strongest currently safe edge.

    Edges are inspected in the same deterministic strength order used by the
    original aggregative merger. This avoids evaluating the veto over every
    accepted edge on every merge step when the first strong edge is already safe.
    If every edge is vetoed, the sieve is stable for the current component state.
    """
    valid_edges = [
        edge
        for edge in accepted_edges
        if edge.left_component_uid in components
        and edge.right_component_uid in components
        and edge.left_component_uid != edge.right_component_uid
    ]

    valid_edges = sorted(
        valid_edges,
        key=lambda edge: accepted_merge_edge_sort_key(edge, components),
    )

    if not enable_global_merge_veto:
        if not valid_edges:
            return None, [], 0
        return valid_edges[0], [], 1

    rejected_edges: list[RejectedMergeEdge] = []
    checked_edges = 0

    for edge in valid_edges:
        checked_edges += 1
        left = components[edge.left_component_uid]
        right = components[edge.right_component_uid]
        decision = evaluate_global_merge_veto(left=left, right=right, edge=edge)

        if decision.is_vetoed:
            rejected_edges.append(
                RejectedMergeEdge(
                    sieve_name=edge.sieve_name,
                    left_component_uid=edge.left_component_uid,
                    right_component_uid=edge.right_component_uid,
                    positive_reason=edge.reason,
                    positive_metrics=edge.metrics,
                    veto_reason=decision.reason,
                    veto_metrics=decision.metrics,
                )
            )
            continue

        return edge, rejected_edges, checked_edges

    return None, rejected_edges, checked_edges


# ---------------------------------------------------------------------------
# Aggregative merging
# ---------------------------------------------------------------------------


def component_document_order_key(component: MergeComponent) -> tuple[int, int, str]:
    """Stable deterministic component order used for tie-breaking."""
    chunk_min = min(component.chunk_indices) if component.chunk_indices else 10**12
    mention_start_min = (
        min(mention.global_start for mention in component.mentions)
        if component.mentions
        else 10**12
    )
    return (chunk_min, mention_start_min, component.merge_component_uid)


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


def merge_edge_strength(edge: AcceptedMergeEdge) -> tuple[float, float, float, float]:
    """Return a deterministic strength tuple for ranking accepted merge edges.

    The tuple is ordered from most important to least important. Higher is
    better. The function is intentionally sieve-aware because different sieves
    expose different evidence metrics.

    This score does not decide whether an edge is valid. Validity is still
    decided exclusively by the sieve predicates. The score only decides which
    accepted edge is aggregated first when multiple accepted edges are available.
    """
    if edge.sieve_name == "exactMention_stitching":
        return (
            _metric_as_float(edge.metrics, "shared_overlap_mention_count"),
            _metric_as_float(edge.metrics, "shared_canonical_count"),
            0.0,
            0.0,
        )

    if edge.sieve_name == "exact_mention_overlap_coefficient":
        return (
            _metric_as_float(edge.metrics, "overlap_coefficient"),
            _metric_as_float(edge.metrics, "shared_relevant_mentions"),
            min(
                _metric_as_float(edge.metrics, "left_relevant_mentions"),
                _metric_as_float(edge.metrics, "right_relevant_mentions"),
            ),
            0.0,
        )

    if edge.sieve_name == "same_clean_proper_name_anchor":
        return (1.0, 0.0, 0.0, 0.0)

    return (0.0, 0.0, 0.0, 0.0)


def accepted_merge_edge_sort_key(
    edge: AcceptedMergeEdge,
    components: dict[str, MergeComponent],
) -> tuple:
    """Sort accepted edges by evidence strength, then by document order.

    Lower sort keys are preferred, so strength values are negated.
    """
    strength = merge_edge_strength(edge)

    left = components[edge.left_component_uid]
    right = components[edge.right_component_uid]

    left_key = component_document_order_key(left)
    right_key = component_document_order_key(right)

    if right_key < left_key:
        left_key, right_key = right_key, left_key

    return (
        -strength[0],
        -strength[1],
        -strength[2],
        -strength[3],
        left_key,
        right_key,
        edge.left_component_uid,
        edge.right_component_uid,
    )


def choose_best_accepted_merge_edge(
    *,
    components: dict[str, MergeComponent],
    accepted_edges: list[AcceptedMergeEdge],
) -> Optional[AcceptedMergeEdge]:
    """Choose the single best currently valid accepted edge.

    The aggregative strategy intentionally does not collapse the transitive
    closure of all accepted pairwise edges. It selects one edge, merges that
    pair, recomputes the aggregate component, rebuilds indexes, and asks the
    sieve again.

    Consequence: if A-B and B-C are both accepted in the same evaluation, this
    function merges only the strongest one first. The resulting aggregate must
    independently satisfy a later sieve evaluation before C is absorbed.
    """
    valid_edges = [
        edge
        for edge in accepted_edges
        if edge.left_component_uid in components
        and edge.right_component_uid in components
        and edge.left_component_uid != edge.right_component_uid
    ]

    if not valid_edges:
        return None

    return min(
        valid_edges,
        key=lambda edge: accepted_merge_edge_sort_key(edge, components),
    )


def merge_two_components(
    *,
    components: dict[str, MergeComponent],
    left_uid: str,
    right_uid: str,
    next_component_index: int,
) -> tuple[dict[str, MergeComponent], int, str]:
    """Merge exactly two active components into one new aggregate component."""
    if left_uid == right_uid:
        raise ValueError("Cannot merge a component with itself.")

    if left_uid not in components:
        raise KeyError(f"Unknown left component uid: {left_uid}")

    if right_uid not in components:
        raise KeyError(f"Unknown right component uid: {right_uid}")

    left = components[left_uid]
    right = components[right_uid]

    new_uid = f"merge_component_{next_component_index:06d}"
    next_component_index += 1

    merged = MergeComponent(
        merge_component_uid=new_uid,
        local_cluster_uids=set(left.local_cluster_uids) | set(right.local_cluster_uids),
        mentions=list(left.mentions) + list(right.mentions),
        canonical_name_counts=left.canonical_name_counts + right.canonical_name_counts,
        original_canonical_name_counts=(
            left.original_canonical_name_counts
            + right.original_canonical_name_counts
        ),
    )
    merged.recompute()

    new_components = dict(components)
    del new_components[left_uid]
    del new_components[right_uid]
    new_components[new_uid] = merged

    return new_components, next_component_index, new_uid


def run_sieve_aggregative_until_stability(
    *,
    components: dict[str, MergeComponent],
    sieve_fn: Callable[..., SieveEvaluation],
    next_component_index: int,
    verbose: bool,
) -> tuple[dict[str, MergeComponent], int]:
    """Run one sieve greedily until no more aggregative pair merge is valid."""
    merge_step = 0

    while True:
        components_before = len(components)

        indexes = build_current_component_indexes(components)
        evaluation = sieve_fn(components=components, indexes=indexes)

        best_edge = choose_best_accepted_merge_edge(
            components=components,
            accepted_edges=evaluation.accepted_edges,
        )

        if best_edge is None:
            if verbose:
                print(
                    f"[global-coref][{evaluation.sieve_name}] "
                    f"stable after {merge_step} aggregative merge steps"
                )
            break

        components, next_component_index, new_uid = merge_two_components(
            components=components,
            left_uid=best_edge.left_component_uid,
            right_uid=best_edge.right_component_uid,
            next_component_index=next_component_index,
        )

        merge_step += 1
        components_after = len(components)

        if verbose:
            print(
                f"[global-coref][{evaluation.sieve_name}][step {merge_step}] "
                f"components {components_before} -> {components_after}; "
                f"candidate_pairs={evaluation.candidate_pairs}; "
                f"accepted_edges={len(evaluation.accepted_edges)}; "
                f"chosen_edge=({best_edge.left_component_uid}, "
                f"{best_edge.right_component_uid}); "
                f"new_component={new_uid}; "
                f"reason={best_edge.reason}; "
                f"metrics={best_edge.metrics}"
            )

    return components, next_component_index


def run_sieve_aggregative_until_stability_with_veto(
    *,
    components: dict[str, MergeComponent],
    sieve_fn: Callable[..., SieveEvaluation],
    next_component_index: int,
    verbose: bool,
    enable_global_merge_veto: bool = True,
) -> tuple[dict[str, MergeComponent], int, list[RejectedMergeEdge]]:
    """Run one sieve greedily with an optional global veto layer.

    Accepted edges rejected by the veto are discarded only for the current
    evaluation. After a real merge, components and indexes are rebuilt and all
    decisions are recomputed from scratch.
    """
    merge_step = 0
    all_rejected_edges: list[RejectedMergeEdge] = []
    seen_rejected_edge_keys: set[tuple[str, str, str, str]] = set()

    while True:
        components_before = len(components)

        indexes = build_current_component_indexes(components)
        evaluation = sieve_fn(components=components, indexes=indexes)

        best_edge, rejected_edges, checked_edges = (
            choose_best_non_vetoed_accepted_merge_edge(
                components=components,
                accepted_edges=evaluation.accepted_edges,
                enable_global_merge_veto=enable_global_merge_veto,
            )
        )
        for rejected_edge in rejected_edges:
            rejected_key = (
                rejected_edge.sieve_name,
                rejected_edge.left_component_uid,
                rejected_edge.right_component_uid,
                rejected_edge.veto_reason,
            )
            if rejected_key in seen_rejected_edge_keys:
                continue
            seen_rejected_edge_keys.add(rejected_key)
            all_rejected_edges.append(rejected_edge)

        if best_edge is None:
            if verbose:
                print(
                    f"[global-coref][{evaluation.sieve_name}] "
                    f"stable after {merge_step} aggregative merge steps; "
                    f"candidate_pairs={evaluation.candidate_pairs}; "
                    f"accepted_edges={len(evaluation.accepted_edges)}; "
                    f"checked_edges={checked_edges}; "
                    f"vetoed_edges={len(rejected_edges)}"
                )
            break

        components, next_component_index, new_uid = merge_two_components(
            components=components,
            left_uid=best_edge.left_component_uid,
            right_uid=best_edge.right_component_uid,
            next_component_index=next_component_index,
        )

        merge_step += 1
        components_after = len(components)

        if verbose:
            print(
                f"[global-coref][{evaluation.sieve_name}][step {merge_step}] "
                f"components {components_before} -> {components_after}; "
                f"candidate_pairs={evaluation.candidate_pairs}; "
                f"accepted_edges={len(evaluation.accepted_edges)}; "
                f"checked_edges={checked_edges}; "
                f"vetoed_edges={len(rejected_edges)}; "
                f"chosen_edge=({best_edge.left_component_uid}, "
                f"{best_edge.right_component_uid}); "
                f"new_component={new_uid}; "
                f"reason={best_edge.reason}; "
                f"metrics={best_edge.metrics}"
            )

    return components, next_component_index, all_rejected_edges


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def assign_global_cluster_uids(
    components: dict[str, MergeComponent],
) -> dict[str, str]:
    """Assign stable document-order global cluster IDs."""

    def component_sort_key(component: MergeComponent) -> tuple[int, int, str]:
        chunk_min = min(component.chunk_indices) if component.chunk_indices else 10**12
        mention_start_min = (
            min(mention.global_start for mention in component.mentions)
            if component.mentions
            else 10**12
        )
        return (chunk_min, mention_start_min, component.merge_component_uid)

    ordered_components = sorted(components.values(), key=component_sort_key)

    return {
        component.merge_component_uid: f"global_cluster_{index:06d}"
        for index, component in enumerate(ordered_components)
    }


def choose_representative_canonical_name(component: MergeComponent) -> str:
    """Choose a human-readable canonical name for the exported global cluster."""
    if not component.original_canonical_name_counts:
        return ""
    return component.original_canonical_name_counts.most_common(1)[0][0]


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



def export_global_merge_veto_log_csv(
    *,
    rejected_edges: list[RejectedMergeEdge],
    output_path: Path,
) -> None:
    """Write a diagnostic CSV for merge edges rejected by the global veto."""
    rows = []

    for index, rejected_edge in enumerate(rejected_edges):
        rows.append(
            {
                "rejected_edge_index": index,
                "sieve_name": rejected_edge.sieve_name,
                "left_component_uid": rejected_edge.left_component_uid,
                "right_component_uid": rejected_edge.right_component_uid,
                "positive_reason": rejected_edge.positive_reason,
                "positive_metrics": repr(rejected_edge.positive_metrics),
                "veto_reason": rejected_edge.veto_reason,
                "veto_metrics": repr(rejected_edge.veto_metrics),
            }
        )

    columns = [
        "rejected_edge_index",
        "sieve_name",
        "left_component_uid",
        "right_component_uid",
        "positive_reason",
        "positive_metrics",
        "veto_reason",
        "veto_metrics",
    ]

    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


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
        sieve_exact_mention_overlap_coefficient,
        sieve_same_clean_proper_name_anchor,
    ]

    rejected_merge_edges: list[RejectedMergeEdge] = []

    for sieve_fn in sieve_functions:
        components, next_component_index, sieve_rejected_edges = (
            run_sieve_aggregative_until_stability_with_veto(
                components=components,
                sieve_fn=sieve_fn,
                next_component_index=next_component_index,
                verbose=verbose,
                enable_global_merge_veto=enable_global_merge_veto,
            )
        )
        rejected_merge_edges.extend(sieve_rejected_edges)

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

    export_global_merge_veto_log_csv(
        rejected_edges=rejected_merge_edges,
        output_path=output_path / "global_merge_veto_log.csv",
    )

    if verbose:
        print(f"[global-coref] exported {output_path / 'global_clusters.csv'}")
        print(f"[global-coref] exported {output_path / 'global_mentions.csv'}")
        print(f"[global-coref] exported {output_path / 'global_merge_veto_log.csv'}")
        print(f"[global-coref] rejected merge edges: {len(rejected_merge_edges)}")
        print(f"[global-coref] final global clusters: {len(components)}")

    return GlobalCorefMergeResult(
        components=components,
        global_cluster_uid_by_component_uid=global_cluster_uid_by_component_uid,
        output_dir=output_path,
        rejected_merge_edges=rejected_merge_edges,
    )
