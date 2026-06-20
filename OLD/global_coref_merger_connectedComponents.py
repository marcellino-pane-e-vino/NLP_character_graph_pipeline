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
    After connected-component merging, one MergeComponent can wrap multiple
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
    """Accepted pairwise merge relation used as an edge for union-find."""

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
) -> GlobalCorefMergeResult:
    """CSV-path wrapper around merge_local_coreference_clusters."""
    clusters_rows = pd.read_csv(clusters_csv).to_dict("records")
    mentions_rows = pd.read_csv(mentions_csv).to_dict("records")

    return merge_local_coreference_clusters(
        clusters_rows=clusters_rows,
        mentions_rows=mentions_rows,
        output_dir=output_dir,
        verbose=verbose,
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
# Connected-component merging
# ---------------------------------------------------------------------------


class UnionFind:
    """Small union-find implementation for connected components."""

    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def groups(self) -> list[set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)

        for item in self.parent:
            grouped[self.find(item)].add(item)

        return list(grouped.values())


def merge_connected_components(
    *,
    components: dict[str, MergeComponent],
    accepted_edges: list[AcceptedMergeEdge],
    next_component_index: int,
) -> tuple[dict[str, MergeComponent], int, int]:
    """Merge every connected group induced by accepted edges."""
    if not accepted_edges:
        return components, next_component_index, 0

    involved_component_uids: set[str] = set()

    for edge in accepted_edges:
        involved_component_uids.add(edge.left_component_uid)
        involved_component_uids.add(edge.right_component_uid)

    union_find = UnionFind(involved_component_uids)

    for edge in accepted_edges:
        union_find.union(edge.left_component_uid, edge.right_component_uid)

    merge_groups = [group for group in union_find.groups() if len(group) >= 2]

    if not merge_groups:
        return components, next_component_index, 0

    new_components = dict(components)

    for group in merge_groups:
        new_uid = f"merge_component_{next_component_index:06d}"
        next_component_index += 1

        merged_local_cluster_uids: set[str] = set()
        merged_mentions: list[LocalMention] = []
        merged_canonical_counts: Counter[str] = Counter()
        merged_original_canonical_counts: Counter[str] = Counter()

        for old_uid in group:
            old = new_components[old_uid]
            merged_local_cluster_uids.update(old.local_cluster_uids)
            merged_mentions.extend(old.mentions)
            merged_canonical_counts.update(old.canonical_name_counts)
            merged_original_canonical_counts.update(old.original_canonical_name_counts)

        merged = MergeComponent(
            merge_component_uid=new_uid,
            local_cluster_uids=merged_local_cluster_uids,
            mentions=merged_mentions,
            canonical_name_counts=merged_canonical_counts,
            original_canonical_name_counts=merged_original_canonical_counts,
        )
        merged.recompute()

        for old_uid in group:
            del new_components[old_uid]

        new_components[new_uid] = merged

    return new_components, next_component_index, len(merge_groups)


def run_sieve_until_stability(
    *,
    components: dict[str, MergeComponent],
    sieve_fn: Callable[..., SieveEvaluation],
    next_component_index: int,
    verbose: bool,
) -> tuple[dict[str, MergeComponent], int]:
    """Run one sieve repeatedly until it produces no more merges."""
    round_index = 0

    while True:
        round_index += 1
        components_before = len(components)

        indexes = build_current_component_indexes(components)
        evaluation = sieve_fn(components=components, indexes=indexes)

        if not evaluation.accepted_edges:
            if verbose:
                print(
                    f"[global-coref][{evaluation.sieve_name}] "
                    f"stable after {round_index - 1} merge rounds"
                )
            break

        components, next_component_index, merged_groups = merge_connected_components(
            components=components,
            accepted_edges=evaluation.accepted_edges,
            next_component_index=next_component_index,
        )

        components_after = len(components)

        if verbose:
            print(
                f"[global-coref][{evaluation.sieve_name}][round {round_index}] "
                f"components {components_before} -> {components_after}; "
                f"candidate_pairs={evaluation.candidate_pairs}; "
                f"accepted_edges={len(evaluation.accepted_edges)}; "
                f"merged_groups={merged_groups}"
            )

        if merged_groups == 0:
            break

    return components, next_component_index


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_local_coreference_clusters(
    *,
    clusters_rows: list[dict],
    mentions_rows: list[dict],
    output_dir: str | Path,
    verbose: bool = True,
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

    for sieve_fn in sieve_functions:
        components, next_component_index = run_sieve_until_stability(
            components=components,
            sieve_fn=sieve_fn,
            next_component_index=next_component_index,
            verbose=verbose,
        )

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

    if verbose:
        print(f"[global-coref] exported {output_path / 'global_clusters.csv'}")
        print(f"[global-coref] exported {output_path / 'global_mentions.csv'}")
        print(f"[global-coref] final global clusters: {len(components)}")

    return GlobalCorefMergeResult(
        components=components,
        global_cluster_uid_by_component_uid=global_cluster_uid_by_component_uid,
        output_dir=output_path,
    )
