"""
filter_mentions_foreign_anchor.py
---------------------------------
Destructive mention filter for local coreference clusters.

New strategy:
  - Build a conservative bank of strong anchors from cluster canonical_name values.
  - Anchors are selected after canonicization, using the top X percentile by n_mentions.
  - For each mention, keep it unless it matches a strong anchor that belongs to a
    different canonical-name family from the mention's own cluster canonical_name.

This intentionally inverts the older strategy:
  OLD: drop non-pronoun mentions that do NOT match their own canonical_name.
  NEW: drop non-pronoun mentions that DO match a selected canonical_name of another
       entity/anchor family.

Example:
  Cluster canonical_name = "Dorothy"
  Mention text = "the Scarecrow"
  Strong anchor bank contains "Scarecrow"
  => mention is dropped from Dorothy's cluster.

Usage:
    python filter_mentions_foreign_anchor.py \
        --mentions mentions.csv \
        --clusters clusters.csv \
        --output mentions_filtered.csv \
        --dropped-output mentions_dropped.csv \
        --anchor-percentile 0.95 \
        --fuzzy-threshold 0.86
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Pronouns and generic labels
# ---------------------------------------------------------------------------

MASC_PRONOUNS = {"he", "him", "his", "himself"}
FEM_PRONOUNS = {"she", "her", "hers", "herself"}
NEUT_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "one",
}
ALL_PRONOUNS = MASC_PRONOUNS | FEM_PRONOUNS | NEUT_PRONOUNS

# Deliberately not capitalization-based.
# These are labels that are too generic to be safe destructive anchors.
# Keep this list conservative. Expand it only after inspecting real false drops.
DEFAULT_GENERIC_ANCHOR_KEYS = {
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "no one", "nobody", "person", "people", "persons",
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "friend", "friends", "companion", "companions",
    "traveler", "travelers", "stranger", "strangers",
    "thing", "things", "creature", "creatures", "animal", "animals",
    "road", "forest", "house", "room", "place", "country", "city", "town",
    "voice", "voices", "head", "hand", "hands", "eyes",
}

LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9]+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    key: str
    display_name: str
    n_mentions: int
    source_cluster_uids: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    score: float = 0.0
    method: str = "none"


@dataclass(frozen=True)
class ForeignAnchorHit:
    anchor_key: str
    anchor_display_name: str
    anchor_n_mentions: int
    score: float
    method: str


# ---------------------------------------------------------------------------
# Normalization / canonicization
# ---------------------------------------------------------------------------

def canonicize_text(text: object) -> str:
    """Normalize a mention/canonical label for surface-form matching.

    This is intentionally lexical, not semantic:
    - lowercase
    - remove leading English articles
    - remove possessive suffixes
    - collapse punctuation to spaces
    - collapse repeated whitespace
    """
    if pd.isna(text):
        return ""

    t = str(text).strip().lower()
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", t)

    # Remove leading articles repeatedly: "the the wizard" -> "wizard".
    previous = None
    while previous != t:
        previous = t
        t = LEADING_ARTICLE_RE.sub("", t).strip()

    t = NON_WORD_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def tokens_of(key: str) -> tuple[str, ...]:
    return tuple(tok for tok in key.split() if tok)


def is_pronoun_text(text: object) -> bool:
    return canonicize_text(text) in ALL_PRONOUNS


def load_generic_anchor_keys(path: str | None) -> set[str]:
    keys = set(DEFAULT_GENERIC_ANCHOR_KEYS)
    if not path:
        return keys

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        clean = canonicize_text(line)
        if clean and not clean.startswith("#"):
            keys.add(clean)
    return keys


def is_anchor_candidate(key: str, generic_anchor_keys: set[str]) -> bool:
    """Return whether a canonicalized label is safe enough to become an anchor."""
    if not key:
        return False
    if key in ALL_PRONOUNS:
        return False
    if key in generic_anchor_keys:
        return False
    if key.isdigit():
        return False

    # Reject labels that are only one very short token: e.g. "he", "x".
    toks = tokens_of(key)
    if len(toks) == 1 and len(toks[0]) <= 2:
        return False

    return True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def lexical_match(
    left_key: str,
    right_key: str,
    *,
    fuzzy_threshold: float,
    min_fuzzy_chars: int,
) -> MatchResult:
    """High-precision lexical matcher for mention-vs-anchor comparison.

    Matching order:
      1. exact canonicalized key
      2. token-subset match, never raw substring match
      3. SequenceMatcher ratio, gated by minimum string length

    Raw substring matching is intentionally avoided because it makes "man" match
    "woodman", which is destructive in this pipeline.
    """
    if not left_key or not right_key:
        return MatchResult(False)

    if left_key == right_key:
        return MatchResult(True, 1.0, "exact")

    left_tokens = set(tokens_of(left_key))
    right_tokens = set(tokens_of(right_key))
    if left_tokens and right_tokens:
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            # Jaccard is diagnostic only; subset match is already accepted.
            score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            return MatchResult(True, score, "token_subset")

    if min(len(left_key), len(right_key)) < min_fuzzy_chars:
        return MatchResult(False)

    score = SequenceMatcher(None, left_key, right_key).ratio()
    if score >= fuzzy_threshold:
        return MatchResult(True, score, "fuzzy")

    return MatchResult(False, score, "none")


def same_anchor_family(
    own_canonical_key: str,
    candidate_anchor_key: str,
    *,
    fuzzy_threshold: float,
    min_fuzzy_chars: int,
) -> bool:
    """Avoid treating spelling variants / equivalent labels as foreign anchors."""
    return lexical_match(
        own_canonical_key,
        candidate_anchor_key,
        fuzzy_threshold=fuzzy_threshold,
        min_fuzzy_chars=min_fuzzy_chars,
    ).matched


def best_foreign_anchor_hit(
    mention_key: str,
    own_canonical_key: str,
    anchors: list[Anchor],
    *,
    fuzzy_threshold: float,
    own_family_fuzzy_threshold: float,
    min_fuzzy_chars: int,
) -> ForeignAnchorHit | None:
    best: ForeignAnchorHit | None = None

    for anchor in anchors:
        if same_anchor_family(
            own_canonical_key,
            anchor.key,
            fuzzy_threshold=own_family_fuzzy_threshold,
            min_fuzzy_chars=min_fuzzy_chars,
        ):
            continue

        match = lexical_match(
            mention_key,
            anchor.key,
            fuzzy_threshold=fuzzy_threshold,
            min_fuzzy_chars=min_fuzzy_chars,
        )
        if not match.matched:
            continue

        hit = ForeignAnchorHit(
            anchor_key=anchor.key,
            anchor_display_name=anchor.display_name,
            anchor_n_mentions=anchor.n_mentions,
            score=match.score,
            method=match.method,
        )
        if best is None or hit.score > best.score:
            best = hit

    return best


# ---------------------------------------------------------------------------
# Anchor construction
# ---------------------------------------------------------------------------

def infer_cluster_mention_counts(mentions: pd.DataFrame, clusters: pd.DataFrame) -> pd.Series:
    if "n_mentions" in clusters.columns:
        return pd.to_numeric(clusters["n_mentions"], errors="coerce").fillna(0).astype(int)

    counts = mentions.groupby("local_cluster_uid").size()
    return clusters["local_cluster_uid"].map(counts).fillna(0).astype(int)


def build_anchor_bank(
    *,
    mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    anchor_percentile: float,
    min_anchor_mentions: int,
    generic_anchor_keys: set[str],
) -> list[Anchor]:
    required = {"local_cluster_uid", "canonical_name"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"clusters.csv is missing required columns: {sorted(missing)}")
    if "local_cluster_uid" not in mentions.columns:
        raise ValueError("mentions.csv is missing required column: local_cluster_uid")

    work = clusters.copy()
    work["local_cluster_uid"] = work["local_cluster_uid"].astype(str)
    work["canonical_key"] = work["canonical_name"].map(canonicize_text)
    work["cluster_n_mentions"] = infer_cluster_mention_counts(mentions, work)
    work = work[work["canonical_key"].map(lambda k: is_anchor_candidate(k, generic_anchor_keys))]

    if work.empty:
        return []

    grouped = (
        work.groupby("canonical_key", as_index=False)
        .agg(
            n_mentions=("cluster_n_mentions", "sum"),
            display_name=("canonical_name", lambda s: str(s.iloc[0])),
            source_cluster_uids=("local_cluster_uid", lambda s: tuple(sorted(map(str, s)))),
        )
    )

    grouped = grouped[grouped["n_mentions"] >= min_anchor_mentions]
    if grouped.empty:
        return []

    cutoff = grouped["n_mentions"].quantile(anchor_percentile)
    selected = grouped[grouped["n_mentions"] >= cutoff].sort_values(
        ["n_mentions", "canonical_key"], ascending=[False, True]
    )

    return [
        Anchor(
            key=str(row.canonical_key),
            display_name=str(row.display_name),
            n_mentions=int(row.n_mentions),
            source_cluster_uids=tuple(row.source_cluster_uids),
        )
        for row in selected.itertuples(index=False)
    ]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_mentions_by_foreign_anchors(
    *,
    mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    anchors: list[Anchor],
    fuzzy_threshold: float,
    own_family_fuzzy_threshold: float,
    min_fuzzy_chars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_mentions = {"local_cluster_uid", "text"}
    missing_mentions = required_mentions - set(mentions.columns)
    if missing_mentions:
        raise ValueError(f"mentions.csv is missing required columns: {sorted(missing_mentions)}")

    required_clusters = {"local_cluster_uid", "canonical_name"}
    missing_clusters = required_clusters - set(clusters.columns)
    if missing_clusters:
        raise ValueError(f"clusters.csv is missing required columns: {sorted(missing_clusters)}")

    cluster_lookup = clusters[["local_cluster_uid", "canonical_name"]].copy()
    cluster_lookup["local_cluster_uid"] = cluster_lookup["local_cluster_uid"].astype(str)
    cluster_lookup["own_canonical_key"] = cluster_lookup["canonical_name"].map(canonicize_text)

    merged = mentions.copy()
    merged["local_cluster_uid"] = merged["local_cluster_uid"].astype(str)
    merged = merged.merge(
        cluster_lookup[["local_cluster_uid", "canonical_name", "own_canonical_key"]],
        on="local_cluster_uid",
        how="left",
        validate="many_to_one",
    )

    keep_mask: list[bool] = []
    dropped_records: list[dict[str, object]] = []

    for row in merged.itertuples(index=True):
        mention_text = getattr(row, "text")
        mention_key = canonicize_text(mention_text)
        own_key = getattr(row, "own_canonical_key") or ""

        # Pronouns are intentionally not destructively removed by this layer.
        if is_pronoun_text(mention_text):
            keep_mask.append(True)
            continue

        hit = best_foreign_anchor_hit(
            mention_key,
            own_key,
            anchors,
            fuzzy_threshold=fuzzy_threshold,
            own_family_fuzzy_threshold=own_family_fuzzy_threshold,
            min_fuzzy_chars=min_fuzzy_chars,
        )

        if hit is None:
            keep_mask.append(True)
            continue

        keep_mask.append(False)
        dropped_records.append({
            "row_index": row.Index,
            "local_cluster_uid": getattr(row, "local_cluster_uid"),
            "cluster_canonical_name": getattr(row, "canonical_name"),
            "mention_text": mention_text,
            "mention_key": mention_key,
            "matched_anchor": hit.anchor_display_name,
            "matched_anchor_key": hit.anchor_key,
            "matched_anchor_n_mentions": hit.anchor_n_mentions,
            "match_score": hit.score,
            "match_method": hit.method,
        })

    filtered = merged.loc[keep_mask].drop(columns=["canonical_name", "own_canonical_key"])
    dropped = pd.DataFrame(dropped_records)
    return filtered, dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Destructively remove mentions that match strong foreign canonical-name anchors."
    )
    parser.add_argument("--mentions", required=True, help="Path to mentions.csv")
    parser.add_argument("--clusters", required=True, help="Path to clusters.csv")
    parser.add_argument("--output", required=True, help="Output path for filtered mentions.csv")
    parser.add_argument(
        "--dropped-output",
        default=None,
        help="Optional audit CSV containing dropped mentions and the anchor that caused the drop.",
    )
    parser.add_argument(
        "--anchors-output",
        default=None,
        help="Optional CSV containing the selected anchor bank.",
    )
    parser.add_argument(
        "--anchor-percentile",
        type=float,
        default=0.95,
        help="Select anchors with n_mentions >= this quantile after canonicization. 0.95 means top 5%%.",
    )
    parser.add_argument(
        "--min-anchor-mentions",
        type=int,
        default=2,
        help="Minimum total n_mentions required for a canonicalized label to be eligible as an anchor.",
    )
    parser.add_argument(
        "--generic-anchor-blacklist",
        default=None,
        help="Optional newline-separated list of extra generic labels to exclude from anchor selection.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.86,
        help="SequenceMatcher threshold for mention-vs-foreign-anchor fuzzy matching. Higher = stricter.",
    )
    parser.add_argument(
        "--own-family-fuzzy-threshold",
        type=float,
        default=0.90,
        help="Threshold used to avoid treating spelling variants of the own canonical name as foreign anchors.",
    )
    parser.add_argument(
        "--min-fuzzy-chars",
        type=int,
        default=5,
        help="Do not run fuzzy matching when either normalized string is shorter than this many characters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 <= args.anchor_percentile <= 1:
        raise ValueError("--anchor-percentile must be between 0 and 1")
    if not 0 <= args.fuzzy_threshold <= 1:
        raise ValueError("--fuzzy-threshold must be between 0 and 1")
    if not 0 <= args.own_family_fuzzy_threshold <= 1:
        raise ValueError("--own-family-fuzzy-threshold must be between 0 and 1")

    mentions = pd.read_csv(args.mentions)
    clusters = pd.read_csv(args.clusters)

    generic_anchor_keys = load_generic_anchor_keys(args.generic_anchor_blacklist)
    anchors = build_anchor_bank(
        mentions=mentions,
        clusters=clusters,
        anchor_percentile=args.anchor_percentile,
        min_anchor_mentions=args.min_anchor_mentions,
        generic_anchor_keys=generic_anchor_keys,
    )

    filtered, dropped = filter_mentions_by_foreign_anchors(
        mentions=mentions,
        clusters=clusters,
        anchors=anchors,
        fuzzy_threshold=args.fuzzy_threshold,
        own_family_fuzzy_threshold=args.own_family_fuzzy_threshold,
        min_fuzzy_chars=args.min_fuzzy_chars,
    )

    filtered.to_csv(args.output, index=False)

    if args.dropped_output:
        dropped.to_csv(args.dropped_output, index=False)

    if args.anchors_output:
        pd.DataFrame([
            {
                "anchor_key": anchor.key,
                "display_name": anchor.display_name,
                "n_mentions": anchor.n_mentions,
                "source_cluster_uids": "|".join(anchor.source_cluster_uids),
            }
            for anchor in anchors
        ]).to_csv(args.anchors_output, index=False)

    before = len(mentions)
    after = len(filtered)
    dropped_n = before - after

    print("Done.")
    print(f"  Selected anchors : {len(anchors):,}")
    print(f"  Mentions before  : {before:,}")
    print(f"  Mentions after   : {after:,}")
    print(f"  Dropped          : {dropped_n:,} ({(dropped_n / before * 100) if before else 0:.1f}%)")

    if anchors:
        print("  Top anchors:")
        for anchor in anchors[:20]:
            print(f"    - {anchor.display_name!r} [{anchor.key}] n_mentions={anchor.n_mentions}")


if __name__ == "__main__":
    main()
