# coreference_sub_orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.serialization as torch_serialization

from coreference.annotator import annotate_doc_with_global_coref
from artifact_io import LocalClusterMarathonStore, load_local_clusters_artifact
from coreference.coref_schema import register_spacy_coref_extension, require_coref_layer
from coreference.coreference_cluster_extractor import create_coreference_cluster_extractor
from coreference.global_coref_merger_aggregative import merge_local_coreference_clusters
from coreference.local_clusters import (
    create_local_clusters_metadata,
    extracted_chunk_to_rows,
    validate_chunk_rows,
)
from coreference.mention_filter import (
    build_anchor_bank,
    filter_mentions_by_foreign_anchors,
    load_generic_anchor_keys,
)
from runtime_config import (
    CHUNK_SIZE,
    CHUNK_SIZE_SEMANTICS,
    DEVICE,
    MAX_EXPANDED_CHUNK_TOKENS,
    OVERLAP_SENTENCES,
    RUNTIME_PROFILE,
    cleanup_after_chunk,
    memory_snapshot,
)


# ---------------------------------------------------------------------------
# Maverick extraction config
# ---------------------------------------------------------------------------

HF_NAME_OR_PATH = "sapienzanlp/maverick-mes-litbank"
SINGLETONS = True
CLEAN_MENTIONS = True
MAX_MENTION_TOKENS = 20
DROP_CROSS_SENTENCE_MENTIONS = True


# ---------------------------------------------------------------------------
# Local marathon-cache behavior
# ---------------------------------------------------------------------------

LOCAL_CLUSTER_EXECUTION_MODE = "marathon_cache"
FAIL_ON_INCOMPATIBLE_MANIFEST = True

RECOMPUTE_CORRUPT_CHUNKS = True
KEEP_CORRUPT_SHARD_BACKUPS = True
FORCE_RECOMPUTE_CHUNKS: set[str] = set()
FORCE_RECOMPUTE_ALL_LOCAL_CLUSTERS = False
ENABLE_SHARD_CHECKSUMS = True
FINAL_STREAMING_VALIDATE_GLOBAL_IDS = True
DELETE_CACHE_AFTER_FINALIZE = False
OVERWRITE_EXISTING_FINAL_ZIP = True

# Per-chunk validation catches broken rows before committing a shard.
PER_CHUNK_VALIDATION_MODE = "basic"  # "basic" or "debug"


# ---------------------------------------------------------------------------
# Destructive filter config
# ---------------------------------------------------------------------------

ANCHOR_PERCENTILE = 0.95
MIN_ANCHOR_MENTIONS = 2
FUZZY_THRESHOLD = 0.86
OWN_FAMILY_FUZZY_THRESHOLD = 0.90
MIN_FUZZY_CHARS = 5

# Optional custom blacklist file.
# Use None to rely only on the built-in conservative generic blacklist.
GENERIC_ANCHOR_BLACKLIST_PATH = None


def patch_torch_load_weights_only_false() -> None:
    """
    Idempotently patch torch.load so PyTorch/Lightning checkpoint loading uses
    weights_only=False, without creating recursive wrappers when the function is
    executed multiple times.
    """

    # Recover the real PyTorch implementation if torch.load was already patched.
    # In normal PyTorch, torch.load is exported from torch.serialization.load.
    torch.load = torch_serialization.load

    if not hasattr(torch, "_maverick_original_torch_load"):
        torch._maverick_original_torch_load = torch_serialization.load

    def _torch_load_force_weights_only_false(*args, **kwargs):
        kwargs["weights_only"] = False
        return torch._maverick_original_torch_load(*args, **kwargs)

    _torch_load_force_weights_only_false._maverick_weights_patch = True
    torch.load = _torch_load_force_weights_only_false


def rows_to_dataframe(rows) -> pd.DataFrame:
    """
    Accept either a DataFrame or a list[dict]-style row collection.
    """
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(rows)


def dataframe_to_original_row_shape(df: pd.DataFrame, original_rows):
    """
    Return DataFrame if the pipeline originally used DataFrames.
    Return list[dict] if the pipeline originally used row dictionaries.
    """
    if isinstance(original_rows, pd.DataFrame):
        return df
    return df.to_dict(orient="records")


def _validate_coreference_inputs(*, doc, chunker, chunk_plan) -> None:
    if doc is None:
        raise ValueError("doc must not be None.")

    if not hasattr(chunker, "materialize"):
        raise TypeError("chunker must expose a materialize(doc, spec) method.")

    if not hasattr(chunk_plan, "specs"):
        raise TypeError("chunk_plan must expose a .specs attribute.")

    if len(chunk_plan) != len(chunk_plan.specs):
        raise ValueError("chunk_plan length is inconsistent with chunk_plan.specs.")


def _derive_coreference_paths(output_dir: Path) -> dict[str, Path]:
    """Derive internal coreference artifact paths from the public output_dir.

    The local Maverick artifact/cache paths intentionally preserve the original
    notebook layout:
        OUTPUT_ROOT / "maverick_local_clusters.zip"
        OUTPUT_ROOT / ".maverick_local_cache"

    Since output_dir is expected to be OUTPUT_ROOT / "global_coref", those paths
    are derived from output_dir.parent while keeping the public API minimal.
    """
    output_root = output_dir.parent

    return {
        "local_clusters_artifact_path": output_root / "maverick_local_clusters.zip",
        "local_marathon_cache_dir": output_root / ".maverick_local_cache",
        "destructive_filter_output_dir": output_dir / "destructive_mention_filter",
        "global_clusters_csv": output_dir / "global_clusters.csv",
        "global_mentions_csv": output_dir / "global_mentions.csv",
    }


def _build_maverick_config() -> dict[str, Any]:
    return {
        "hf_name_or_path": HF_NAME_OR_PATH,
        "device": DEVICE,
        "singletons": SINGLETONS,
        "clean_mentions": CLEAN_MENTIONS,
        "max_mention_tokens": MAX_MENTION_TOKENS,
        "drop_cross_sentence_mentions": DROP_CROSS_SENTENCE_MENTIONS,
        "chunk_size": CHUNK_SIZE,
        "chunk_size_semantics": CHUNK_SIZE_SEMANTICS,
        "overlap_sentences": OVERLAP_SENTENCES,
        "max_expanded_chunk_tokens": MAX_EXPANDED_CHUNK_TOKENS,
        "runtime_profile": RUNTIME_PROFILE,
    }


def _load_local_tables(local_clusters_artifact_path: Path):
    if not local_clusters_artifact_path.exists():
        raise FileNotFoundError(
            f"Final artifact does not exist yet: {local_clusters_artifact_path}"
        )

    loaded_tables = load_local_clusters_artifact(local_clusters_artifact_path)
    print("Artifact metadata:")
    print(loaded_tables.metadata)

    return loaded_tables


def _apply_destructive_filter(
    *,
    loaded_tables,
    destructive_filter_output_dir: Path,
):
    destructive_filter_output_dir.mkdir(parents=True, exist_ok=True)

    filtered_mentions_path = destructive_filter_output_dir / "mentions.filtered.csv"
    dropped_mentions_path = destructive_filter_output_dir / "mentions.dropped.csv"
    selected_anchors_path = destructive_filter_output_dir / "selected_anchors.csv"

    clusters_df = rows_to_dataframe(loaded_tables.clusters_rows)
    mentions_df = rows_to_dataframe(loaded_tables.mentions_rows)

    print("[destructive-filter] Building foreign-anchor bank...")

    generic_anchor_keys = load_generic_anchor_keys(GENERIC_ANCHOR_BLACKLIST_PATH)

    anchors = build_anchor_bank(
        mentions=mentions_df,
        clusters=clusters_df,
        anchor_percentile=ANCHOR_PERCENTILE,
        min_anchor_mentions=MIN_ANCHOR_MENTIONS,
        generic_anchor_keys=generic_anchor_keys,
    )

    print(f"[destructive-filter] Selected anchors: {len(anchors):,}")

    filtered_mentions_df, dropped_mentions_df = filter_mentions_by_foreign_anchors(
        mentions=mentions_df,
        clusters=clusters_df,
        anchors=anchors,
        fuzzy_threshold=FUZZY_THRESHOLD,
        own_family_fuzzy_threshold=OWN_FAMILY_FUZZY_THRESHOLD,
        min_fuzzy_chars=MIN_FUZZY_CHARS,
    )

    selected_anchors_df = pd.DataFrame(
        [
            {
                "anchor_key": anchor.key,
                "display_name": anchor.display_name,
                "n_mentions": anchor.n_mentions,
                "source_cluster_uids": "|".join(anchor.source_cluster_uids),
            }
            for anchor in anchors
        ]
    )

    filtered_mentions_df.to_csv(filtered_mentions_path, index=False)
    dropped_mentions_df.to_csv(dropped_mentions_path, index=False)
    selected_anchors_df.to_csv(selected_anchors_path, index=False)

    before = len(mentions_df)
    after = len(filtered_mentions_df)
    dropped = before - after

    print("[destructive-filter] Done.")
    print(f"  Mentions before : {before:,}")
    print(f"  Mentions after  : {after:,}")
    print(f"  Dropped         : {dropped:,} ({(dropped / before * 100) if before else 0:.2f}%)")
    print(f"  Audit dropped   : {dropped_mentions_path}")
    print(f"  Audit anchors   : {selected_anchors_path}")

    return dataframe_to_original_row_shape(
        filtered_mentions_df,
        loaded_tables.mentions_rows,
    )


def run_coreference_resolution(
    *,
    doc,
    chunker,
    chunk_plan,
    document_id: str,
    output_dir: Path,
):
    """
    Run the complete coreference phase.

    Inputs:
        doc:
            Tokenized spaCy Doc to annotate.
        chunker:
            External chunker object. The sub-orchestrator uses it only to
            materialize chunks from the externally provided chunk_plan.
        chunk_plan:
            External chunk plan. The sub-orchestrator does not build it.
        document_id:
            Stable document identifier for artifact metadata.
        output_dir:
            Global coreference output directory.

    Output:
        The same logical Doc, annotated with the final global coreference layer.

    Side effects:
        - writes/updates the local Maverick marathon cache;
        - writes/updates the local cluster artifact;
        - writes destructive-filter audit CSVs;
        - writes global_clusters.csv and global_mentions.csv;
        - does not save the final annotated Doc pickle.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _derive_coreference_paths(output_dir)
    local_clusters_artifact_path = paths["local_clusters_artifact_path"]
    local_marathon_cache_dir = paths["local_marathon_cache_dir"]
    destructive_filter_output_dir = paths["destructive_filter_output_dir"]
    global_clusters_csv = paths["global_clusters_csv"]
    global_mentions_csv = paths["global_mentions_csv"]

    _validate_coreference_inputs(
        doc=doc,
        chunker=chunker,
        chunk_plan=chunk_plan,
    )

    patch_torch_load_weights_only_false()
    print("torch.load safely patched:", getattr(torch.load, "_maverick_weights_patch", False))

    coref_extractor = create_coreference_cluster_extractor(
        hf_name_or_path=HF_NAME_OR_PATH,
        device=DEVICE,
        singletons=SINGLETONS,
        verbose=True,
        clean_mentions=CLEAN_MENTIONS,
        max_mention_tokens=MAX_MENTION_TOKENS,
        drop_cross_sentence_mentions=DROP_CROSS_SENTENCE_MENTIONS,
        cpu_load_first=RUNTIME_PROFILE["cpu_load_first"],
        precision_policy=RUNTIME_PROFILE["precision_policy"],
        p100_fallback_to_float32=RUNTIME_PROFILE["p100_fallback_to_float32"],
    )

    maverick_config = _build_maverick_config()

    metadata = create_local_clusters_metadata(
        doc=doc,
        chunk_count=len(chunk_plan),
        chunk_size=CHUNK_SIZE,
        overlap_sentences=OVERLAP_SENTENCES,
        max_expanded_chunk_tokens=MAX_EXPANDED_CHUNK_TOKENS,
        maverick_config=maverick_config,
        document_id=document_id,
    )

    store = LocalClusterMarathonStore(
        cache_dir=local_marathon_cache_dir,
        output_zip_path=local_clusters_artifact_path,
        metadata=metadata,
        chunk_plan=chunk_plan,
        maverick_config=maverick_config,
        recompute_corrupt_chunks=RECOMPUTE_CORRUPT_CHUNKS,
        keep_corrupt_shard_backups=KEEP_CORRUPT_SHARD_BACKUPS,
        force_recompute_chunks=FORCE_RECOMPUTE_CHUNKS,
        force_recompute_all=FORCE_RECOMPUTE_ALL_LOCAL_CLUSTERS,
        enable_checksums=ENABLE_SHARD_CHECKSUMS,
        final_streaming_validate_global_ids=FINAL_STREAMING_VALIDATE_GLOBAL_IDS,
        overwrite_existing_final_zip=OVERWRITE_EXISTING_FINAL_ZIP,
        delete_cache_after_finalize=DELETE_CACHE_AFTER_FINALIZE,
    )

    store.validate_or_initialize_run()

    current_chunk_id = None
    try:
        for spec in chunk_plan.specs:
            current_chunk_id = spec.chunk_id

            if not store.should_process_chunk(spec):
                print(f"[chunk][skip] {spec.chunk_id}")
                continue

            print(
                f"[chunk] Processing {spec.chunk_id} "
                f"({spec.chunk_index + 1}/{len(chunk_plan)}) "
                f"expanded_tokens={spec.n_tokens}"
            )
            store.mark_chunk_started(spec)

            chunk = chunker.materialize(doc, spec)
            extracted = coref_extractor.extract(chunk)
            chunk_rows = extracted_chunk_to_rows(chunk=chunk, extracted=extracted)

            chunk_validation_report = validate_chunk_rows(
                doc=doc,
                chunk=chunk,
                rows=chunk_rows,
                mode=PER_CHUNK_VALIDATION_MODE,
            )

            store.commit_chunk(
                spec=spec,
                rows=chunk_rows,
                validation_report={
                    **chunk_validation_report,
                    "memory_after_validation": memory_snapshot(),
                },
            )

            print(
                f"[chunk] {spec.chunk_id}: "
                f"clusters={len(chunk_rows.clusters_rows)}, "
                f"mentions={len(chunk_rows.mentions_rows)}"
            )

            del chunk_rows
            del extracted
            del chunk

            coref_extractor.clear_runtime_state()
            cleanup_after_chunk(RUNTIME_PROFILE)

        store.validate_all_shards_complete()
        artifact_path = store.finalize_streaming()
        print(f"Saved final artifact to {artifact_path}")

    except Exception as exc:
        if current_chunk_id is not None:
            # Find the current spec without relying on partially materialized runtime objects.
            failing_spec = next(
                (s for s in chunk_plan.specs if s.chunk_id == current_chunk_id),
                None,
            )
            if failing_spec is not None:
                store.record_chunk_failure(spec=failing_spec, exc=exc)

        print("[failure] The final artifact was not created successfully.")
        print("[failure] Marathon cache directory:", local_marathon_cache_dir)
        raise

    finally:
        try:
            coref_extractor.clear_runtime_state()
        except Exception:
            pass
        cleanup_after_chunk(RUNTIME_PROFILE)

    loaded_tables = _load_local_tables(local_clusters_artifact_path)

    filtered_mentions_rows = _apply_destructive_filter(
        loaded_tables=loaded_tables,
        destructive_filter_output_dir=destructive_filter_output_dir,
    )

    global_coref_result = merge_local_coreference_clusters(
        clusters_rows=loaded_tables.clusters_rows,
        mentions_rows=filtered_mentions_rows,
        output_dir=output_dir,
        verbose=True,
    )

    print("Final global clusters:", len(global_coref_result.components))

    register_spacy_coref_extension()

    doc = annotate_doc_with_global_coref(
        doc=doc,
        clusters_csv=global_clusters_csv,
        mentions_csv=global_mentions_csv,
        verbose=True,
    )

    return doc


def print_non_singleton_coref_clusters(doc) -> None:
    coref_layer = require_coref_layer(doc)

    clusters_non_singleton = [
        (cluster_id, cluster)
        for cluster_id, cluster in coref_layer.clusters.items()
        if len(cluster.mention_ids) > 1
    ]

    clusters_sorted = sorted(
        clusters_non_singleton,
        key=lambda item: len(item[1].mention_ids),
        reverse=True,
    )

    print(f"Non-singleton clusters: {len(clusters_sorted)}")
    print()

    for cluster_id, cluster in clusters_sorted:
        print(
            f"cluster_id={cluster_id} | "
            f"canonical_name={cluster.canonical_name!r} | "
            f"n_mentions={len(cluster.mention_ids)}"
        )
        print()
