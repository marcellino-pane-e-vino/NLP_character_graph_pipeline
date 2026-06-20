from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from coreference.local_clusters import (
    CHUNKS_COLUMNS,
    CLUSTERS_COLUMNS,
    MENTIONS_COLUMNS,
    LocalClusterChunkRows,
    LocalClustersTables,
)

__all__ = [
    "REQUIRED_ARTIFACT_FILES",
    "LocalClustersArtifactWriter",
    "LocalClusterMarathonStore",
    "save_local_clusters_artifact",
    "load_local_clusters_artifact",
    "validate_artifact_files",
    "local_clusters_artifact_exists",
    "stable_json_hash",
    "file_sha256",
]

REQUIRED_ARTIFACT_FILES = ("metadata.json", "chunks.csv", "clusters.csv", "mentions.csv")


class LocalClustersArtifactWriter:
    """Streaming builder for the final local Maverick artifact.

    This compatibility writer streams rows into one temporary build directory and
    finalizes one zip. It is still useful for small/debug runs, but it is not a
    resumable cache. For long local runs where a kernel crash is expected, prefer
    LocalClusterMarathonStore.
    """

    def __init__(
        self,
        *,
        output_zip_path: str | Path,
        metadata: dict[str, Any],
        chunks_rows: list[dict[str, Any]],
        keep_failed_build_dir: bool = True,
        build_dir: str | Path | None = None,
    ) -> None:
        self.output_zip_path = Path(output_zip_path)
        self.output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.keep_failed_build_dir = bool(keep_failed_build_dir)
        self._finalized = False
        self._failed = False
        self._open_files: list[Any] = []
        self._writers: dict[str, csv.DictWriter] = {}
        self.row_counts = {
            "chunks.csv": len(chunks_rows),
            "clusters.csv": 0,
            "mentions.csv": 0,
        }
        self.n_overlap_mentions = 0

        if build_dir is None:
            self.build_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{self.output_zip_path.stem}_build_",
                    dir=str(self.output_zip_path.parent),
                )
            )
        else:
            self.build_dir = Path(build_dir)
            self.build_dir.mkdir(parents=True, exist_ok=False)

        _write_csv(self.build_dir / "chunks.csv", chunks_rows, CHUNKS_COLUMNS)
        self._open_stream("clusters.csv", CLUSTERS_COLUMNS)
        self._open_stream("mentions.csv", MENTIONS_COLUMNS)
        self.write_progress({"event": "build_started", "build_dir": str(self.build_dir)})

    def append_clusters(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows("clusters.csv", rows)

    def append_mentions(self, rows: list[dict[str, Any]]) -> None:
        self._append_rows("mentions.csv", rows)
        self.n_overlap_mentions += sum(1 for row in rows if row.get("zone") != "core")

    def append_chunk_rows(self, *, clusters_rows: list[dict[str, Any]], mentions_rows: list[dict[str, Any]]) -> None:
        self.append_clusters(clusters_rows)
        self.append_mentions(mentions_rows)

    def write_progress(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            **event,
        }
        with (self.build_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def to_tables(self) -> LocalClustersTables:
        """Load current build CSVs into memory for final validation/debug."""
        self._flush()
        return LocalClustersTables(
            metadata=self._metadata_with_counts(),
            chunks_rows=_read_csv_path(self.build_dir / "chunks.csv"),
            clusters_rows=_read_csv_path(self.build_dir / "clusters.csv"),
            mentions_rows=_read_csv_path(self.build_dir / "mentions.csv"),
        )

    def finalize(self, *, metadata: dict[str, Any] | None = None) -> Path:
        """Write metadata and create the final zip. Only call after validation."""
        if self._failed:
            raise RuntimeError("Cannot finalize an artifact writer that has recorded a failure.")
        self._close_streams()
        final_metadata = dict(metadata or self._metadata_with_counts())
        (self.build_dir / "metadata.json").write_text(
            json.dumps(final_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with zipfile.ZipFile(self.output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in REQUIRED_ARTIFACT_FILES:
                zf.write(self.build_dir / filename, arcname=filename)

        self._finalized = True
        shutil.rmtree(self.build_dir, ignore_errors=True)
        return self.output_zip_path

    def record_failure(self, exc: BaseException, *, chunk_id: str | None = None) -> None:
        """Record a clean failure report and keep the temp directory for diagnostics."""
        if self._failed:
            return
        self._failed = True
        self._close_streams()
        failure_report = {
            "timestamp": utc_now_iso(),
            "chunk_id": chunk_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "output_zip_path": str(self.output_zip_path),
            "build_dir": str(self.build_dir),
            "row_counts": dict(self.row_counts),
        }
        (self.build_dir / "failure_report.json").write_text(
            json.dumps(failure_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.build_dir / "failure_traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
        self.write_progress({"event": "build_failed", **failure_report})
        if not self.keep_failed_build_dir:
            shutil.rmtree(self.build_dir, ignore_errors=True)

    def close(self) -> None:
        self._close_streams()

    def _metadata_with_counts(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("files", {})
        metadata["files"] = {
            "chunks.csv": {"rows": self.row_counts["chunks.csv"]},
            "clusters.csv": {"rows": self.row_counts["clusters.csv"]},
            "mentions.csv": {"rows": self.row_counts["mentions.csv"]},
        }
        metadata.setdefault("chunking", {})
        metadata["chunking"]["n_chunks"] = self.row_counts["chunks.csv"]
        metadata["stats"] = {
            "n_chunks": self.row_counts["chunks.csv"],
            "n_local_clusters": self.row_counts["clusters.csv"],
            "n_local_mentions": self.row_counts["mentions.csv"],
            "n_overlap_mentions": self.n_overlap_mentions,
        }
        return metadata

    def _open_stream(self, filename: str, columns: list[str]) -> None:
        f = (self.build_dir / filename).open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        self._open_files.append(f)
        self._writers[filename] = writer

    def _append_rows(self, filename: str, rows: list[dict[str, Any]]) -> None:
        writer = self._writers[filename]
        columns = writer.fieldnames or []
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        self.row_counts[filename] += len(rows)
        self._flush()

    def _flush(self) -> None:
        for f in self._open_files:
            try:
                f.flush()
            except Exception:
                pass

    def _close_streams(self) -> None:
        for f in list(self._open_files):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self._open_files.clear()
        self._writers.clear()


class LocalClusterMarathonStore:
    """Local crash-resumable store for long Maverick local-cluster runs.

    The store persists one independent shard per chunk. Completed shards can be
    skipped across notebook/kernel restarts. Corrupt shards are treated as local
    failures and recomputed individually. Final artifact creation is a separate
    streaming aggregation step that reads shard CSVs into a temporary final build
    directory, validates global IDs, and atomically overwrites the final zip.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        output_zip_path: str | Path,
        metadata: dict[str, Any],
        chunk_plan: Any,
        maverick_config: dict[str, Any],
        recompute_corrupt_chunks: bool = True,
        keep_corrupt_shard_backups: bool = True,
        force_recompute_chunks: set[str] | None = None,
        force_recompute_all: bool = False,
        enable_checksums: bool = True,
        final_streaming_validate_global_ids: bool = True,
        overwrite_existing_final_zip: bool = True,
        delete_cache_after_finalize: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.output_zip_path = Path(output_zip_path)
        self.output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.chunk_plan = chunk_plan
        self.maverick_config = dict(maverick_config)
        self.recompute_corrupt_chunks = bool(recompute_corrupt_chunks)
        self.keep_corrupt_shard_backups = bool(keep_corrupt_shard_backups)
        self.force_recompute_chunks = set(force_recompute_chunks or set())
        self.force_recompute_all = bool(force_recompute_all)
        self.enable_checksums = bool(enable_checksums)
        self.final_streaming_validate_global_ids = bool(final_streaming_validate_global_ids)
        self.overwrite_existing_final_zip = bool(overwrite_existing_final_zip)
        self.delete_cache_after_finalize = bool(delete_cache_after_finalize)

        self.chunks_dir = self.cache_dir / "chunks"
        self.failures_dir = self.cache_dir / "failures"
        self.corrupt_dir = self.cache_dir / "corrupt"
        self.progress_path = self.cache_dir / "progress.jsonl"
        self.manifest_path = self.cache_dir / "run_manifest.json"
        self.chunk_plan_path = self.cache_dir / "chunk_plan.json"

        self.chunk_plan_payload = serialize_chunk_plan(self.chunk_plan)
        self.chunk_plan_fingerprint = stable_json_hash(self.chunk_plan_payload)
        self.maverick_config_fingerprint = stable_json_hash(_json_safe(self.maverick_config))
        self.run_manifest = self._build_run_manifest()
        self.run_fingerprint = self.run_manifest["fingerprints"]["run_fingerprint"]

    def validate_or_initialize_run(self) -> None:
        """Create a new cache or validate that the existing cache matches this run."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing_fingerprint = existing.get("fingerprints", {}).get("run_fingerprint")
            if existing_fingerprint != self.run_fingerprint:
                raise RuntimeError(
                    "Existing marathon cache is incompatible with the current run.\n"
                    f"cache_dir={self.cache_dir}\n"
                    f"existing_run_fingerprint={existing_fingerprint}\n"
                    f"current_run_fingerprint={self.run_fingerprint}\n"
                    "Use a different cache directory or intentionally delete the existing cache."
                )
            if not self.chunk_plan_path.exists():
                raise RuntimeError(f"Cache manifest exists but chunk_plan.json is missing: {self.chunk_plan_path}")
            existing_plan = json.loads(self.chunk_plan_path.read_text(encoding="utf-8"))
            existing_plan_fingerprint = stable_json_hash(existing_plan)
            if existing_plan_fingerprint != self.chunk_plan_fingerprint:
                raise RuntimeError(
                    "Existing chunk_plan.json does not match the current chunk plan despite matching manifest. "
                    "This cache is inconsistent and should be inspected manually."
                )
            self.write_progress({"event": "cache_resumed", "cache_dir": str(self.cache_dir)})
            return

        if any(self.chunks_dir.iterdir()):
            raise RuntimeError(
                f"Cache directory contains chunk shards but no run_manifest.json: {self.cache_dir}. "
                "Refusing to infer compatibility from orphaned shards."
            )

        write_json_atomic(self.manifest_path, self.run_manifest)
        write_json_atomic(self.chunk_plan_path, self.chunk_plan_payload)
        self.write_progress({"event": "cache_initialized", "cache_dir": str(self.cache_dir)})

    def should_process_chunk(self, spec: Any) -> bool:
        """Return True when a chunk must be processed/reprocessed."""
        self._delete_stale_tmp_shard(spec)

        if self.force_recompute_all or spec.chunk_id in self.force_recompute_chunks:
            self.write_progress({"event": "chunk_forced_recompute", "chunk_id": spec.chunk_id})
            return True

        valid, reason = self._validate_shard(spec)
        if valid:
            self.write_progress({"event": "chunk_skipped", "chunk_id": spec.chunk_id})
            return False

        final_dir = self._chunk_dir(spec)
        if not final_dir.exists():
            return True

        self.write_progress({"event": "chunk_corrupt", "chunk_id": spec.chunk_id, "reason": reason})
        if not self.recompute_corrupt_chunks:
            raise RuntimeError(f"Chunk shard is invalid and recompute_corrupt_chunks=False: {spec.chunk_id}: {reason}")

        self._move_corrupt_shard_aside(spec, reason=reason)
        return True

    def mark_chunk_started(self, spec: Any) -> None:
        self.write_progress({
            "event": "chunk_started",
            "chunk_id": spec.chunk_id,
            "chunk_index": spec.chunk_index,
            "expanded_tokens": spec.n_tokens,
        })

    def commit_chunk(
        self,
        *,
        spec: Any,
        rows: LocalClusterChunkRows,
        validation_report: dict[str, Any],
    ) -> None:
        """Atomically commit one completed chunk shard."""
        if rows.chunk_id != spec.chunk_id:
            raise ValueError(f"Rows chunk_id mismatch: rows={rows.chunk_id!r}, spec={spec.chunk_id!r}")

        tmp_dir = self._tmp_chunk_dir(spec)
        final_dir = self._chunk_dir(spec)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            clusters_path = tmp_dir / "clusters.csv"
            mentions_path = tmp_dir / "mentions.csv"
            _write_csv(clusters_path, rows.clusters_rows, CLUSTERS_COLUMNS)
            _write_csv(mentions_path, rows.mentions_rows, MENTIONS_COLUMNS)
            write_json_atomic(tmp_dir / "validation_report.json", validation_report)

            status = {
                "status": "complete",
                "run_fingerprint": self.run_fingerprint,
                "chunk_id": spec.chunk_id,
                "chunk_index": int(spec.chunk_index),
                "n_clusters": len(rows.clusters_rows),
                "n_mentions": len(rows.mentions_rows),
                "clusters_sha256": file_sha256(clusters_path),
                "mentions_sha256": file_sha256(mentions_path),
                "completed_at": utc_now_iso(),
            }
            write_json_atomic(tmp_dir / "status.json", status)

            valid, reason = self._validate_shard(spec, shard_dir=tmp_dir)
            if not valid:
                raise RuntimeError(f"Newly written shard failed validation before commit: {spec.chunk_id}: {reason}")

            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            tmp_dir.rename(final_dir)

            self.write_progress({
                "event": "chunk_committed",
                "chunk_id": spec.chunk_id,
                "chunk_index": spec.chunk_index,
                "n_clusters": len(rows.clusters_rows),
                "n_mentions": len(rows.mentions_rows),
            })
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def record_chunk_failure(self, *, spec: Any, exc: BaseException) -> None:
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        failure_report = {
            "timestamp": utc_now_iso(),
            "chunk_id": spec.chunk_id,
            "chunk_index": int(spec.chunk_index),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "cache_dir": str(self.cache_dir),
            "output_zip_path": str(self.output_zip_path),
        }
        safe_chunk_id = _safe_path_name(spec.chunk_id)
        write_json_atomic(self.failures_dir / f"{safe_chunk_id}_failure.json", failure_report)
        (self.failures_dir / f"{safe_chunk_id}_traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
        self.write_progress({"event": "chunk_failed", **failure_report})

    def validate_all_shards_complete(self) -> None:
        errors: list[str] = []
        for spec in self.chunk_plan.specs:
            valid, reason = self._validate_shard(spec)
            if not valid:
                errors.append(f"{spec.chunk_id}: {reason}")
        if errors:
            preview = "\n".join(f"- {error}" for error in errors[:30])
            if len(errors) > 30:
                preview += f"\n... and {len(errors) - 30} more invalid/missing shards."
            raise RuntimeError("Cannot finalize because not all chunk shards are complete and valid:\n" + preview)

    def finalize_streaming(self) -> Path:
        """Stream all valid shards into the final artifact zip."""
        self.validate_all_shards_complete()
        self.write_progress({"event": "finalization_started", "output_zip_path": str(self.output_zip_path)})

        build_dir = self.output_zip_path.parent / f"{self.output_zip_path.stem}_final_build.__tmp__"
        tmp_zip_path = self.output_zip_path.with_suffix(self.output_zip_path.suffix + ".__tmp__")

        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        if tmp_zip_path.exists():
            tmp_zip_path.unlink()
        build_dir.mkdir(parents=True, exist_ok=False)

        try:
            counts = self._stream_final_csvs(build_dir)
            final_metadata = self._metadata_with_counts(counts)
            write_json_atomic(build_dir / "metadata.json", final_metadata)

            with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for filename in REQUIRED_ARTIFACT_FILES:
                    zf.write(build_dir / filename, arcname=filename)

            validate_artifact_files(tmp_zip_path)
            if self.output_zip_path.exists() and not self.overwrite_existing_final_zip:
                raise FileExistsError(f"Final artifact already exists: {self.output_zip_path}")
            os.replace(tmp_zip_path, self.output_zip_path)
            self.write_progress({"event": "finalization_completed", "output_zip_path": str(self.output_zip_path), "counts": counts})

            if self.delete_cache_after_finalize:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            return self.output_zip_path
        except Exception:
            if tmp_zip_path.exists():
                tmp_zip_path.unlink()
            raise
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    def write_progress(self, event: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": utc_now_iso(),
            **event,
        }
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _build_run_manifest(self) -> dict[str, Any]:
        stable_payload = {
            "artifact_type": "local_cluster_marathon_cache",
            "schema_version": self.metadata.get("schema_version"),
            "document": self.metadata.get("document", {}),
            "offsets": self.metadata.get("offsets", {}),
            "chunking": self.metadata.get("chunking", {}),
            "maverick": _json_safe(self.maverick_config),
            "chunk_plan_fingerprint": self.chunk_plan_fingerprint,
            "maverick_config_fingerprint": self.maverick_config_fingerprint,
        }
        run_fingerprint = stable_json_hash(stable_payload)
        return {
            "artifact_type": "local_cluster_marathon_cache",
            "schema_version": self.metadata.get("schema_version"),
            "created_at": utc_now_iso(),
            "document": self.metadata.get("document", {}),
            "offsets": self.metadata.get("offsets", {}),
            "chunking": self.metadata.get("chunking", {}),
            "maverick": _json_safe(self.maverick_config),
            "fingerprints": {
                "run_fingerprint": run_fingerprint,
                "chunk_plan_fingerprint": self.chunk_plan_fingerprint,
                "maverick_config_fingerprint": self.maverick_config_fingerprint,
            },
        }

    def _validate_shard(self, spec: Any, *, shard_dir: Path | None = None) -> tuple[bool, str]:
        shard_dir = shard_dir or self._chunk_dir(spec)
        if not shard_dir.exists():
            return False, "missing shard directory"
        if not shard_dir.is_dir():
            return False, "shard path exists but is not a directory"

        status_path = shard_dir / "status.json"
        clusters_path = shard_dir / "clusters.csv"
        mentions_path = shard_dir / "mentions.csv"

        if not status_path.exists():
            return False, "missing status.json"
        if not clusters_path.exists():
            return False, "missing clusters.csv"
        if not mentions_path.exists():
            return False, "missing mentions.csv"

        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"unreadable status.json: {exc}"

        checks = {
            "status": status.get("status") == "complete",
            "run_fingerprint": status.get("run_fingerprint") == self.run_fingerprint,
            "chunk_id": status.get("chunk_id") == spec.chunk_id,
            "chunk_index": int(status.get("chunk_index", -1)) == int(spec.chunk_index),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            return False, f"status field mismatch: {failed}"

        try:
            cluster_count = validate_csv_file_for_chunk(
                clusters_path,
                expected_columns=CLUSTERS_COLUMNS,
                expected_chunk_id=spec.chunk_id,
            )
            mention_count = validate_csv_file_for_chunk(
                mentions_path,
                expected_columns=MENTIONS_COLUMNS,
                expected_chunk_id=spec.chunk_id,
            )
        except Exception as exc:
            return False, str(exc)

        if int(status.get("n_clusters", -1)) != cluster_count:
            return False, f"cluster row count mismatch: status={status.get('n_clusters')} actual={cluster_count}"
        if int(status.get("n_mentions", -1)) != mention_count:
            return False, f"mention row count mismatch: status={status.get('n_mentions')} actual={mention_count}"

        if self.enable_checksums:
            actual_clusters_sha = file_sha256(clusters_path)
            actual_mentions_sha = file_sha256(mentions_path)
            if status.get("clusters_sha256") != actual_clusters_sha:
                return False, "clusters.csv checksum mismatch"
            if status.get("mentions_sha256") != actual_mentions_sha:
                return False, "mentions.csv checksum mismatch"

        return True, "ok"

    def _stream_final_csvs(self, build_dir: Path) -> dict[str, int]:
        chunk_rows = chunk_plan_payload_to_rows(self.chunk_plan_payload)
        _write_csv(build_dir / "chunks.csv", chunk_rows, CHUNKS_COLUMNS)

        seen_cluster_uids: set[str] = set()
        seen_mention_uids: set[str] = set()
        referenced_cluster_uids: set[str] = set()
        n_clusters = 0
        n_mentions = 0
        n_overlap_mentions = 0

        with (build_dir / "clusters.csv").open("w", encoding="utf-8", newline="") as final_clusters:
            writer = csv.DictWriter(final_clusters, fieldnames=CLUSTERS_COLUMNS, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for spec in self.chunk_plan.specs:
                shard_path = self._chunk_dir(spec) / "clusters.csv"
                for row in iter_csv_rows_checked(shard_path, CLUSTERS_COLUMNS):
                    if row.get("chunk_id") != spec.chunk_id:
                        raise ValueError(f"Cluster row in {shard_path} has wrong chunk_id={row.get('chunk_id')!r}")
                    uid = str(row.get("local_cluster_uid", ""))
                    if self.final_streaming_validate_global_ids:
                        if uid in seen_cluster_uids:
                            raise ValueError(f"Duplicate local_cluster_uid during final aggregation: {uid}")
                        seen_cluster_uids.add(uid)
                    writer.writerow({column: row.get(column, "") for column in CLUSTERS_COLUMNS})
                    n_clusters += 1

        with (build_dir / "mentions.csv").open("w", encoding="utf-8", newline="") as final_mentions:
            writer = csv.DictWriter(final_mentions, fieldnames=MENTIONS_COLUMNS, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for spec in self.chunk_plan.specs:
                shard_path = self._chunk_dir(spec) / "mentions.csv"
                for row in iter_csv_rows_checked(shard_path, MENTIONS_COLUMNS):
                    if row.get("chunk_id") != spec.chunk_id:
                        raise ValueError(f"Mention row in {shard_path} has wrong chunk_id={row.get('chunk_id')!r}")
                    mention_uid = str(row.get("local_mention_uid", ""))
                    cluster_uid = str(row.get("local_cluster_uid", ""))
                    if self.final_streaming_validate_global_ids:
                        if mention_uid in seen_mention_uids:
                            raise ValueError(f"Duplicate local_mention_uid during final aggregation: {mention_uid}")
                        seen_mention_uids.add(mention_uid)
                        referenced_cluster_uids.add(cluster_uid)
                    if row.get("zone") != "core":
                        n_overlap_mentions += 1
                    writer.writerow({column: row.get(column, "") for column in MENTIONS_COLUMNS})
                    n_mentions += 1

        if self.final_streaming_validate_global_ids:
            missing_refs = sorted(referenced_cluster_uids - seen_cluster_uids)
            if missing_refs:
                preview = missing_refs[:20]
                raise ValueError(
                    "Mention rows reference missing local_cluster_uid values. "
                    f"Preview={preview}, total_missing={len(missing_refs)}"
                )

        return {
            "chunks.csv": len(chunk_rows),
            "clusters.csv": n_clusters,
            "mentions.csv": n_mentions,
            "n_overlap_mentions": n_overlap_mentions,
        }

    def _metadata_with_counts(self, counts: dict[str, int]) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata["created_at"] = utc_now_iso()
        metadata["files"] = {
            "chunks.csv": {"rows": counts["chunks.csv"]},
            "clusters.csv": {"rows": counts["clusters.csv"]},
            "mentions.csv": {"rows": counts["mentions.csv"]},
        }
        metadata.setdefault("chunking", {})["n_chunks"] = counts["chunks.csv"]
        metadata["stats"] = {
            "n_chunks": counts["chunks.csv"],
            "n_local_clusters": counts["clusters.csv"],
            "n_local_mentions": counts["mentions.csv"],
            "n_overlap_mentions": counts["n_overlap_mentions"],
        }
        metadata["marathon_cache"] = {
            "run_fingerprint": self.run_fingerprint,
            "chunk_plan_fingerprint": self.chunk_plan_fingerprint,
            "maverick_config_fingerprint": self.maverick_config_fingerprint,
            "checksums_enabled": self.enable_checksums,
            "final_streaming_validate_global_ids": self.final_streaming_validate_global_ids,
            "private_cache_included_in_artifact": False,
        }
        return metadata

    def _delete_stale_tmp_shard(self, spec: Any) -> None:
        tmp_dir = self._tmp_chunk_dir(spec)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self.write_progress({"event": "stale_tmp_shard_deleted", "chunk_id": spec.chunk_id})

    def _move_corrupt_shard_aside(self, spec: Any, *, reason: str) -> None:
        final_dir = self._chunk_dir(spec)
        if not final_dir.exists():
            return
        if not self.keep_corrupt_shard_backups:
            shutil.rmtree(final_dir, ignore_errors=True)
            return
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)
        target = self.corrupt_dir / f"{_safe_path_name(spec.chunk_id)}__corrupt_{timestamp_for_path()}"
        counter = 1
        while target.exists():
            target = self.corrupt_dir / f"{_safe_path_name(spec.chunk_id)}__corrupt_{timestamp_for_path()}_{counter}"
            counter += 1
        final_dir.rename(target)
        write_json_atomic(target / "corruption_reason.json", {"reason": reason, "moved_at": utc_now_iso()})

    def _chunk_dir(self, spec: Any) -> Path:
        return self.chunks_dir / str(spec.chunk_id)

    def _tmp_chunk_dir(self, spec: Any) -> Path:
        return self.chunks_dir / f"{spec.chunk_id}.__tmp__"


def local_clusters_artifact_exists(path: str | Path) -> bool:
    """Return True if the local clusters artifact zip exists at the expected path."""
    return Path(path).is_file()


def save_local_clusters_artifact(tables: LocalClustersTables, output_zip_path: str | Path) -> Path:
    """Save the final local Maverick artifact as one zip file.

    This compatibility function writes from in-memory tables. For long local
    marathon runs, prefer LocalClusterMarathonStore.finalize_streaming(...).
    """
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "metadata.json").write_text(
            json.dumps(tables.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(root / "chunks.csv", tables.chunks_rows, CHUNKS_COLUMNS)
        _write_csv(root / "clusters.csv", tables.clusters_rows, CLUSTERS_COLUMNS)
        _write_csv(root / "mentions.csv", tables.mentions_rows, MENTIONS_COLUMNS)

        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in REQUIRED_ARTIFACT_FILES:
                zf.write(root / filename, arcname=filename)

    return output_zip_path


def load_local_clusters_artifact(input_zip_path: str | Path) -> LocalClustersTables:
    """Load a maverick_local_clusters.zip artifact."""
    input_zip_path = Path(input_zip_path)
    validate_artifact_files(input_zip_path)

    with zipfile.ZipFile(input_zip_path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        chunks_rows = _read_csv_from_zip(zf, "chunks.csv")
        clusters_rows = _read_csv_from_zip(zf, "clusters.csv")
        mentions_rows = _read_csv_from_zip(zf, "mentions.csv")

    return LocalClustersTables(
        metadata=metadata,
        chunks_rows=chunks_rows,
        clusters_rows=clusters_rows,
        mentions_rows=mentions_rows,
    )


def validate_artifact_files(input_zip_path: str | Path) -> None:
    input_zip_path = Path(input_zip_path)
    if not input_zip_path.exists():
        raise FileNotFoundError(f"Artifact not found: {input_zip_path}")
    with zipfile.ZipFile(input_zip_path, "r") as zf:
        names = set(zf.namelist())
    missing = [name for name in REQUIRED_ARTIFACT_FILES if name not in names]
    if missing:
        raise ValueError(f"Artifact {input_zip_path} is missing required files: {missing}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def stable_json_hash(payload: Any) -> str:
    data = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".__tmp__")
    tmp_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def serialize_chunk_plan(chunk_plan: Any) -> dict[str, Any]:
    return {
        "chunk_size": int(getattr(chunk_plan, "chunk_size")),
        "overlap_sentences": int(getattr(chunk_plan, "overlap_sentences")),
        "max_expanded_chunk_tokens": int(getattr(chunk_plan, "max_expanded_chunk_tokens")),
        "specs": [chunk_spec_to_dict(spec) for spec in getattr(chunk_plan, "specs")],
    }


def chunk_spec_to_dict(spec: Any) -> dict[str, int | str]:
    return {
        "chunk_id": str(spec.chunk_id),
        "chunk_index": int(spec.chunk_index),
        "global_start": int(spec.global_start),
        "global_end": int(spec.global_end),
        "core_start": int(spec.core_start),
        "core_end": int(spec.core_end),
        "left_overlap_start": int(spec.left_overlap_start),
        "left_overlap_end": int(spec.left_overlap_end),
        "right_overlap_start": int(spec.right_overlap_start),
        "right_overlap_end": int(spec.right_overlap_end),
        "n_tokens": int(spec.n_tokens),
        "sentence_start": int(spec.sentence_start),
        "sentence_end": int(spec.sentence_end),
    }


def chunk_plan_payload_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in payload.get("specs", []):
        rows.append({column: spec.get(column, "") for column in CHUNKS_COLUMNS})
    return rows


def validate_csv_file_for_chunk(path: str | Path, *, expected_columns: list[str], expected_chunk_id: str) -> int:
    count = 0
    for row in iter_csv_rows_checked(path, expected_columns):
        if row.get("chunk_id") != expected_chunk_id:
            raise ValueError(f"{path} contains row with chunk_id={row.get('chunk_id')!r}, expected {expected_chunk_id!r}")
        count += 1
    return count


def iter_csv_rows_checked(path: str | Path, expected_columns: list[str]) -> Iterable[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"CSV header mismatch in {path}: expected={expected_columns}, actual={reader.fieldnames}")
        for row in reader:
            yield dict(row)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with zf.open(name, "r") as raw:
        text = raw.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _read_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_path_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
