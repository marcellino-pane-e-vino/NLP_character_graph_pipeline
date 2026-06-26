"""JSONL artifact helpers for cluster typing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator
import json
import re


__all__ = [
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "completed_mention_ids_from_jsonl",
    "cluster_typing_output_dir",
    "default_cluster_typing_jsonl_path",
    "safe_filename_component",
]


def safe_filename_component(value: Any, *, default: str = "unknown") -> str:
    text = str(value if value is not None else default).strip() or default
    text = " ".join(text.split())
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or default


def cluster_typing_output_dir(
    *,
    output_root: str | Path,
    n_mentions_per_cluster: int | None,
) -> Path:
    n_part = "all" if n_mentions_per_cluster is None else str(n_mentions_per_cluster)
    return Path(output_root) / "cluster_typing" / n_part


def default_cluster_typing_jsonl_path(
    output_dir: str | Path,
    *,
    cluster_id: int,
    subject: str,
    n_mentions: int | None,
) -> Path:
    subject_part = safe_filename_component(subject, default="unknown_subject")
    n_part = "all" if n_mentions is None else str(n_mentions)
    return (
        Path(output_dir)
        / f"cluster_typing_evidence_cluster_{int(cluster_id)}_{subject_part}_{n_part}.jsonl"
    )


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}, line {line_index}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object at {path}, line {line_index}")
            yield payload


def _record_is_complete(record: dict[str, Any]) -> bool:
    if "mention_id" not in record:
        return False
    selected_path = record.get("selected_path")
    weight = record.get("mention_weight_raw")
    return isinstance(selected_path, list) and isinstance(weight, dict)


def completed_mention_ids_from_jsonl(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[int] = set()
    for record in read_jsonl(path):
        if not _record_is_complete(record):
            continue
        completed.add(int(record["mention_id"]))
    return completed
