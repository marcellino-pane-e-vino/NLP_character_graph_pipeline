# runtime_config.py
from __future__ import annotations

import gc
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Runtime profile
# ---------------------------------------------------------------------------

# Local runtime profile. Set to None, "local_cpu", or "local_cuda".
FORCE_RUNTIME_PROFILE: str | None = None


def detect_runtime_profile(*, force_profile: str | None = None) -> dict[str, Any]:
    """Return one explicit local runtime profile.

    This is shared pipeline infrastructure:
    - the main notebook uses the default chunking policy;
    - the coreference sub-orchestrator uses the Maverick execution policy.
    """
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""

    if force_profile:
        kind = force_profile
    elif cuda_available:
        kind = "local_cuda"
    else:
        kind = "local_cpu"

    defaults = {
        "local_cpu": {"chunk_size": 3000, "overlap_sentences": 30},
        "local_cuda": {"chunk_size": 6000, "overlap_sentences": 60},
    }
    profile_defaults = defaults.get(kind, defaults["local_cpu"])

    return {
        "kind": kind,
        "env": "Local",
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "device": "cuda:0" if cuda_available else "cpu",
        "cpu_load_first": bool(cuda_available),
        "precision_policy": "auto",
        "p100_fallback_to_float32": False,
        "default_chunk_size": profile_defaults["chunk_size"],
        "default_overlap_sentences": profile_defaults["overlap_sentences"],
    }


RUNTIME_PROFILE = detect_runtime_profile(force_profile=FORCE_RUNTIME_PROFILE)
DEVICE = RUNTIME_PROFILE["device"]

# CHUNK_SIZE means expanded chunk size, including overlap.
# If CHUNK_SIZE = 2000, every materialized DocChunk must contain <= 2000 tokens total.
CHUNK_SIZE = RUNTIME_PROFILE["default_chunk_size"]
OVERLAP_SENTENCES = RUNTIME_PROFILE["default_overlap_sentences"]
MAX_EXPANDED_CHUNK_TOKENS = CHUNK_SIZE

CHUNK_SIZE_SEMANTICS = "expanded_chunk_tokens_including_overlap"


# ---------------------------------------------------------------------------
# Runtime utilities
# ---------------------------------------------------------------------------

def memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    try:
        import resource

        snapshot["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    except Exception:
        pass

    return snapshot


def cleanup_after_chunk(runtime_profile: dict[str, Any] = RUNTIME_PROFILE) -> None:
    gc.collect()

    if str(runtime_profile.get("device", "")).startswith("cuda"):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
