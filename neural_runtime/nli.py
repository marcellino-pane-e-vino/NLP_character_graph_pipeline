from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gc
import math
from typing import Any


DEFAULT_NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


@dataclass(frozen=True)
class DirectNLIConfig:
    """Configuration for lower-level direct-NLI pair scoring."""

    pair_batch_size: int = 16
    truncation: bool = True
    max_length: int | None = None
    device: str | None = None


def softmax_values(values: list[float]) -> list[float]:
    """Return a stable softmax over a list of numeric values."""

    if not values:
        return []

    max_value = max(float(value) for value in values)
    exps = [math.exp(float(value) - max_value) for value in values]
    denominator = sum(exps)

    if denominator <= 0.0:
        return [1.0 / len(values)] * len(values)

    return [value / denominator for value in exps]


def device_name(device: str | None = None) -> str:
    """Return the configured device name, or the auto-detected torch device."""

    import torch

    if device is not None:
        return str(device)

    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"

    return "cpu"


def release_chunk_memory() -> None:
    """Best-effort memory cleanup after a scoring chunk."""

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda_if_available() -> None:
    """Synchronize CUDA work when CUDA is available."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def find_entailment_id(model: Any) -> int:
    """Infer the entailment logit index from a HuggingFace sequence-classification model."""

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

    raise ValueError("Could not infer entailment label id from model config.")


@lru_cache(maxsize=4)
def _load_direct_nli_components(
    *,
    model_name: str,
    device: str | None,
) -> tuple[Any, Any, Any, int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    torch_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.to(torch_device)
    model.eval()

    entailment_id = find_entailment_id(model)
    return tokenizer, model, torch_device, entailment_id


def direct_entailment_logits_for_pairs(
    pairs: list[tuple[str, str]],
    *,
    model_name: str = DEFAULT_NLI_MODEL_NAME,
    nli_config: DirectNLIConfig | None = None,
) -> list[float]:
    """Score premise/hypothesis pairs and return one entailment logit per pair."""

    import torch

    nli_config = nli_config or DirectNLIConfig()

    if nli_config.pair_batch_size <= 0:
        raise ValueError(
            f"pair_batch_size must be > 0, got {nli_config.pair_batch_size}"
        )

    if not pairs:
        return []

    tokenizer, model, device, entailment_id = _load_direct_nli_components(
        model_name=model_name,
        device=nli_config.device,
    )

    scores: list[float] = []

    for start in range(0, len(pairs), nli_config.pair_batch_size):
        batch = pairs[start : start + nli_config.pair_batch_size]
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

        scores.extend(float(value) for value in batch_logits.tolist())

        del batch, premises, hypotheses, encoded, output, batch_logits
        release_chunk_memory()

    return scores


def entailment_probabilities_for_hypotheses(
    *,
    premise: str,
    hypotheses: list[str],
    model_name: str = DEFAULT_NLI_MODEL_NAME,
    nli_config: DirectNLIConfig | None = None,
) -> list[float]:
    """Score one premise against hypotheses and return grouped-softmax probabilities."""

    logits = direct_entailment_logits_for_pairs(
        [(premise, hypothesis) for hypothesis in hypotheses],
        model_name=model_name,
        nli_config=nli_config,
    )
    return softmax_values([float(value) for value in logits])
