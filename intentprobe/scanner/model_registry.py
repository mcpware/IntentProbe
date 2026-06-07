"""Model and feature extraction registry for intentprobe scanner runtime."""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: str
    hf_model_id: str | None
    default_layers: tuple[int, ...] | None = None
    default_depths: tuple[float, ...] = (0.25, 0.40, 0.55)
    notes: str = ""
    gated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SaeSpec:
    name: str
    sensor: str
    release: str
    sae_id: str
    layer: int | None
    hook_name: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureBundle:
    model_name: str
    feature_kind: str
    layers: tuple[int, ...]
    features_by_layer: dict[int, np.ndarray]
    elapsed_seconds: float
    details: dict


SENSORS: dict[str, SensorSpec] = {
    "lexical-smoke": SensorSpec(
        name="lexical-smoke",
        kind="lexical",
        hf_model_id=None,
        default_layers=(0,),
        notes="Cheap deterministic feature set for harness smoke tests only.",
    ),
    "gpt2": SensorSpec(
        name="gpt2",
        kind="transformer",
        hf_model_id="gpt2",
        default_layers=(3,),
        notes="Legacy paper baseline; layer 3 is the old best layer.",
    ),
    "pythia-70m": SensorSpec(
        name="pythia-70m",
        kind="transformer",
        hf_model_id="EleutherAI/pythia-70m-deduped",
        notes="Tiny CI canary and cheap layer-sweep candidate.",
    ),
    "smollm2-135m": SensorSpec(
        name="smollm2-135m",
        kind="transformer",
        hf_model_id="HuggingFaceTB/SmolLM2-135M",
        notes="Small local fallback seen in earlier experiments.",
    ),
    "qwen2.5-0.5b": SensorSpec(
        name="qwen2.5-0.5b",
        kind="transformer",
        hf_model_id="Qwen/Qwen2.5-0.5B",
        notes="Small local fallback seen in earlier experiments.",
    ),
    "gemma-3-270m": SensorSpec(
        name="gemma-3-270m",
        kind="transformer",
        hf_model_id="google/gemma-3-270m",
        notes="Fast local product-candidate sensor model.",
        gated=True,
    ),
    "gemma-3-1b-pt": SensorSpec(
        name="gemma-3-1b-pt",
        kind="transformer",
        hf_model_id="google/gemma-3-1b-pt",
        notes="Main local product-candidate sensor model.",
        gated=True,
    ),
    "gemma-2-2b": SensorSpec(
        name="gemma-2-2b",
        kind="transformer",
        hf_model_id="google/gemma-2-2b",
        notes="Offline/golden audit reference, not default always-on scanner.",
        gated=True,
    ),
}


SAES: dict[str, SaeSpec] = {
    "gpt2-res-jb-l7": SaeSpec(
        name="gpt2-res-jb-l7",
        sensor="gpt2",
        release="gpt2-small-res-jb",
        sae_id="blocks.7.hook_resid_pre",
        layer=7,
        hook_name="blocks.7.hook_resid_pre",
        notes="Existing GPT-2 SAE used by the earlier cross-style experiments.",
    ),
    "pythia-70m-deduped-l2": SaeSpec(
        name="pythia-70m-deduped-l2",
        sensor="pythia-70m",
        release="pythia-70m-deduped-res-sm",
        sae_id="blocks.2.hook_resid_post",
        layer=2,
        hook_name="blocks.2.hook_resid_post",
        notes="Pythia residual-post SAE; load smoke passed locally.",
    ),
    "gemma-scope-2-270m-pt": SaeSpec(
        name="gemma-scope-2-270m-pt",
        sensor="gemma-3-270m",
        release="gemma-scope-2-270m-pt-res-all",
        sae_id="layer_10_width_16k_l0_small",
        layer=10,
        hook_name="blocks.10.hook_resid_post",
        notes="Gemma Scope 2 residual-post SAE; load smoke passed locally.",
    ),
    "gemma-scope-2-1b-pt": SaeSpec(
        name="gemma-scope-2-1b-pt",
        sensor="gemma-3-1b-pt",
        release="gemma-scope-2-1b-pt-res-all",
        sae_id="layer_14_width_16k_l0_small",
        layer=14,
        hook_name="blocks.14.hook_resid_post",
        notes="Gemma Scope 2 residual-post SAE; inferred from repo layout, not yet smoke-tested.",
    ),
    "gemma-scope-2b-l15": SaeSpec(
        name="gemma-scope-2b-l15",
        sensor="gemma-2-2b",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_15/width_16k/canonical",
        layer=15,
        notes="Existing Gemma-2-2b audit reference.",
    ),
}


SUSPICIOUS_TERMS = (
    "ignore",
    "instruction",
    "system prompt",
    "secret",
    "token",
    "credential",
    "api key",
    "exfiltrate",
    "send",
    "email",
    "upload",
    "hidden",
    "background",
    "silently",
    "override",
    "bypass",
    "shell",
    "execute",
    "download",
    "webhook",
)

SAFE_TERMS = (
    "return",
    "json",
    "read",
    "list",
    "search",
    "weather",
    "calendar",
    "file",
    "calculate",
    "summarize",
)

_TRANSFORMER_MODEL_CACHE: dict[tuple[str, str, bool, str], tuple[Any, Any, Any, int]] = {}
_SAE_CACHE: dict[tuple[str, str, str], Any] = {}


def torch_dtype_for_name(torch: Any, dtype: str) -> Any | None:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "auto":
        return None
    raise ValueError("dtype must be one of: auto, float32, bfloat16")


def load_transformer_runtime(
    spec: SensorSpec,
    *,
    dtype: str,
    local_files_only: bool,
    device: str,
) -> tuple[Any, Any, Any, int]:
    """Load and cache the HF tokenizer/model inside a warm scanner process."""

    if not spec.hf_model_id:
        raise ValueError(f"Sensor {spec.name} has no Hugging Face model id")

    cache_key = (spec.name, dtype, bool(local_files_only), str(device))
    cached = _TRANSFORMER_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "output_hidden_states": True,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    torch_dtype = torch_dtype_for_name(torch, dtype)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModel.from_pretrained(spec.hf_model_id, **model_kwargs).eval()
    model.to(device)
    n_hidden_layers = int(model.config.num_hidden_layers)
    loaded = (torch, tokenizer, model, n_hidden_layers)
    _TRANSFORMER_MODEL_CACHE[cache_key] = loaded
    return loaded


def load_sae_runtime(sae_spec: SaeSpec, *, device: str) -> Any:
    cache_key = (sae_spec.release, sae_spec.sae_id, str(device))
    cached = _SAE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from sae_lens import SAE

    sae_result = SAE.from_pretrained(
        release=sae_spec.release,
        sae_id=sae_spec.sae_id,
        device=device,
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    _SAE_CACHE[cache_key] = sae
    return sae


def sensor_names() -> list[str]:
    return sorted(SENSORS)


def sae_names() -> list[str]:
    return sorted(SAES)


def get_sensor(name: str) -> SensorSpec:
    try:
        return SENSORS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown sensor {name!r}; known sensors: {', '.join(sensor_names())}") from exc


def get_sae(name: str) -> SaeSpec:
    try:
        return SAES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown SAE {name!r}; known SAEs: {', '.join(sae_names())}") from exc


def parse_layers(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    layers = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not layers:
        return None
    return layers


def lexical_features(texts: Iterable[str]) -> np.ndarray:
    rows = []
    for text in texts:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_]+", lowered)
        token_count = max(1, len(tokens))
        char_count = max(1, len(text))
        suspicious = sum(lowered.count(term) for term in SUSPICIOUS_TERMS)
        safe = sum(lowered.count(term) for term in SAFE_TERMS)
        rows.append(
            [
                math.log1p(char_count),
                math.log1p(token_count),
                suspicious,
                safe,
                suspicious / token_count,
                safe / token_count,
                lowered.count("."),
                lowered.count(":"),
                lowered.count("@"),
                lowered.count("/"),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def select_layers(
    n_hidden_layers: int,
    spec: SensorSpec,
    requested_layers: tuple[int, ...] | None,
    layer_sweep: bool,
    depths: tuple[float, ...] | None = None,
) -> tuple[int, ...]:
    max_layer = n_hidden_layers
    if requested_layers:
        layers = requested_layers
    elif layer_sweep:
        layers = tuple(range(max_layer + 1))
    elif spec.default_layers:
        layers = spec.default_layers
    else:
        use_depths = depths or spec.default_depths
        layers = tuple(sorted({max(1, min(max_layer, round(depth * max_layer))) for depth in use_depths}))

    bad = [layer for layer in layers if layer < 0 or layer > max_layer]
    if bad:
        raise ValueError(f"Layer(s) {bad} outside valid range 0..{max_layer} for {spec.name}")
    return tuple(sorted(set(layers)))


def extract_features(
    sensor_name: str,
    texts: list[str],
    *,
    layers: tuple[int, ...] | None = None,
    layer_sweep: bool = False,
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cpu",
    dtype: str = "float32",
    local_files_only: bool = False,
) -> FeatureBundle:
    spec = get_sensor(sensor_name)
    started = time.perf_counter()

    if spec.kind == "lexical":
        matrix = lexical_features(texts)
        return FeatureBundle(
            model_name=spec.name,
            feature_kind="lexical",
            layers=(0,),
            features_by_layer={0: matrix},
            elapsed_seconds=time.perf_counter() - started,
            details={"feature_count": int(matrix.shape[1]), "notes": spec.notes},
        )

    if spec.kind != "transformer" or not spec.hf_model_id:
        raise ValueError(f"Unsupported sensor kind for {sensor_name}: {spec.kind}")

    torch, tokenizer, model, n_hidden_layers = load_transformer_runtime(
        spec,
        dtype=dtype,
        local_files_only=local_files_only,
        device=device,
    )
    selected_layers = select_layers(n_hidden_layers, spec, layers, layer_sweep)
    chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
        denom = mask.sum(dim=1).clamp_min(1)
        for layer in selected_layers:
            hidden = output.hidden_states[layer].to(torch.float32)
            mean = (hidden * mask).sum(dim=1) / denom
            chunks[layer].append(mean.cpu().numpy())

    features_by_layer = {layer: np.vstack(parts) for layer, parts in chunks.items()}
    elapsed = time.perf_counter() - started

    return FeatureBundle(
        model_name=spec.name,
        feature_kind="raw_activation",
        layers=selected_layers,
        features_by_layer=features_by_layer,
        elapsed_seconds=elapsed,
        details={
            "hf_model_id": spec.hf_model_id,
            "n_hidden_layers": n_hidden_layers,
            "batch_size": batch_size,
            "max_length": max_length,
            "device": device,
            "dtype": dtype,
            "gated": spec.gated,
        },
    )


def sae_registry_for_sensor(sensor_name: str) -> list[SaeSpec]:
    return [spec for spec in SAES.values() if spec.sensor == sensor_name]


def default_sae_for_sensor(sensor_name: str) -> SaeSpec:
    matches = sae_registry_for_sensor(sensor_name)
    if not matches:
        raise ValueError(f"No SAE registered for sensor {sensor_name!r}")
    return matches[0]


def hidden_state_index_for_sae(sae_spec: SaeSpec) -> int:
    if sae_spec.layer is None:
        raise ValueError(f"SAE {sae_spec.name} has no layer metadata")
    hook = sae_spec.hook_name or ""
    if "hook_resid_pre" in hook:
        return sae_spec.layer
    if "hook_resid_post" in hook:
        return sae_spec.layer + 1
    return sae_spec.layer


def extract_sae_features(
    sensor_name: str,
    sae_name: str | None,
    texts: list[str],
    *,
    batch_size: int = 4,
    max_length: int = 256,
    device: str = "cpu",
    dtype: str = "float32",
    local_files_only: bool = False,
) -> FeatureBundle:
    sensor = get_sensor(sensor_name)
    sae_spec = get_sae(sae_name) if sae_name else default_sae_for_sensor(sensor_name)
    if sae_spec.sensor != sensor_name:
        raise ValueError(f"SAE {sae_spec.name} belongs to {sae_spec.sensor}, not {sensor_name}")
    if not sensor.hf_model_id:
        raise ValueError(f"Sensor {sensor_name} has no Hugging Face model id")

    started = time.perf_counter()
    torch, tokenizer, model, _n_hidden_layers = load_transformer_runtime(
        sensor,
        dtype=dtype,
        local_files_only=local_files_only,
        device=device,
    )
    sae = load_sae_runtime(sae_spec, device=device)
    hs_index = hidden_state_index_for_sae(sae_spec)
    chunks: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        hidden = output.hidden_states[hs_index].to(torch.float32)
        if hidden.shape[-1] != sae.cfg.d_in:
            raise ValueError(
                f"Hidden width {hidden.shape[-1]} does not match SAE d_in {sae.cfg.d_in} "
                f"for {sae_spec.name}"
            )
        flat = hidden.reshape(-1, hidden.shape[-1])
        encoded_sae = sae.encode(flat).to(torch.float32)
        encoded_sae = encoded_sae.reshape(hidden.shape[0], hidden.shape[1], -1)
        mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
        pooled = (encoded_sae * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        chunks.append(pooled.cpu().numpy())

    matrix = np.vstack(chunks)
    elapsed = time.perf_counter() - started
    layer = sae_spec.layer if sae_spec.layer is not None else hs_index
    return FeatureBundle(
        model_name=sensor.name,
        feature_kind="sae",
        layers=(layer,),
        features_by_layer={layer: matrix},
        elapsed_seconds=elapsed,
        details={
            "hf_model_id": sensor.hf_model_id,
            "sae_name": sae_spec.name,
            "sae_release": sae_spec.release,
            "sae_id": sae_spec.sae_id,
            "hook_name": sae_spec.hook_name,
            "hidden_state_index": hs_index,
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
            "batch_size": batch_size,
            "max_length": max_length,
            "device": device,
            "dtype": dtype,
        },
    )
