import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gguf
import numpy as np
import torch
from pydantic import AliasChoices, ConfigDict, Field
from torch import Tensor, nn

from src.llama import (
    LlamaCppSessionConfig,
    LlamaEmbeddingSession,
    LlamaTokenEmbeddings,
)
from src.nodes.base_node import BaseNode, BaseNodeModel
from src.prompt import process_prompt

CWD = Path(os.getcwd())
QWEN_SDXL_NEGATIVE_CACHE_DIR = CWD / "caches" / "artifacts" / "qwen_sdxl_negatives"
QWEN_SDXL_NEGATIVE_CACHE_VERSION = 1


class QwenInputs(BaseNodeModel):
    prompt: list[str] | str = Field(..., description="Text prompt for image generation")
    negative_prompt: list[str] | str | None = Field(
        None, description="Negative text prompt for image generation"
    )
    model: str = Field("juggernaut", description="Model to use for image generation")
    qwen_model_path: str | None = Field(
        None,
        description="Local GGUF path for the Qwen3.5-27B llama.cpp model",
    )
    qwen_llama_cpp_lib_path: str | None = Field(
        None,
        description="Directory containing, or full path to, the libllama shared library used by llama-cpp-python",
    )
    qwen_ignore_env_llama_cpp_lib_path: bool = Field(
        True,
        description="When true, ignore LLAMA_CPP_LIB_PATH from the process environment and use the packaged llama-cpp-python runtime unless qwen_llama_cpp_lib_path is set explicitly",
    )
    qwen_llama_cpp_extra_lib_paths: list[str] | None = Field(
        None,
        description="Optional extra shared-library directories to prepend before importing llama-cpp-python",
    )
    qwen_llama_cpp_preload_libs: list[str] | None = Field(
        None,
        description="Optional shared libraries to preload before importing llama-cpp-python, useful for ROCm dependencies",
    )
    qwen_n_ctx: int = Field(512, ge=512, description="llama.cpp context size")
    qwen_n_batch: int = Field(128, ge=1, description="llama.cpp prompt batch size")
    qwen_n_gpu_layers: int = Field(
        0, description="Number of Qwen layers to offload to GPU"
    )
    qwen_n_threads: int | None = Field(
        None, ge=1, description="Optional CPU thread count for llama.cpp"
    )
    qwen_offload_kqv: bool = Field(
        False,
        description="When false, keep llama.cpp K/Q/V execution on CPU even on ROCm builds",
    )
    qwen_op_offload: bool | None = Field(
        False,
        description="When false, keep individual llama.cpp ops on CPU for CPU-only safety",
    )
    qwen_pooling: Literal["mean"] = Field(
        "mean",
        description="Mean pooling used before the pooled SDXL projector",
    )
    qwen_normalize_embeddings: bool = Field(
        False,
        description="When true, request normalized llama.cpp pooled embeddings before projection",
    )
    use_input_layernorm: bool | None = Field(
        None,
        validation_alias=AliasChoices("use_input_layernorm", "qwen_normalize_layers"),
        description="Optional override for projector trunk LayerNorm. When unset, follow the GGUF artifact layout; when true, enable trunk LayerNorm with identity weights if the artifact does not store it; when false, disable it even if the artifact stores it.",
    )
    qwen_use_cached_negative_prompt_embeds: bool = Field(
        True,
        description="When true, load cached SDXL negative prompt embeddings for the negative branch",
    )
    projector_seed: int = Field(
        1337,
        description="Seed for deterministic placeholder projector initialization",
    )
    projector_path: str | None = Field(
        None,
        description="Optional state dict checkpoint for the Qwen-to-SDXL projector; falls back to QWEN_2_SDXL_PROJECTOR_PATH when unset",
    )
    qwen_negative_prompt_cache_dir: str | None = Field(
        None,
        description="Optional directory containing exported SDXL negative prompt embedding artifacts",
    )
    model_config = ConfigDict(extra="allow")


class PromptEmbeds:
    def __init__(
        self,
        prompt_embeds: Tensor,
        pooled_prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor,
        negative_pooled_prompt_embeds: Tensor,
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

    def keys(self):
        return [
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)


def replace_negative_prompt_embeds(
    positive_embeds: PromptEmbeds,
    negative_embeds,
) -> PromptEmbeds:
    prompt_embeds = positive_embeds.prompt_embeds
    pooled_prompt_embeds = positive_embeds.pooled_prompt_embeds
    negative_prompt_embeds = negative_embeds.negative_prompt_embeds.to(
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    negative_pooled_prompt_embeds = negative_embeds.negative_pooled_prompt_embeds.to(
        device=pooled_prompt_embeds.device,
        dtype=pooled_prompt_embeds.dtype,
    )

    if int(prompt_embeds.shape[0]) != int(negative_prompt_embeds.shape[0]):
        raise RuntimeError(
            "Positive and negative prompt embeddings must have the same batch size."
        )
    if int(pooled_prompt_embeds.shape[0]) != int(
        negative_pooled_prompt_embeds.shape[0]
    ):
        raise RuntimeError(
            "Positive and negative pooled prompt embeddings must have the same batch size."
        )

    return PromptEmbeds(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )


def _sanitize_cache_key_part(value: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in value
    ).strip("_")
    return sanitized or "value"


def negative_prompt_cache_path(
    model: str,
    negative_prompt: str,
    cache_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else QWEN_SDXL_NEGATIVE_CACHE_DIR
    )
    model_slug = _sanitize_cache_key_part(model.lower())
    prompt_hash = hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest()[:16]
    return base_dir / model_slug / f"{prompt_hash}.pt"


def _normalize_cached_negative_prompt_embeds(
    negative_prompt_embeds: Tensor,
    negative_pooled_prompt_embeds: Tensor,
) -> tuple[Tensor, Tensor]:
    prompt_embeds = negative_prompt_embeds.detach().to(device="cpu")
    pooled_embeds = negative_pooled_prompt_embeds.detach().to(device="cpu")

    if prompt_embeds.ndim == 3:
        if int(prompt_embeds.shape[0]) != 1:
            raise ValueError(
                "Cached SDXL negative prompt embeds must represent exactly one prompt."
            )
        prompt_embeds = prompt_embeds.squeeze(0)
    if pooled_embeds.ndim == 2:
        if int(pooled_embeds.shape[0]) != 1:
            raise ValueError(
                "Cached SDXL negative pooled embeds must represent exactly one prompt."
            )
        pooled_embeds = pooled_embeds.squeeze(0)

    if prompt_embeds.ndim != 2:
        raise ValueError(
            "Cached SDXL negative prompt embeds must have shape [seq_len, hidden]."
        )
    if pooled_embeds.ndim != 1:
        raise ValueError(
            "Cached SDXL negative pooled embeds must have shape [pooled_hidden]."
        )

    return prompt_embeds.contiguous(), pooled_embeds.contiguous()


def save_sdxl_negative_prompt_embeddings(
    model: str,
    negative_prompt: str,
    negative_prompt_embeds: Tensor,
    negative_pooled_prompt_embeds: Tensor,
    cache_dir: str | Path | None = None,
) -> Path:
    cache_path = negative_prompt_cache_path(
        model=model,
        negative_prompt=negative_prompt,
        cache_dir=cache_dir,
    )
    prompt_embeds, pooled_embeds = _normalize_cached_negative_prompt_embeds(
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": QWEN_SDXL_NEGATIVE_CACHE_VERSION,
            "model": model,
            "negative_prompt": negative_prompt,
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_embeds,
        },
        cache_path,
    )
    return cache_path


def load_sdxl_negative_prompt_embeddings(
    model: str,
    negative_prompt: str,
    target_device: torch.device | str | None = None,
    target_dtype: torch.dtype | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Tensor, Tensor]:
    cache_path = negative_prompt_cache_path(
        model=model,
        negative_prompt=negative_prompt,
        cache_dir=cache_dir,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing cached SDXL negative embeddings for model '{model}'. Export them first: {cache_path}"
        )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != QWEN_SDXL_NEGATIVE_CACHE_VERSION:
        raise RuntimeError(
            f"Unsupported cached SDXL negative embedding artifact version in {cache_path}."
        )
    if payload.get("model") != model:
        raise RuntimeError(
            f"Cached SDXL negative embeddings at {cache_path} were exported for a different model."
        )
    if payload.get("negative_prompt") != negative_prompt:
        raise RuntimeError(
            f"Cached SDXL negative embeddings at {cache_path} were exported for a different negative prompt."
        )

    prompt_embeds, pooled_embeds = _normalize_cached_negative_prompt_embeds(
        payload["negative_prompt_embeds"],
        payload["negative_pooled_prompt_embeds"],
    )

    if target_device is not None and not isinstance(target_device, torch.device):
        target_device = torch.device(target_device)

    if target_device is not None or target_dtype is not None:
        prompt_embeds = prompt_embeds.to(device=target_device, dtype=target_dtype)
        pooled_embeds = pooled_embeds.to(device=target_device, dtype=target_dtype)

    return prompt_embeds, pooled_embeds


class QwenToSdxlProjector(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        output_hidden_size: int,
        pooled_output_size: int,
        seed: int,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.token_projector = nn.Linear(input_hidden_size, output_hidden_size)
        self.pooled_projector = nn.Linear(input_hidden_size, pooled_output_size)
        self._initialize(seed)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        self.eval()

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for layer in (self.token_projector, self.pooled_projector):
            weight = torch.empty_like(layer.weight, dtype=torch.float32)
            weight.normal_(mean=0.0, std=0.02, generator=generator)
            bias = torch.zeros_like(layer.bias, dtype=torch.float32)
            layer.weight = nn.Parameter(weight)
            layer.bias = nn.Parameter(bias)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        resolved = Path(checkpoint_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Projector checkpoint not found: {resolved}")

        if resolved.suffix.lower() == ".gguf":
            raise RuntimeError(
                f"Projector checkpoint at {resolved} is a GGUF artifact. QwenToSdxlProjector currently expects a PyTorch checkpoint, not GGUF."
            )

        state_dict = torch.load(resolved, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and isinstance(
            state_dict.get("state_dict"), dict
        ):
            state_dict = state_dict["state_dict"]

        try:
            self.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Projector checkpoint at {resolved} does not match the expected Qwen-to-SDXL projector shape."
            ) from exc

    def project(
        self,
        llama_embeddings: LlamaTokenEmbeddings,
        target_seq_len: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
        pooling: Literal["mean", "last"],
    ) -> tuple[Tensor, Tensor]:
        del pooling
        with torch.no_grad():
            pooled_source = torch.from_numpy(llama_embeddings.embedding)
            if pooled_source.ndim != 1:
                raise RuntimeError(
                    "Qwen projector expects a mean-pooled llama embedding vector."
                )
            prompt_embeds = self.token_projector(pooled_source).unsqueeze(0)
            prompt_embeds = _align_sequence_length(prompt_embeds, target_seq_len)
            pooled_embeds = self.pooled_projector(pooled_source)

        return (
            prompt_embeds.to(device=target_device, dtype=target_dtype),
            pooled_embeds.to(device=target_device, dtype=target_dtype),
        )


class QwenToSdProjector(nn.Module):
    def __init__(self, qwen_dim: int, sd_dim: int, hidden_dim: int):
        super().__init__()
        self.qwen_dim = qwen_dim
        self.sd_dim = sd_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sd_dim),
        )

    def forward(self, qwen_embedding: Tensor) -> Tensor:
        squeeze_output = qwen_embedding.ndim == 1
        if squeeze_output:
            qwen_embedding = qwen_embedding.unsqueeze(0)
        projected = self.projection(qwen_embedding)
        return projected.squeeze(0) if squeeze_output else projected


class _QwenToSdxlProjectorTrunkLayer(nn.Module):
    def __init__(self, hidden_dim: int, residual: bool, use_layernorm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, hidden: Tensor) -> Tensor:
        update = self.fc1(hidden)
        update = self.layernorm(update)
        update = self.activation(update)
        update = self.fc2(update)
        if self.residual:
            return hidden + update
        return update


class _OutputAffineCalibrator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden * self.gain + self.bias


@dataclass(frozen=True)
class EmbeddingStandardizationStats:
    qwen_mean: Tensor
    qwen_std: Tensor
    target_means: dict[str, Tensor]
    target_stds: dict[str, Tensor]


class QwenToSdxlGgufProjector(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        prompt_seq_len: int,
        prompt_dim: int,
        pooled_dim: int,
        hidden_dim: int,
        prompt_token_dim: int,
        trunk_depth: int = 0,
        residual_trunk: bool = False,
        prompt_head_hidden_dim: int | None = None,
        pooled_head_hidden_dim: int | None = None,
        use_input_layernorm: bool = False,
        prompt_head_second_activation: bool = True,
        use_output_calibrator: bool = False,
        output_mode: str = "raw",
    ):
        super().__init__()
        self.qwen_dim = qwen_dim
        self.prompt_seq_len = prompt_seq_len
        self.prompt_dim = prompt_dim
        self.pooled_dim = pooled_dim
        self.hidden_dim = hidden_dim
        self.prompt_token_dim = prompt_token_dim
        self.trunk_depth = trunk_depth
        self.residual_trunk = residual_trunk
        self.prompt_head_hidden_dim = prompt_head_hidden_dim or prompt_dim
        self.pooled_head_hidden_dim = pooled_head_hidden_dim or hidden_dim
        self.use_input_layernorm = use_input_layernorm
        self.prompt_head_second_activation = prompt_head_second_activation
        self.use_output_calibrator = use_output_calibrator
        self.output_mode = output_mode
        self.input_projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trunk_layers = nn.ModuleList(
            _QwenToSdxlProjectorTrunkLayer(
                hidden_dim,
                residual_trunk,
                use_layernorm=use_input_layernorm,
            )
            for _ in range(trunk_depth)
        )
        self.prompt_seed = nn.Linear(hidden_dim, prompt_seq_len * prompt_token_dim)
        self.prompt_projection = nn.Sequential(
            nn.GELU(),
            nn.Linear(prompt_token_dim, self.prompt_head_hidden_dim),
            nn.GELU() if prompt_head_second_activation else nn.Identity(),
            nn.Linear(self.prompt_head_hidden_dim, prompt_dim),
        )
        self.pooled_head = nn.Sequential(
            nn.Linear(hidden_dim, self.pooled_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.pooled_head_hidden_dim, pooled_dim),
        )
        self.prompt_output_calibrator = (
            _OutputAffineCalibrator(prompt_dim)
            if use_output_calibrator
            else nn.Identity()
        )
        self.pooled_output_calibrator = (
            _OutputAffineCalibrator(pooled_dim)
            if use_output_calibrator
            else nn.Identity()
        )
        self.uses_embedding_standardization = False
        self.embedding_standardization_stats: EmbeddingStandardizationStats | None = None
        self.requires_input_standardization = False
        self.requires_output_denormalization = False

    def forward(self, qwen_embedding: Tensor) -> tuple[Tensor, Tensor]:
        squeeze_output = qwen_embedding.ndim == 1
        if squeeze_output:
            qwen_embedding = qwen_embedding.unsqueeze(0)

        hidden = self.input_projection(qwen_embedding)
        for layer in self.trunk_layers:
            hidden = layer(hidden)
        prompt_seed = self.prompt_seed(hidden).view(
            hidden.shape[0], self.prompt_seq_len, self.prompt_token_dim
        )
        prompt_embeds = self.prompt_projection(prompt_seed)
        pooled_embeds = self.pooled_head(hidden)
        prompt_embeds = self.prompt_output_calibrator(prompt_embeds)
        pooled_embeds = self.pooled_output_calibrator(pooled_embeds)

        if squeeze_output:
            return prompt_embeds.squeeze(0), pooled_embeds.squeeze(0)
        return prompt_embeds, pooled_embeds


def _gguf_field_value(reader: gguf.GGUFReader, key: str, default):
    field = reader.fields.get(key)
    if field is None or not field.parts:
        return default

    value = np.array(field.parts[-1], copy=True)
    if value.dtype == np.uint8:
        return bytes(value.tolist()).decode("utf-8")
    if value.size == 1:
        return value.reshape(()).item()
    return value.tolist()


def _load_embedding_standardization_stats(
    tensor_map: dict[str, Tensor],
) -> EmbeddingStandardizationStats | None:
    tensor_names = {
        "qwen_mean": "standardization.qwen.mean",
        "qwen_std": "standardization.qwen.std",
        "prompt_mean": "standardization.prompt_embeds.mean",
        "prompt_std": "standardization.prompt_embeds.std",
        "pooled_mean": "standardization.pooled_prompt_embeds.mean",
        "pooled_std": "standardization.pooled_prompt_embeds.std",
    }
    present_names = [name for name in tensor_names.values() if name in tensor_map]
    if not present_names:
        return None

    missing_names = [name for name in tensor_names.values() if name not in tensor_map]
    if missing_names:
        missing_text = ", ".join(sorted(missing_names))
        raise RuntimeError(
            "GGUF projector is missing part of the embedding standardization payload: "
            f"{missing_text}"
        )

    return EmbeddingStandardizationStats(
        qwen_mean=tensor_map[tensor_names["qwen_mean"]].clone(),
        qwen_std=tensor_map[tensor_names["qwen_std"]].clone(),
        target_means={
            "prompt_embeds": tensor_map[tensor_names["prompt_mean"]].clone(),
            "pooled_prompt_embeds": tensor_map[tensor_names["pooled_mean"]].clone(),
        },
        target_stds={
            "prompt_embeds": tensor_map[tensor_names["prompt_std"]].clone(),
            "pooled_prompt_embeds": tensor_map[tensor_names["pooled_std"]].clone(),
        },
    )


def _apply_feature_standardization(hidden: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    mean = mean.to(device=hidden.device, dtype=hidden.dtype)
    std = std.to(device=hidden.device, dtype=hidden.dtype)
    return (hidden - mean) / std


def _apply_feature_denormalization(hidden: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    mean = mean.to(device=hidden.device, dtype=hidden.dtype)
    std = std.to(device=hidden.device, dtype=hidden.dtype)
    return hidden * std + mean


def project_sdxl_qwen_embedding(
    projector: nn.Module,
    qwen_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    transformed_embedding = qwen_embedding
    if getattr(projector, "uses_embedding_standardization", False):
        stats = getattr(projector, "embedding_standardization_stats", None)
        if stats is None:
            raise RuntimeError(
                "Projector requires embedding standardization, but no standardization stats are loaded."
            )
        transformed_embedding = _apply_feature_standardization(
            qwen_embedding,
            stats.qwen_mean,
            stats.qwen_std,
        )

    prompt_embeds, pooled_embeds = projector(transformed_embedding)

    if getattr(projector, "uses_embedding_standardization", False):
        stats = getattr(projector, "embedding_standardization_stats", None)
        if stats is None:
            raise RuntimeError(
                "Projector requires embedding denormalization, but no standardization stats are loaded."
            )
        prompt_embeds = _apply_feature_denormalization(
            prompt_embeds,
            stats.target_means["prompt_embeds"],
            stats.target_stds["prompt_embeds"],
        )
        pooled_embeds = _apply_feature_denormalization(
            pooled_embeds,
            stats.target_means["pooled_prompt_embeds"],
            stats.target_stds["pooled_prompt_embeds"],
        )

    return prompt_embeds, pooled_embeds


def _normalize_sdxl_projector_state_dict_layout(
    state_dict_tensors: dict[str, Tensor],
) -> dict[str, Tensor]:
    normalized_state_dict = dict(state_dict_tensors)

    if (
        "input_projection.3.weight" in normalized_state_dict
        and "input_projection.2.weight" not in normalized_state_dict
    ):
        normalized_state_dict["input_projection.2.weight"] = normalized_state_dict.pop(
            "input_projection.3.weight"
        )
        normalized_state_dict["input_projection.2.bias"] = normalized_state_dict.pop(
            "input_projection.3.bias"
        )

    if "input_projection.1.weight" in normalized_state_dict:
        layernorm_weight = normalized_state_dict.pop("input_projection.1.weight")
        layernorm_bias = normalized_state_dict.pop("input_projection.1.bias")
        normalized_state_dict.setdefault(
            "trunk_layers.0.layernorm.weight",
            layernorm_weight,
        )
        normalized_state_dict.setdefault(
            "trunk_layers.0.layernorm.bias",
            layernorm_bias,
        )

    if "input_layernorm.weight" in normalized_state_dict:
        layernorm_weight = normalized_state_dict.pop("input_layernorm.weight")
        layernorm_bias = normalized_state_dict.pop("input_layernorm.bias")
        normalized_state_dict.setdefault(
            "trunk_layers.0.layernorm.weight",
            layernorm_weight,
        )
        normalized_state_dict.setdefault(
            "trunk_layers.0.layernorm.bias",
            layernorm_bias,
        )

    return normalized_state_dict


def load_projector_from_gguf(
    gguf_path: str | Path,
    device: torch.device | str | None = None,
    use_input_layernorm: bool | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    reader = gguf.GGUFReader(str(gguf_path))
    target_family = _gguf_field_value(reader, "projector.target_family", "sd")
    schema_version = int(_gguf_field_value(reader, "projector.schema_version", 0))
    tensor_map = {
        tensor.name: torch.from_numpy(np.array(tensor.data, copy=True)).float()
        for tensor in reader.tensors
    }
    embedding_standardization_stats = _load_embedding_standardization_stats(tensor_map)

    if target_family == "sdxl":
        state_dict_tensors = {
            name[len("state_dict.") :]: tensor
            for name, tensor in tensor_map.items()
            if name.startswith("state_dict.")
        }
        if state_dict_tensors:
            state_dict_tensors = _normalize_sdxl_projector_state_dict_layout(
                state_dict_tensors
            )
            input_projection_w = state_dict_tensors["input_projection.0.weight"]
            hidden_dim = int(
                _gguf_field_value(
                    reader, "projector.hidden_dim", input_projection_w.shape[0]
                )
            )
            trunk_depth = int(
                _gguf_field_value(
                    reader,
                    "projector.trunk_depth",
                    sum(
                        1
                        for name in state_dict_tensors
                        if name.startswith("trunk_layers.")
                        and name.endswith(".fc1.weight")
                    ),
                )
            )
            artifact_uses_input_layernorm = any(
                name.startswith("trunk_layers.") and ".layernorm." in name
                for name in state_dict_tensors
            )
            artifact_uses_output_calibrator = any(
                name.startswith("prompt_output_calibrator.")
                or name.startswith("pooled_output_calibrator.")
                for name in state_dict_tensors
            )
            resolved_use_input_layernorm = trunk_depth > 0 and (
                artifact_uses_input_layernorm
                if use_input_layernorm is None
                else use_input_layernorm
            )
            if resolved_use_input_layernorm and not artifact_uses_input_layernorm:
                for layer_index in range(trunk_depth):
                    state_dict_tensors.setdefault(
                        f"trunk_layers.{layer_index}.layernorm.weight",
                        torch.ones(hidden_dim, dtype=input_projection_w.dtype),
                    )
                    state_dict_tensors.setdefault(
                        f"trunk_layers.{layer_index}.layernorm.bias",
                        torch.zeros(hidden_dim, dtype=input_projection_w.dtype),
                    )
            if not resolved_use_input_layernorm:
                for name in list(state_dict_tensors):
                    if name.startswith("trunk_layers.") and ".layernorm." in name:
                        state_dict_tensors.pop(name)
            prompt_seed_w = state_dict_tensors["prompt_seed.weight"]
            prompt_head_w1 = state_dict_tensors["prompt_projection.1.weight"]
            prompt_head_w2 = state_dict_tensors["prompt_projection.3.weight"]
            pooled_head_w1 = state_dict_tensors["pooled_head.0.weight"]
            pooled_head_w2 = state_dict_tensors["pooled_head.2.weight"]

            qwen_dim = int(
                _gguf_field_value(
                    reader, "projector.qwen_dim", input_projection_w.shape[1]
                )
            )
            prompt_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_dim", prompt_head_w2.shape[0]
                )
            )
            pooled_dim = int(
                _gguf_field_value(
                    reader, "projector.pooled_dim", pooled_head_w2.shape[0]
                )
            )
            metadata_uses_output_calibrator = bool(
                int(
                    _gguf_field_value(
                        reader,
                        "projector.use_output_calibrator",
                        1 if artifact_uses_output_calibrator else 0,
                    )
                )
            )
            resolved_use_output_calibrator = (
                artifact_uses_output_calibrator or metadata_uses_output_calibrator
            )
            metadata_uses_embedding_standardization = bool(
                int(
                    _gguf_field_value(
                        reader,
                        "projector.uses_embedding_standardization",
                        1 if embedding_standardization_stats is not None else 0,
                    )
                )
            )
            resolved_uses_embedding_standardization = (
                metadata_uses_embedding_standardization
                or embedding_standardization_stats is not None
            )
            if (
                resolved_uses_embedding_standardization
                and embedding_standardization_stats is None
            ):
                raise RuntimeError(
                    "GGUF projector metadata requires embedding standardization, but the standardization tensors are missing."
                )
            output_mode = str(
                _gguf_field_value(
                    reader,
                    "projector.output_mode",
                    "calibrated" if resolved_use_output_calibrator else "raw",
                )
            )
            prompt_token_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_token_dim", prompt_head_w1.shape[1]
                )
            )
            prompt_seq_len = int(
                _gguf_field_value(
                    reader,
                    "projector.prompt_seq_len",
                    prompt_seed_w.shape[0] // max(prompt_token_dim, 1),
                )
            )
            residual_trunk = bool(
                int(_gguf_field_value(reader, "projector.residual_trunk", 1))
            )
            prompt_head_hidden_dim = int(
                _gguf_field_value(
                    reader,
                    "projector.prompt_head_hidden_dim",
                    prompt_head_w1.shape[0],
                )
            )
            pooled_head_hidden_dim = int(
                _gguf_field_value(
                    reader,
                    "projector.pooled_head_hidden_dim",
                    pooled_head_w1.shape[0],
                )
            )
            if resolved_use_output_calibrator:
                state_dict_tensors.setdefault(
                    "prompt_output_calibrator.gain",
                    torch.ones(prompt_dim, dtype=input_projection_w.dtype),
                )
                state_dict_tensors.setdefault(
                    "prompt_output_calibrator.bias",
                    torch.zeros(prompt_dim, dtype=input_projection_w.dtype),
                )
                state_dict_tensors.setdefault(
                    "pooled_output_calibrator.gain",
                    torch.ones(pooled_dim, dtype=input_projection_w.dtype),
                )
                state_dict_tensors.setdefault(
                    "pooled_output_calibrator.bias",
                    torch.zeros(pooled_dim, dtype=input_projection_w.dtype),
                )
            model = QwenToSdxlGgufProjector(
                qwen_dim=qwen_dim,
                prompt_seq_len=prompt_seq_len,
                prompt_dim=prompt_dim,
                pooled_dim=pooled_dim,
                hidden_dim=hidden_dim,
                prompt_token_dim=prompt_token_dim,
                trunk_depth=trunk_depth,
                residual_trunk=residual_trunk,
                prompt_head_hidden_dim=prompt_head_hidden_dim,
                pooled_head_hidden_dim=pooled_head_hidden_dim,
                use_input_layernorm=resolved_use_input_layernorm,
                prompt_head_second_activation=True,
                use_output_calibrator=resolved_use_output_calibrator,
                output_mode=output_mode,
            )
            model.load_state_dict(state_dict_tensors)
            model.embedding_standardization_stats = embedding_standardization_stats
            model.uses_embedding_standardization = (
                resolved_uses_embedding_standardization
            )
            model.requires_input_standardization = (
                resolved_uses_embedding_standardization
            )
            model.requires_output_denormalization = (
                resolved_uses_embedding_standardization
            )
            metadata = {
                "target_family": target_family,
                "schema_version": schema_version,
                "qwen_dim": qwen_dim,
                "hidden_dim": hidden_dim,
                "prompt_seq_len": prompt_seq_len,
                "prompt_dim": prompt_dim,
                "pooled_dim": pooled_dim,
                "prompt_token_dim": prompt_token_dim,
                "trunk_depth": trunk_depth,
                "residual_trunk": residual_trunk,
                "prompt_head_hidden_dim": prompt_head_hidden_dim,
                "pooled_head_hidden_dim": pooled_head_hidden_dim,
                "use_input_layernorm": resolved_use_input_layernorm,
                "use_output_calibrator": resolved_use_output_calibrator,
                "output_mode": output_mode,
                "uses_embedding_standardization": resolved_uses_embedding_standardization,
                "requires_input_standardization": resolved_uses_embedding_standardization,
                "requires_output_denormalization": resolved_uses_embedding_standardization,
                "normalize_input": False,
                "prompt_head_second_activation": True,
            }
        else:
            trunk_w1 = tensor_map["trunk_w1"]
            trunk_b1 = tensor_map["trunk_b1"]
            trunk_w2 = tensor_map["trunk_w2"]
            trunk_b2 = tensor_map["trunk_b2"]
            prompt_seed_w = tensor_map["prompt_seed_w"]
            prompt_seed_b = tensor_map["prompt_seed_b"]
            prompt_proj_w = tensor_map["prompt_proj_w"]
            prompt_proj_b = tensor_map["prompt_proj_b"]
            pooled_w = tensor_map["pooled_w"]
            pooled_b = tensor_map["pooled_b"]

            qwen_dim = int(
                _gguf_field_value(reader, "projector.qwen_dim", trunk_w1.shape[1])
            )
            hidden_dim = int(
                _gguf_field_value(reader, "projector.hidden_dim", trunk_w1.shape[0])
            )
            prompt_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_dim", prompt_proj_w.shape[0]
                )
            )
            pooled_dim = int(
                _gguf_field_value(reader, "projector.pooled_dim", pooled_w.shape[0])
            )
            prompt_token_dim = int(
                _gguf_field_value(
                    reader, "projector.prompt_token_dim", prompt_proj_w.shape[1]
                )
            )
            prompt_seq_len = int(
                _gguf_field_value(
                    reader,
                    "projector.prompt_seq_len",
                    prompt_seed_w.shape[0] // max(prompt_token_dim, 1),
                )
            )
            resolved_use_input_layernorm = False

            model = QwenToSdxlGgufProjector(
                qwen_dim=qwen_dim,
                prompt_seq_len=prompt_seq_len,
                prompt_dim=prompt_dim,
                pooled_dim=pooled_dim,
                hidden_dim=hidden_dim,
                prompt_token_dim=prompt_token_dim,
                trunk_depth=0,
                residual_trunk=False,
                prompt_head_hidden_dim=prompt_dim,
                pooled_head_hidden_dim=hidden_dim,
                use_input_layernorm=resolved_use_input_layernorm,
                prompt_head_second_activation=False,
                use_output_calibrator=False,
                output_mode="raw",
            )
            model.input_projection[0].weight.data.copy_(trunk_w1)
            model.input_projection[0].bias.data.copy_(trunk_b1)
            model.input_projection[2].weight.data.copy_(trunk_w2)
            model.input_projection[2].bias.data.copy_(trunk_b2)
            model.prompt_seed.weight.data.copy_(prompt_seed_w)
            model.prompt_seed.bias.data.copy_(prompt_seed_b)
            model.prompt_projection[1].weight.data.copy_(prompt_proj_w)
            model.prompt_projection[1].bias.data.copy_(prompt_proj_b)
            model.prompt_projection[3].weight.data.copy_(
                torch.eye(prompt_dim, dtype=model.prompt_projection[3].weight.dtype)
            )
            model.prompt_projection[3].bias.data.zero_()
            model.pooled_head[0].weight.data.copy_(
                torch.eye(hidden_dim, dtype=model.pooled_head[0].weight.dtype)
            )
            model.pooled_head[0].bias.data.zero_()
            model.pooled_head[2].weight.data.copy_(pooled_w)
            model.pooled_head[2].bias.data.copy_(pooled_b)
            model.embedding_standardization_stats = None
            model.uses_embedding_standardization = False
            model.requires_input_standardization = False
            model.requires_output_denormalization = False
            metadata = {
                "target_family": target_family,
                "schema_version": schema_version,
                "qwen_dim": qwen_dim,
                "hidden_dim": hidden_dim,
                "prompt_seq_len": prompt_seq_len,
                "prompt_dim": prompt_dim,
                "pooled_dim": pooled_dim,
                "prompt_token_dim": prompt_token_dim,
                "trunk_depth": 0,
                "residual_trunk": False,
                "prompt_head_hidden_dim": prompt_dim,
                "pooled_head_hidden_dim": hidden_dim,
                "use_input_layernorm": resolved_use_input_layernorm,
                "use_output_calibrator": False,
                "output_mode": "raw",
                "uses_embedding_standardization": False,
                "requires_input_standardization": False,
                "requires_output_denormalization": False,
                "normalize_input": False,
                "prompt_head_second_activation": False,
            }
    else:
        w1 = tensor_map["proj_w1"]
        b1 = tensor_map["proj_b1"]
        w2 = tensor_map["proj_w2"]
        b2 = tensor_map["proj_b2"]
        qwen_dim = int(_gguf_field_value(reader, "projector.qwen_dim", w1.shape[1]))
        hidden_dim = int(_gguf_field_value(reader, "projector.hidden_dim", w1.shape[0]))
        sd_dim = int(_gguf_field_value(reader, "projector.sd_dim", w2.shape[0]))

        model = QwenToSdProjector(
            qwen_dim=qwen_dim,
            sd_dim=sd_dim,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(
            {
                "projection.0.weight": w1,
                "projection.0.bias": b1,
                "projection.2.weight": w2,
                "projection.2.bias": b2,
            }
        )
        metadata = {
            "target_family": "sd",
            "qwen_dim": qwen_dim,
            "hidden_dim": hidden_dim,
            "sd_dim": sd_dim,
        }

    model = model.to(device).eval()
    return model, metadata


def _ensure_prompt_list(value: list[str] | str, name: str) -> list[str]:
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f"{name} must contain at least one prompt.")
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain only strings.")
    if any(not item.strip() for item in values):
        raise ValueError(f"{name} entries must be non-empty strings.")
    return values


def _normalize_prompt_batch(
    prompt: list[str] | str,
    negative_prompt: list[str] | str | None,
    model: str,
) -> tuple[list[str], list[str]]:
    prompts = _ensure_prompt_list(prompt, "prompt")
    if negative_prompt is None:
        negative_prompts_in = [None] * len(prompts)
    elif isinstance(negative_prompt, str):
        negative_prompts_in = [negative_prompt] * len(prompts)
    else:
        negative_prompts_in = list(negative_prompt)
        if len(negative_prompts_in) != len(prompts):
            raise ValueError(
                "negative_prompt must be a string, None, or a list with the same length as prompt."
            )

    positive_prompts: list[str] = []
    negative_prompts: list[str] = []
    for raw_prompt, raw_negative_prompt in zip(prompts, negative_prompts_in):
        positive_prompt, normalized_negative_prompt = process_prompt(
            raw_prompt,
            raw_negative_prompt,
            model,
        )
        positive_prompts.append(positive_prompt)
        negative_prompts.append(normalized_negative_prompt)

    return positive_prompts, negative_prompts


def _sequence_argument(values: list[str]) -> str | list[str]:
    return values[0] if len(values) == 1 else values


def _align_sequence_length(token_embeddings: Tensor, target_seq_len: int) -> Tensor:
    current_seq_len = int(token_embeddings.shape[0])
    if current_seq_len == target_seq_len:
        return token_embeddings
    if current_seq_len > target_seq_len:
        return token_embeddings[:target_seq_len]

    pad = token_embeddings.new_zeros(
        (target_seq_len - current_seq_len, token_embeddings.shape[-1])
    )
    return torch.cat((token_embeddings, pad), dim=0)


class QwenNode(BaseNode):
    output_key = "embeds"

    def __init__(self, inputs: QwenInputs):
        super().__init__(**inputs.model_dump())
        self.params = inputs
        self.node_type = "qwen"

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        qwen_model_path = self._resolve_qwen_model_path()
        projector_path = self._resolve_projector_path()
        positive_prompts, negative_prompts = _normalize_prompt_batch(
            self.params.prompt,
            self.params.negative_prompt,
            self.params.model,
        )
        output_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dtype = torch.float32
        projector, metadata = self._load_sdxl_projector(
            projector_path=projector_path,
            target_device=torch.device("cpu"),
        )
        if self.params.qwen_use_cached_negative_prompt_embeds:
            negative_prompt_embeds, negative_pooled_prompt_embeds = (
                self._load_negative_prompt_batch(
                    negative_prompts=negative_prompts,
                    output_device=output_device,
                    output_dtype=output_dtype,
                    expected_prompt_seq_len=int(metadata["prompt_seq_len"]),
                    expected_prompt_dim=int(metadata["prompt_dim"]),
                    expected_pooled_dim=int(metadata["pooled_dim"]),
                )
            )
        else:
            batch_size = len(negative_prompts)
            negative_prompt_embeds = torch.zeros(
                (
                    batch_size,
                    int(metadata["prompt_seq_len"]),
                    int(metadata["prompt_dim"]),
                ),
                device=output_device,
                dtype=output_dtype,
            )
            negative_pooled_prompt_embeds = torch.zeros(
                (batch_size, int(metadata["pooled_dim"])),
                device=output_device,
                dtype=output_dtype,
            )
        effective_n_batch = self._effective_qwen_n_batch()

        with LlamaEmbeddingSession(
            LlamaCppSessionConfig(
                model_path=qwen_model_path,
                n_ctx=self.params.qwen_n_ctx,
                n_batch=effective_n_batch,
                n_gpu_layers=self.params.qwen_n_gpu_layers,
                n_threads=self.params.qwen_n_threads,
                offload_kqv=self.params.qwen_offload_kqv,
                op_offload=self.params.qwen_op_offload,
                normalize_embeddings=self.params.qwen_normalize_embeddings,
                llama_cpp_lib_path=self.params.qwen_llama_cpp_lib_path,
                ignore_env_llama_cpp_lib_path=self.params.qwen_ignore_env_llama_cpp_lib_path,
                llama_cpp_extra_lib_paths=tuple(
                    self.params.qwen_llama_cpp_extra_lib_paths or ()
                ),
                llama_cpp_preload_libs=tuple(
                    self.params.qwen_llama_cpp_preload_libs or ()
                ),
            )
        ) as session:
            prompt_embeds, pooled_prompt_embeds = self._project_prompt_batch(
                session=session,
                prompts=positive_prompts,
                projector=projector,
                metadata=metadata,
                output_device=output_device,
                output_dtype=output_dtype,
            )

        return PromptEmbeds(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

    def _resolve_qwen_model_path(self) -> str:
        model_path = self.params.qwen_model_path or os.getenv("QWEN_GGUF_PATH")
        if not model_path:
            raise ValueError(
                "Qwen GGUF path is required. Set qwen_model_path or QWEN_GGUF_PATH."
            )

        resolved = Path(model_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Qwen GGUF file not found: {resolved}")
        return str(resolved)

    def _resolve_projector_path(self) -> str:
        projector_path = self.params.projector_path or os.getenv(
            "QWEN_2_SDXL_PROJECTOR_PATH"
        )
        if not projector_path:
            raise ValueError(
                "Qwen projector path is required. Set projector_path or QWEN_2_SDXL_PROJECTOR_PATH."
            )

        resolved = Path(projector_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Projector checkpoint not found: {resolved}")
        return str(resolved)

    def _project_prompt_batch(
        self,
        session: LlamaEmbeddingSession,
        prompts: list[str],
        projector: nn.Module,
        metadata: dict[str, int | str],
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        prompt_tensors: list[Tensor] = []
        pooled_tensors: list[Tensor] = []
        target_seq_len = int(metadata["prompt_seq_len"])
        target_hidden_size = int(metadata["prompt_dim"])
        target_pooled_size = int(metadata["pooled_dim"])

        for prompt in prompts:
            llama_embeddings = session.embed_text(prompt)
            if llama_embeddings.hidden_size != int(metadata["qwen_dim"]):
                raise RuntimeError(
                    "Qwen embedding size does not match the loaded projector input dimension."
                )

            pooled_source = torch.from_numpy(llama_embeddings.embedding)
            if pooled_source.ndim != 1:
                raise RuntimeError(
                    "Qwen runtime expects llama.cpp to return a mean-pooled embedding vector."
                )
            with torch.inference_mode():
                prompt_tensor, pooled_tensor = project_sdxl_qwen_embedding(
                    projector,
                    pooled_source.to(
                        device=next(projector.parameters()).device,
                        dtype=torch.float32,
                    )
                )

            if prompt_tensor.ndim != 2 or pooled_tensor.ndim != 1:
                raise RuntimeError(
                    "Qwen projector returned tensors with an unexpected rank for SDXL conditioning."
                )
            if int(prompt_tensor.shape[-1]) != target_hidden_size:
                raise RuntimeError(
                    "Qwen projector returned prompt embeddings with an unexpected hidden size."
                )
            if int(pooled_tensor.shape[-1]) != target_pooled_size:
                raise RuntimeError(
                    "Qwen projector returned pooled embeddings with an unexpected size."
                )

            prompt_tensor = _align_sequence_length(prompt_tensor, target_seq_len)
            prompt_tensors.append(prompt_tensor)
            pooled_tensors.append(pooled_tensor)

        return (
            torch.stack(prompt_tensors, dim=0).to(
                device=output_device,
                dtype=output_dtype,
            ),
            torch.stack(pooled_tensors, dim=0).to(
                device=output_device,
                dtype=output_dtype,
            ),
        )

    def _load_negative_prompt_batch(
        self,
        negative_prompts: list[str],
        output_device: torch.device,
        output_dtype: torch.dtype | None,
        expected_prompt_seq_len: int,
        expected_prompt_dim: int,
        expected_pooled_dim: int,
    ) -> tuple[Tensor, Tensor]:
        prompt_tensors: list[Tensor] = []
        pooled_tensors: list[Tensor] = []

        for negative_prompt in negative_prompts:
            prompt_tensor, pooled_tensor = load_sdxl_negative_prompt_embeddings(
                model=self.params.model,
                negative_prompt=negative_prompt,
                target_device=output_device,
                target_dtype=output_dtype,
                cache_dir=self.params.qwen_negative_prompt_cache_dir,
            )

            if tuple(prompt_tensor.shape) != (
                expected_prompt_seq_len,
                expected_prompt_dim,
            ):
                raise RuntimeError(
                    "Cached SDXL negative prompt embeddings do not match the active projector output shape."
                )
            if tuple(pooled_tensor.shape) != (expected_pooled_dim,):
                raise RuntimeError(
                    "Cached SDXL negative pooled embeddings do not match the active projector output size."
                )

            prompt_tensors.append(prompt_tensor)
            pooled_tensors.append(pooled_tensor)

        return torch.stack(prompt_tensors, dim=0), torch.stack(pooled_tensors, dim=0)

    def _load_sdxl_projector(
        self,
        projector_path: str,
        target_device: torch.device,
    ) -> tuple[nn.Module, dict[str, int | str]]:
        resolved = Path(projector_path).expanduser()
        if resolved.suffix.lower() != ".gguf":
            raise RuntimeError(
                "Qwen runtime requires a GGUF projector artifact for SDXL."
            )

        projector, metadata = load_projector_from_gguf(
            resolved,
            device=target_device,
            use_input_layernorm=self.params.use_input_layernorm,
        )
        if metadata.get("target_family") != "sdxl":
            raise RuntimeError(
                "Qwen runtime requires an SDXL projector artifact. The configured projector targets a different family."
            )
        return projector, metadata

    def _effective_qwen_n_batch(self) -> int:
        # llama.cpp embedding mode on the 27B ROCm build becomes memory-hungry at
        # larger batch sizes even when n_ctx is small. Keep the CPU-only path on
        # the validated 128-token batch unless the caller explicitly asks for less.
        if self.params.qwen_n_gpu_layers == 0:
            return min(self.params.qwen_n_batch, 128)
        return self.params.qwen_n_batch
