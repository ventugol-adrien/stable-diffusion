from __future__ import annotations

import ctypes
import gc
import os
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import numpy as np


@dataclass(slots=True)
class LlamaCppSessionConfig:
    model_path: str
    n_ctx: int = 4096
    n_batch: int = 512
    pooling: Literal["mean", "none"] = "mean"
    normalize_embeddings: bool = True
    n_gpu_layers: int = 0
    n_threads: int | None = None
    offload_kqv: bool = True
    op_offload: bool | None = None
    llama_cpp_lib_path: str | None = None
    ignore_env_llama_cpp_lib_path: bool = False
    llama_cpp_extra_lib_paths: tuple[str, ...] = ()
    llama_cpp_preload_libs: tuple[str, ...] = ()
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False


@dataclass(slots=True)
class LlamaTokenEmbeddings:
    token_ids: list[int]
    embedding: np.ndarray
    token_embeddings: np.ndarray | None = None

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[-1])


class LlamaEmbeddingSession(AbstractContextManager["LlamaEmbeddingSession"]):
    def __init__(self, config: LlamaCppSessionConfig):
        self.config = config
        self._llama = None
        self._llama_cpp = None
        self._preloaded_shared_libraries: list[ctypes.CDLL] = []

    def __enter__(self) -> "LlamaEmbeddingSession":
        model_path = Path(self.config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Qwen GGUF file not found: {model_path}")

        self._preloaded_shared_libraries = _configure_llama_cpp_runtime(self.config)

        try:
            import llama_cpp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for Qwen embeddings. Install it and rebuild with the backend you want to use, or point LLAMA_CPP_LIB_PATH at a compatible libllama build."
            ) from exc

        pooling_type = _resolve_llama_pooling_type(llama_cpp, self.config.pooling)

        self._llama_cpp = llama_cpp
        self._llama = llama_cpp.Llama(
            model_path=str(model_path),
            embedding=True,
            pooling_type=pooling_type,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            offload_kqv=self.config.offload_kqv,
            op_offload=self.config.op_offload,
            use_mmap=self.config.use_mmap,
            use_mlock=self.config.use_mlock,
            verbose=self.config.verbose,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        llama = self._llama
        self._llama = None
        self._llama_cpp = None
        self._preloaded_shared_libraries = []
        if llama is not None:
            close = getattr(llama, "close", None)
            if callable(close):
                close()
            del llama
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def embed_text(self, text: str) -> LlamaTokenEmbeddings:
        if self._llama is None:
            raise RuntimeError("LlamaEmbeddingSession must be entered before use.")
        if not text.strip():
            raise ValueError("Qwen prompts must be non-empty strings.")

        reset = getattr(self._llama, "reset", None)
        if callable(reset):
            reset()

        token_ids = list(
            self._llama.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        )
        if not token_ids:
            raise RuntimeError("llama.cpp returned no tokens for the prompt.")

        raw_embeddings = self._llama.embed(text, normalize=False, truncate=False)
        if self.config.pooling == "mean":
            embedding = _coerce_embedding_vector(raw_embeddings)
            if self.config.normalize_embeddings:
                embedding = _normalize_embedding_vector(embedding)
            return LlamaTokenEmbeddings(token_ids=token_ids, embedding=embedding)

        token_embeddings = _coerce_token_embedding_matrix(raw_embeddings)
        token_count = min(len(token_ids), int(token_embeddings.shape[0]))
        if token_count <= 0:
            raise RuntimeError("llama.cpp returned no token embeddings for the prompt.")

        token_embeddings = token_embeddings[:token_count]
        embedding = np.asarray(token_embeddings.mean(axis=0), dtype=np.float32)
        if self.config.normalize_embeddings:
            embedding = _normalize_embedding_vector(embedding)

        return LlamaTokenEmbeddings(
            token_ids=token_ids[:token_count],
            embedding=embedding,
            token_embeddings=token_embeddings,
        )


def _resolve_llama_pooling_type(llama_cpp, pooling: Literal["mean", "none"]):
    attr_name = {
        "mean": "LLAMA_POOLING_TYPE_MEAN",
        "none": "LLAMA_POOLING_TYPE_NONE",
    }[pooling]
    pooling_type = getattr(llama_cpp, attr_name, None)
    if pooling_type is None:
        raise RuntimeError(
            f"Installed llama-cpp-python does not expose {attr_name}. Upgrade to a newer release."
        )
    return pooling_type


def _coerce_embedding_vector(raw_embeddings) -> np.ndarray:
    embedding = np.asarray(raw_embeddings, dtype=np.float32)

    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]
    elif embedding.ndim == 3 and embedding.shape[0] == 1 and embedding.shape[1] == 1:
        embedding = embedding[0, 0]

    if embedding.ndim != 1:
        raise RuntimeError(
            f"Unexpected llama.cpp embedding rank: {embedding.ndim}. Expected a pooled embedding vector."
        )

    return embedding


def _normalize_embedding_vector(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0 or not np.isfinite(norm):
        return embedding
    return np.asarray(embedding / norm, dtype=np.float32)


def _coerce_token_embedding_matrix(raw_embeddings) -> np.ndarray:
    token_embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if token_embeddings.ndim == 3 and token_embeddings.shape[0] == 1:
        token_embeddings = token_embeddings[0]

    if token_embeddings.ndim == 1:
        raise RuntimeError(
            "llama.cpp returned a pooled embedding instead of token-level embeddings. Use LLAMA_POOLING_TYPE_NONE for generation-style models."
        )

    if token_embeddings.ndim != 2:
        raise RuntimeError(
            f"Unexpected llama.cpp embedding rank: {token_embeddings.ndim}. Expected token-level embeddings."
        )

    return token_embeddings


def _configure_llama_cpp_runtime(
    config: LlamaCppSessionConfig,
) -> list[ctypes.CDLL]:
    requested_lib_path = config.llama_cpp_lib_path
    if requested_lib_path is None and not config.ignore_env_llama_cpp_lib_path:
        requested_lib_path = os.getenv("LLAMA_CPP_LIB_PATH")

    if requested_lib_path:
        resolved_lib_dir = _resolve_llama_cpp_lib_path(requested_lib_path)
        _assert_llama_cpp_not_already_loaded_from_elsewhere(resolved_lib_dir)
        os.environ["LLAMA_CPP_LIB_PATH"] = str(resolved_lib_dir)
    elif config.ignore_env_llama_cpp_lib_path:
        os.environ.pop("LLAMA_CPP_LIB_PATH", None)

    extra_paths = _resolve_existing_directories(
        [
            *config.llama_cpp_extra_lib_paths,
            *_split_env_paths("LLAMA_CPP_EXTRA_LIB_PATHS"),
        ],
        label="llama.cpp extra library path",
    )
    if extra_paths:
        _prepend_dynamic_library_search_paths(extra_paths)

    preload_paths = _resolve_existing_files(
        [
            *config.llama_cpp_preload_libs,
            *_split_env_paths("LLAMA_CPP_PRELOAD_LIBS"),
        ],
        label="llama.cpp preload library",
    )
    return [_load_shared_library(path) for path in preload_paths]


def _resolve_llama_cpp_lib_path(raw_path: str) -> Path:
    resolved = Path(raw_path).expanduser()
    if resolved.is_dir():
        return resolved.resolve()

    if resolved.is_file():
        valid_names = {
            "libllama.so",
            "libllama.dylib",
            "llama.dll",
            "libllama.dll",
        }
        if resolved.name not in valid_names:
            raise ValueError(
                "llama_cpp_lib_path must point to a directory containing libllama or to the libllama shared library itself, not the llama.cpp CLI binary."
            )
        return resolved.parent.resolve()

    raise FileNotFoundError(f"llama.cpp shared library path not found: {resolved}")


def _assert_llama_cpp_not_already_loaded_from_elsewhere(
    requested_lib_dir: Path,
) -> None:
    loaded_module = sys.modules.get("llama_cpp.llama_cpp")
    if loaded_module is None:
        return

    loaded_base_path = getattr(loaded_module, "_base_path", None)
    if loaded_base_path is None:
        return

    resolved_loaded_base_path = Path(str(loaded_base_path)).expanduser().resolve()
    if resolved_loaded_base_path != requested_lib_dir:
        raise RuntimeError(
            "llama_cpp is already loaded from "
            f"{resolved_loaded_base_path}. Restart the process to switch to {requested_lib_dir}."
        )


def _split_env_paths(variable_name: str) -> list[str]:
    raw_value = os.getenv(variable_name)
    if not raw_value:
        return []
    return [item for item in raw_value.split(os.pathsep) if item]


def _resolve_existing_directories(raw_paths: list[str], label: str) -> list[Path]:
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        resolved = Path(raw_path).expanduser()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        resolved_paths.append(resolved.resolve())
    return resolved_paths


def _resolve_existing_files(raw_paths: list[str], label: str) -> list[Path]:
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        resolved = Path(raw_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        resolved_paths.append(resolved.resolve())
    return resolved_paths


def _prepend_dynamic_library_search_paths(paths: list[Path]) -> None:
    if sys.platform == "win32":
        env_var = "PATH"
    elif sys.platform == "darwin":
        env_var = "DYLD_LIBRARY_PATH"
    else:
        env_var = "LD_LIBRARY_PATH"

    current_entries = [
        entry for entry in os.getenv(env_var, "").split(os.pathsep) if entry
    ]
    merged_entries: list[str] = []
    for path in paths:
        path_str = str(path)
        if path_str not in merged_entries:
            merged_entries.append(path_str)
    for entry in current_entries:
        if entry not in merged_entries:
            merged_entries.append(entry)

    if merged_entries:
        os.environ[env_var] = os.pathsep.join(merged_entries)

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        for path in paths:
            os.add_dll_directory(str(path))


def _load_shared_library(path: Path) -> ctypes.CDLL:
    cdll_kwargs = {}
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", None)
    if rtld_global is not None:
        cdll_kwargs["mode"] = rtld_global

    try:
        return ctypes.CDLL(str(path), **cdll_kwargs)
    except OSError as exc:
        raise RuntimeError(f"Failed to preload shared library {path}: {exc}") from exc


async def pause_llm():
    llama_url = os.getenv("LLAMA_URL")
    if not llama_url:
        print("⚠️ LLAMA_URL not set; skipping LLM pause.")
        return
    print("Attempting to pause LLM inference via API...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{llama_url}/pause", timeout=50)
            if response.status_code == 200:
                print("LLM inference paused successfully.")
            else:
                print(
                    f"Failed to pause LLM inference. Status code: {response.status_code}"
                )

    except Exception as e:
        print(f"Error while trying to pause LLM inference: {e}")
