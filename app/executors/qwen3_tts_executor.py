from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import gc
import json
import threading
import time
from typing import Any

from app.config import settings
from app.executors.base import BaseExecutor


# ----------------------------------------------------------------------
# Optional dependencies
#
# Qwen3-TTS 관련 패키지가 설치되지 않아도 Worker Node 전체 import가
# 실패하지 않도록 optional import로 처리한다.
# ----------------------------------------------------------------------

try:
    import torch
except Exception:
    torch = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from qwen_tts import Qwen3TTSModel
except Exception:
    Qwen3TTSModel = None


# ----------------------------------------------------------------------
# Qwen3-TTS public model definitions
# ----------------------------------------------------------------------

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
        "display_name": "Qwen3-TTS 0.6B Base",
        "mode": "voice_clone",
        "size": "0.6B",
        "speakers": [],
        "features": {
            "voice_clone": True,
            "custom_voice": False,
            "voice_design": False,
            "instruction": False,
        },
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {
        "display_name": "Qwen3-TTS 1.7B Base",
        "mode": "voice_clone",
        "size": "1.7B",
        "speakers": [],
        "features": {
            "voice_clone": True,
            "custom_voice": False,
            "voice_design": False,
            "instruction": False,
        },
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "display_name": "Qwen3-TTS 0.6B Custom Voice",
        "mode": "custom_voice",
        "size": "0.6B",
        "speakers": CUSTOM_VOICE_SPEAKERS,
        "features": {
            "voice_clone": False,
            "custom_voice": True,
            "voice_design": False,
            "instruction": False,
        },
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "display_name": "Qwen3-TTS 1.7B Custom Voice",
        "mode": "custom_voice",
        "size": "1.7B",
        "speakers": CUSTOM_VOICE_SPEAKERS,
        "features": {
            "voice_clone": False,
            "custom_voice": True,
            "voice_design": False,
            "instruction": True,
        },
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "display_name": "Qwen3-TTS 1.7B Voice Design",
        "mode": "voice_design",
        "size": "1.7B",
        "speakers": [],
        "features": {
            "voice_clone": False,
            "custom_voice": False,
            "voice_design": True,
            "instruction": True,
        },
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )

    return bool(value)


def _cleanup_gpu_memory() -> None:
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _artifact_metadata(
    file_path: Path,
    artifact_path: str,
    content_type: str,
) -> dict:
    size_bytes = None

    if file_path.exists() and file_path.is_file():
        size_bytes = file_path.stat().st_size

    return {
        "path": artifact_path,
        "name": file_path.name,
        "content_type": content_type,
        "size_bytes": size_bytes,
    }


class Qwen3TTSExecutor(BaseExecutor):
    task_type = "audio_tts"
    label = "Qwen3 Text To Speech"
    description = (
        "Generate speech using Qwen3-TTS voice cloning, "
        "preset custom voices, or natural-language voice design."
    )
    version = "1.0.0"

    # voice_clone은 multipart가 필요하지만
    # custom_voice / voice_design은 JSON으로 제출할 수 있다.
    #
    # 기존 Central Server 호환용 기본값은 json으로 둔다.
    submit_mode = "json"

    def __init__(self):
        self.enabled = _as_bool(
            getattr(settings, "qwen3_tts_enabled", True),
            True,
        )

        self.auto_download = _as_bool(
            getattr(settings, "qwen3_tts_auto_download", True),
            True,
        )

        self.device = str(
            getattr(settings, "qwen3_tts_device", "cuda:0")
        ).strip()

        self.dtype_name = str(
            getattr(settings, "qwen3_tts_dtype", "bfloat16")
        ).strip().lower()

        self.attention = str(
            getattr(settings, "qwen3_tts_attention", "auto")
        ).strip().lower()

        try:
            max_loaded = int(
                getattr(settings, "qwen3_tts_max_loaded_models", 1)
            )
        except (TypeError, ValueError):
            max_loaded = 1

        self.max_loaded_models = max(1, max_loaded)

        self.models_root = (
            Path(settings.model_dir)
            / "qwen3_tts"
        )

        # model_id -> model object
        #
        # OrderedDict를 사용해서 가장 오래 사용하지 않은 모델부터
        # 간단하게 제거할 수 있도록 한다.
        self._loaded_models: OrderedDict[str, Any] = OrderedDict()

        # 모델 load / eviction 보호
        self._model_lock = threading.RLock()

        # 동일 모델이 동시에 여러 번 다운로드되는 것을 방지
        self._download_locks: dict[str, threading.Lock] = {}
        self._download_locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if not self.enabled:
            return False

        if Qwen3TTSModel is None:
            return False

        if torch is None:
            return False

        if sf is None:
            return False

        if self.device.lower().startswith("cuda"):
            if not torch.cuda.is_available():
                return False

        return True

    # ------------------------------------------------------------------
    # Self-describing manifest
    # ------------------------------------------------------------------

    def get_submit_modes(self) -> list[str]:
        return [
            "json",
            "multipart",
        ]

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "TTS Mode",
                    "enum": [
                        "voice_clone",
                        "custom_voice",
                        "voice_design",
                    ],
                    "description": (
                        "voice_clone uses reference audio, "
                        "custom_voice uses a built-in speaker, "
                        "voice_design creates a voice from an instruction."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "title": "Model",
                    "enum": list(SUPPORTED_MODELS.keys()),
                    "default": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                },
                "text": {
                    "type": "string",
                    "title": "Text",
                    "format": "textarea",
                },
                "language": {
                    "type": "string",
                    "title": "Language",
                    "enum": SUPPORTED_LANGUAGES,
                    "default": "Auto",
                },
                "reference_audio": {
                    "type": "file",
                    "title": "Reference Audio",
                    "description": (
                        "Required when mode=voice_clone. "
                        "Upload using multipart/form-data."
                    ),
                },
                "ref_text": {
                    "type": "string",
                    "title": "Reference Transcript",
                    "format": "textarea",
                    "description": (
                        "Transcript of the reference audio. "
                        "Required for normal voice cloning."
                    ),
                },
                "x_vector_only_mode": {
                    "type": "boolean",
                    "title": "X-Vector Only",
                    "default": False,
                    "description": (
                        "Use only the speaker embedding. "
                        "ref_text can be omitted, but quality may be lower."
                    ),
                },
                "speaker": {
                    "type": "string",
                    "title": "Preset Speaker",
                    "enum": CUSTOM_VOICE_SPEAKERS,
                    "description": "Required when mode=custom_voice.",
                },
                "instruct": {
                    "type": "string",
                    "title": "Voice Instruction",
                    "format": "textarea",
                    "description": (
                        "Optional for custom_voice and required "
                        "for voice_design."
                    ),
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["wav"],
                    "default": "wav",
                },

                # Initial allowlisted generation settings.
                "max_new_tokens": {
                    "type": "integer",
                    "title": "Max New Tokens",
                    "minimum": 1,
                    "default": 2048,
                },
                "temperature": {
                    "type": "number",
                    "title": "Temperature",
                    "minimum": 0,
                },
                "top_p": {
                    "type": "number",
                    "title": "Top P",
                    "minimum": 0,
                    "maximum": 1,
                },
                "top_k": {
                    "type": "integer",
                    "title": "Top K",
                    "minimum": 1,
                },
                "repetition_penalty": {
                    "type": "number",
                    "title": "Repetition Penalty",
                    "minimum": 0,
                },
            },
            "required": [
                "mode",
                "model_id",
                "text",
            ],
        }

    def get_file_inputs(self) -> list[dict]:
        return [
            {
                "name": "reference_audio",
                "label": "Reference Audio",
                "payload_key": "reference_audio",
                "required": False,
                "required_when": {
                    "mode": "voice_clone",
                },
                "accept": [
                    "audio/wav",
                    "audio/mpeg",
                    "audio/flac",
                    "audio/mp4",
                    "audio/x-m4a",
                ],
                "max_files": 1,
                "description": (
                    "Reference voice sample used by voice_clone mode."
                ),
            }
        ]

    def get_output_schema(self) -> dict:
        artifact_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "name": {
                    "type": "string",
                },
                "content_type": {
                    "type": "string",
                },
                "size_bytes": {
                    "type": ["integer", "null"],
                },
            },
        }

        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                },
                "backend": {
                    "type": "string",
                },
                "mode": {
                    "type": "string",
                },
                "model_id": {
                    "type": "string",
                },
                "language": {
                    "type": "string",
                },
                "speaker": {
                    "type": ["string", "null"],
                },
                "sample_rate": {
                    "type": "integer",
                },
                "duration_seconds": {
                    "type": "number",
                },
                "downloaded_model": {
                    "type": "boolean",
                },
                "reused_loaded_model": {
                    "type": "boolean",
                },
                "timings": {
                    "type": "object",
                },
                "output_files": {
                    "type": "array",
                    "items": artifact_schema,
                },
                "artifacts": {
                    "type": "array",
                    "items": artifact_schema,
                },
            },
        }

    def get_examples(self) -> list[dict]:
        return [
            {
                "id": "voice_clone",
                "title": "Voice Clone",
                "description": (
                    "Generate speech using an uploaded reference voice."
                ),
                "submit_mode": "multipart",
                "request": {
                    "task_type": self.task_type,
                    "priority": 5,
                    "payload": {
                        "mode": "voice_clone",
                        "model_id": (
                            "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                        ),
                        "text": (
                            "안녕하세요. SHS 음성 생성 테스트입니다."
                        ),
                        "language": "Korean",
                        "ref_text": (
                            "이 문장은 레퍼런스 음성에서 "
                            "실제로 말한 문장입니다."
                        ),
                        "x_vector_only_mode": False,
                        "output_format": "wav",
                        "file_inputs": {
                            "reference_audio": "reference_audio_file"
                        },
                    },
                },
                "files": {
                    "reference_audio_file": "reference.wav",
                },
            },
            {
                "id": "custom_voice",
                "title": "Custom Voice",
                "description": (
                    "Generate speech using a built-in Qwen speaker."
                ),
                "submit_mode": "json",
                "request": {
                    "task_type": self.task_type,
                    "priority": 5,
                    "payload": {
                        "mode": "custom_voice",
                        "model_id": (
                            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
                        ),
                        "text": "오늘 일정을 알려드리겠습니다.",
                        "language": "Korean",
                        "speaker": "Sohee",
                        "output_format": "wav",
                    },
                },
            },
            {
                "id": "voice_design",
                "title": "Voice Design",
                "description": (
                    "Generate speech from a natural-language "
                    "voice description."
                ),
                "submit_mode": "json",
                "request": {
                    "task_type": self.task_type,
                    "priority": 5,
                    "payload": {
                        "mode": "voice_design",
                        "model_id": (
                            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                        ),
                        "text": "탐사대는 새벽에 출발했습니다.",
                        "language": "Korean",
                        "instruct": (
                            "차분하고 낮은 여성 목소리로 "
                            "다큐멘터리 내레이션처럼 말한다."
                        ),
                        "output_format": "wav",
                    },
                },
            },
        ]

    def get_models(self) -> list[dict]:
        with self._model_lock:
            loaded_ids = set(self._loaded_models.keys())

        models = []

        for model_id, definition in SUPPORTED_MODELS.items():
            installed = self._is_model_installed(model_id)

            models.append(
                {
                    "id": model_id,
                    "display_name": definition["display_name"],
                    "mode": definition["mode"],
                    "size": definition["size"],
                    "supported": True,
                    "installed": installed,
                    "loaded": model_id in loaded_ids,
                    "downloadable": bool(
                        self.auto_download
                        and snapshot_download is not None
                    ),
                    "languages": SUPPORTED_LANGUAGES,
                    "speakers": definition["speakers"],
                    "features": definition["features"],
                }
            )

        return models

    # ------------------------------------------------------------------
    # Scheduler hints
    # ------------------------------------------------------------------

    def get_required_resources(self) -> dict:
        return {
            "gpu": self.device.lower() != "cpu",
            "min_vram_mb": 4096,
        }

    def estimate_requirements(self, payload: dict) -> dict:
        model_id = payload.get("model_id")

        definition = SUPPORTED_MODELS.get(model_id)

        if definition is None:
            return self.get_required_resources()

        # 초기 scheduler hint.
        # 실제 VRAM 사용량을 측정한 뒤 조정할 수 있다.
        if definition["size"] == "1.7B":
            min_vram_mb = 8192
        else:
            min_vram_mb = 4096

        return {
            "gpu": self.device.lower() != "cpu",
            "min_vram_mb": min_vram_mb,
            "model": model_id,
        }

    # ------------------------------------------------------------------
    # Model paths / installation
    # ------------------------------------------------------------------

    def _get_model_definition(
        self,
        model_id: str,
    ) -> dict[str, Any]:
        definition = SUPPORTED_MODELS.get(model_id)

        if definition is None:
            raise ValueError(
                f"Unsupported Qwen3-TTS model: {model_id}"
            )

        return definition

    def _model_local_dir(self, model_id: str) -> Path:
        # model_id는 allowlist를 통과한 값만 사용한다.
        model_name = model_id.split("/")[-1]

        return self.models_root / model_name

    def _is_model_installed(self, model_id: str) -> bool:
        if model_id not in SUPPORTED_MODELS:
            return False

        model_dir = self._model_local_dir(model_id)

        if not model_dir.exists() or not model_dir.is_dir():
            return False

        config_path = model_dir / "config.json"

        has_weights = (
            (model_dir / "model.safetensors").is_file()
            or
            (model_dir / "model.safetensors.index.json").is_file()
        )

        return config_path.is_file() and has_weights

    def _get_download_lock(
        self,
        model_id: str,
    ) -> threading.Lock:
        with self._download_locks_guard:
            lock = self._download_locks.get(model_id)

            if lock is None:
                lock = threading.Lock()
                self._download_locks[model_id] = lock

            return lock

    def _ensure_model_installed(
        self,
        model_id: str,
    ) -> tuple[Path, float, bool]:
        self._get_model_definition(model_id)

        model_dir = self._model_local_dir(model_id)

        if self._is_model_installed(model_id):
            return model_dir, 0.0, False

        if not self.auto_download:
            raise RuntimeError(
                f"Qwen3-TTS model is not installed and "
                f"automatic download is disabled: {model_id}"
            )

        if snapshot_download is None:
            raise RuntimeError(
                "huggingface_hub is required to download "
                "Qwen3-TTS models."
            )

        download_lock = self._get_download_lock(model_id)

        with download_lock:
            # 다른 thread가 기다리는 동안 이미 설치했을 수 있다.
            if self._is_model_installed(model_id):
                return model_dir, 0.0, False

            self.models_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            started = time.time()

            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_dir),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download Qwen3-TTS model "
                    f"'{model_id}': {exc}"
                ) from exc

            elapsed = time.time() - started

            if not self._is_model_installed(model_id):
                raise RuntimeError(
                    f"Qwen3-TTS model download completed but "
                    f"installation verification failed: {model_id}"
                )

            return model_dir, elapsed, True

    # ------------------------------------------------------------------
    # Model loading / cache
    # ------------------------------------------------------------------

    def _resolve_dtype(self):
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for Qwen3-TTS."
            )

        if self.dtype_name == "auto":
            return None

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

        dtype = mapping.get(self.dtype_name)

        if dtype is None:
            raise ValueError(
                "qwen3_tts_dtype must be one of: "
                "auto, float16, bfloat16, float32"
            )

        return dtype

    def _evict_oldest_model(self) -> None:
        if not self._loaded_models:
            return

        _old_model_id, old_model = self._loaded_models.popitem(
            last=False
        )

        del old_model

        _cleanup_gpu_memory()

    def _get_or_load_model(
        self,
        model_id: str,
    ) -> tuple[Any, float, float, bool, bool]:
        """
        Returns:
            model
            download_seconds
            load_seconds
            downloaded_model
            reused_loaded_model
        """

        if Qwen3TTSModel is None:
            raise RuntimeError(
                "qwen-tts package is not installed."
            )

        model_dir, download_seconds, downloaded = (
            self._ensure_model_installed(model_id)
        )

        with self._model_lock:
            existing = self._loaded_models.get(model_id)

            if existing is not None:
                self._loaded_models.move_to_end(model_id)

                return (
                    existing,
                    download_seconds,
                    0.0,
                    downloaded,
                    True,
                )

            while (
                len(self._loaded_models)
                >= self.max_loaded_models
            ):
                self._evict_oldest_model()

            dtype = self._resolve_dtype()

            load_kwargs: dict[str, Any] = {
                "device_map": self.device,
            }

            if dtype is not None:
                load_kwargs["dtype"] = dtype

            if self.attention != "auto":
                load_kwargs[
                    "attn_implementation"
                ] = self.attention

            started = time.time()

            try:
                model = Qwen3TTSModel.from_pretrained(
                    str(model_dir),
                    **load_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Qwen3-TTS model "
                    f"'{model_id}': {exc}"
                ) from exc

            load_seconds = time.time() - started

            self._loaded_models[model_id] = model
            self._loaded_models.move_to_end(model_id)

            return (
                model,
                download_seconds,
                load_seconds,
                downloaded,
                False,
            )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _normalize_language(self, language: Any) -> str:
        value = str(language or "Auto").strip()

        lookup = {
            item.lower(): item
            for item in SUPPORTED_LANGUAGES
        }

        normalized = lookup.get(value.lower())

        if normalized is None:
            raise ValueError(
                f"Unsupported language: {value}. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        return normalized

    def _normalize_speaker(self, speaker: Any) -> str:
        value = str(speaker or "").strip()

        if not value:
            raise ValueError(
                "speaker is required for custom_voice mode."
            )

        lookup = {
            item.lower(): item
            for item in CUSTOM_VOICE_SPEAKERS
        }

        normalized = lookup.get(value.lower())

        if normalized is None:
            raise ValueError(
                f"Unsupported Qwen3-TTS speaker: {value}. "
                f"Supported: {', '.join(CUSTOM_VOICE_SPEAKERS)}"
            )

        return normalized

    def _build_generation_kwargs(
        self,
        payload: dict,
    ) -> dict:
        """
        클라이언트 payload 전체를 Qwen generate에 그대로 넘기지 않고
        명시적으로 허용한 generation option만 전달한다.
        """

        allowed_fields = (
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        )

        kwargs = {}

        for field in allowed_fields:
            if payload.get(field) is not None:
                kwargs[field] = payload[field]

        return kwargs

    def _validate_request(
        self,
        payload: dict,
    ) -> dict:
        mode = str(
            payload.get("mode") or ""
        ).strip().lower()

        if mode not in (
            "voice_clone",
            "custom_voice",
            "voice_design",
        ):
            raise ValueError(
                "mode must be one of: "
                "voice_clone, custom_voice, voice_design."
            )

        model_id = str(
            payload.get("model_id") or ""
        ).strip()

        if not model_id:
            raise ValueError("model_id is required.")

        definition = self._get_model_definition(model_id)

        if definition["mode"] != mode:
            raise ValueError(
                f"Model '{model_id}' supports mode="
                f"'{definition['mode']}', not '{mode}'."
            )

        text = str(
            payload.get("text") or ""
        ).strip()

        if not text:
            raise ValueError("text is required.")

        language = self._normalize_language(
            payload.get("language", "Auto")
        )

        output_format = str(
            payload.get("output_format", "wav")
        ).strip().lower()

        if output_format != "wav":
            raise ValueError(
                "output_format currently supports only: wav"
            )

        validated = {
            "mode": mode,
            "model_id": model_id,
            "definition": definition,
            "text": text,
            "language": language,
            "output_format": output_format,
            "generation_kwargs": (
                self._build_generation_kwargs(payload)
            ),
        }

        if mode == "voice_clone":
            reference_audio = payload.get(
                "reference_audio"
            )

            if not reference_audio:
                raise ValueError(
                    "reference_audio is required "
                    "for voice_clone mode."
                )

            reference_path = Path(
                str(reference_audio)
            )

            if (
                not reference_path.exists()
                or not reference_path.is_file()
            ):
                raise FileNotFoundError(
                    f"Reference audio not found: "
                    f"{reference_path}"
                )

            x_vector_only = _as_bool(
                payload.get(
                    "x_vector_only_mode",
                    False,
                ),
                False,
            )

            ref_text = payload.get("ref_text")

            if ref_text is not None:
                ref_text = str(ref_text).strip()

            if not x_vector_only and not ref_text:
                raise ValueError(
                    "ref_text is required for voice_clone "
                    "unless x_vector_only_mode=true."
                )

            validated.update(
                {
                    "reference_audio": reference_path,
                    "ref_text": ref_text or None,
                    "x_vector_only_mode": x_vector_only,
                }
            )

        elif mode == "custom_voice":
            validated["speaker"] = (
                self._normalize_speaker(
                    payload.get("speaker")
                )
            )

            instruct = payload.get("instruct")

            if instruct is not None:
                instruct = str(instruct).strip()

            validated["instruct"] = (
                instruct or None
            )

        elif mode == "voice_design":
            instruct = str(
                payload.get("instruct") or ""
            ).strip()

            if not instruct:
                raise ValueError(
                    "instruct is required "
                    "for voice_design mode."
                )

            validated["instruct"] = instruct

        return validated

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _generate_voice_clone(
        self,
        model,
        request_data: dict,
    ):
        # create_voice_clone_prompt를 통해
        # x_vector_only_mode를 명시적으로 처리한다.
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(
                request_data["reference_audio"]
            ),
            ref_text=request_data["ref_text"],
            x_vector_only_mode=(
                request_data["x_vector_only_mode"]
            ),
        )

        return model.generate_voice_clone(
            text=request_data["text"],
            language=request_data["language"],
            voice_clone_prompt=prompt,
            **request_data["generation_kwargs"],
        )

    def _generate_custom_voice(
        self,
        model,
        request_data: dict,
    ):
        kwargs = {
            "text": request_data["text"],
            "language": request_data["language"],
            "speaker": request_data["speaker"],
            **request_data["generation_kwargs"],
        }

        instruct = request_data.get("instruct")

        if instruct:
            kwargs["instruct"] = instruct

        return model.generate_custom_voice(
            **kwargs
        )

    def _generate_voice_design(
        self,
        model,
        request_data: dict,
    ):
        return model.generate_voice_design(
            text=request_data["text"],
            language=request_data["language"],
            instruct=request_data["instruct"],
            **request_data["generation_kwargs"],
        )

    def run(
        self,
        payload: dict,
        work_dir: Path,
    ) -> dict:
        if not self.enabled:
            raise RuntimeError(
                "Qwen3-TTS executor is disabled."
            )

        if not self.is_available():
            raise RuntimeError(
                "Qwen3-TTS runtime is not available. "
                "Check qwen-tts, PyTorch, SoundFile, "
                "and configured device."
            )

        total_started = time.time()

        request_data = self._validate_request(
            payload
        )

        model_id = request_data["model_id"]
        mode = request_data["mode"]

        (
            model,
            download_seconds,
            load_seconds,
            downloaded_model,
            reused_loaded_model,
        ) = self._get_or_load_model(model_id)

        output_dir = work_dir / "output"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        speech_path = output_dir / "speech.wav"
        result_json_path = (
            output_dir / "result.json"
        )

        inference_started = time.time()

        try:
            if mode == "voice_clone":
                wavs, sample_rate = (
                    self._generate_voice_clone(
                        model,
                        request_data,
                    )
                )

            elif mode == "custom_voice":
                wavs, sample_rate = (
                    self._generate_custom_voice(
                        model,
                        request_data,
                    )
                )

            elif mode == "voice_design":
                wavs, sample_rate = (
                    self._generate_voice_design(
                        model,
                        request_data,
                    )
                )

            else:
                raise RuntimeError(
                    f"Unexpected TTS mode: {mode}"
                )

        except Exception as exc:
            raise RuntimeError(
                f"Qwen3-TTS inference failed "
                f"for mode='{mode}', "
                f"model='{model_id}': {exc}"
            ) from exc

        inference_seconds = (
            time.time() - inference_started
        )

        if not wavs:
            raise RuntimeError(
                "Qwen3-TTS returned no waveform."
            )

        waveform = wavs[0]

        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid sample rate returned "
                f"by Qwen3-TTS: {sample_rate}"
            ) from exc

        if sample_rate <= 0:
            raise RuntimeError(
                f"Invalid sample rate returned "
                f"by Qwen3-TTS: {sample_rate}"
            )

        try:
            sf.write(
                str(speech_path),
                waveform,
                sample_rate,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write generated WAV: {exc}"
            ) from exc

        try:
            duration_seconds = (
                float(len(waveform))
                / float(sample_rate)
            )
        except Exception:
            duration_seconds = 0.0

        output_files = [
            _artifact_metadata(
                speech_path,
                "output/speech.wav",
                "audio/wav",
            )
        ]

        result = {
            "task_type": self.task_type,
            "backend": "qwen3_tts",
            "mode": mode,
            "model_id": model_id,
            "language": request_data["language"],
            "speaker": request_data.get(
                "speaker"
            ),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "output_format": "wav",
            "downloaded_model": downloaded_model,
            "reused_loaded_model": (
                reused_loaded_model
            ),
            "timings": {
                "model_download_seconds": (
                    download_seconds
                ),
                "model_load_seconds": (
                    load_seconds
                ),
                "inference_seconds": (
                    inference_seconds
                ),
                "total_seconds": (
                    time.time() - total_started
                ),
            },
        }

        result_json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        output_files.append(
            _artifact_metadata(
                result_json_path,
                "output/result.json",
                "application/json",
            )
        )

        result["output_files"] = output_files
        result["artifacts"] = output_files

        return result