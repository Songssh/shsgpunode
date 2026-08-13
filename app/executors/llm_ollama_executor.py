from pathlib import Path
import gc
import json
import time

import httpx

try:
    import torch
except ImportError:
    torch = None

from app.config import settings
from app.executors.base import BaseExecutor


def _get_bool_setting(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    return bool(value)


class LLMOllamaExecutor(BaseExecutor):
    task_type = "llm_generate"
    label = "Ollama LLM Generate"
    description = "Generate text using local Ollama models."
    submit_mode = "json"

    def __init__(self):
        self.enabled = _get_bool_setting("ollama_enabled", True)
        self.base_url = getattr(
            settings,
            "ollama_base_url",
            "http://localhost:11434",
        ).rstrip("/")
        self.timeout = getattr(settings, "ollama_timeout_seconds", 300)
        self.default_model = getattr(settings, "default_llm_model", None)

    def is_available(self) -> bool:
        """
        Ollama 서버가 켜져 있고 /api/tags에 응답하면 사용 가능하다고 판단한다.
        """
        if not self.enabled:
            return False

        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_input_schema(self) -> dict:
        """
        Central Server가 llm_generate 작업 입력 폼을 자동으로 만들 수 있도록
        입력 schema를 제공한다.
        """
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "title": "Model",
                    "default": self.default_model or "llama3.2:latest",
                    "description": "Ollama model name.",
                },
                "prompt": {
                    "type": "string",
                    "title": "Prompt",
                    "format": "textarea",
                    "description": "User prompt to send to the model.",
                },
                "system": {
                    "type": "string",
                    "title": "System Prompt",
                    "format": "textarea",
                    "default": "",
                    "description": "Optional system instruction.",
                },
                "temperature": {
                    "type": "number",
                    "title": "Temperature",
                    "default": 0.7,
                    "minimum": 0,
                    "maximum": 2,
                },
                "top_p": {
                    "type": "number",
                    "title": "Top P",
                    "default": 0.9,
                    "minimum": 0,
                    "maximum": 1,
                },
                "max_tokens": {
                    "type": "integer",
                    "title": "Max Tokens",
                    "default": 512,
                    "minimum": 1,
                },
                "format": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["text", "json"],
                    "default": "text",
                    "description": "Use json to request JSON-mode output from Ollama.",
                },
                "keep_alive": {
                    "type": "string",
                    "title": "Keep Alive",
                    "default": "5m",
                    "description": "How long Ollama should keep the model loaded.",
                },
            },
            "required": ["model", "prompt"],
        }

    def get_output_schema(self) -> dict:
        """
        llm_generate 작업 결과 구조를 설명한다.
        실제 결과는 task.result에 저장된다.
        """
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                },
                "backend": {
                    "type": "string",
                },
                "model": {
                    "type": "string",
                },
                "format": {
                    "type": "string",
                },
                "elapsed_seconds": {
                    "type": "number",
                },
                "response": {
                    "type": "string",
                },
                "ollama": {
                    "type": "object",
                },
                "files": {
                    "type": "object",
                    "properties": {
                        "request_json": {
                            "type": "string",
                        },
                        "response_txt": {
                            "type": "string",
                        },
                        "result_json": {
                            "type": "string",
                        },
                    },
                },
            },
        }

    def get_required_resources(self) -> dict:
        """
        LLM 추론은 일반적으로 GPU 사용을 전제로 한다.
        실제 VRAM 요구량은 모델마다 다르지만, 중앙 서버 MVP에서는
        기본값으로 4096MB를 사용한다.
        """
        return {
            "gpu": True,
            "min_vram_mb": 4096,
        }

    def run(self, payload: dict, work_dir: Path) -> dict:
        if not self.enabled:
            raise RuntimeError("Ollama executor is disabled.")

        model = payload.get("model") or self.default_model
        prompt = payload.get("prompt")
        system = payload.get("system")
        output_format = payload.get("format", "text")
        keep_alive = payload.get("keep_alive", "5m")

        if not model:
            raise ValueError("model is required.")

        if not prompt:
            raise ValueError("prompt is required.")

        if output_format not in ("text", "json"):
            raise ValueError("format must be one of: text, json.")

        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
        }

        if system:
            request_body["system"] = system

        if output_format == "json":
            request_body["format"] = "json"

        options = {}

        if payload.get("temperature") is not None:
            options["temperature"] = payload["temperature"]

        if payload.get("top_p") is not None:
            options["top_p"] = payload["top_p"]

        if payload.get("max_tokens") is not None:
            options["num_predict"] = payload["max_tokens"]

        if options:
            request_body["options"] = options

        request_json_path = output_dir / "request.json"
        response_txt_path = output_dir / "response.txt"
        result_json_path = output_dir / "result.json"

        request_json_path.write_text(
            json.dumps(request_body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        start_time = time.time()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=request_body,
                )
                response.raise_for_status()
                ollama_result = response.json()

            elapsed_seconds = time.time() - start_time
            response_text = ollama_result.get("response", "")

            response_txt_path.write_text(response_text, encoding="utf-8")

            final_result = {
                "task_type": self.task_type,
                "backend": "ollama",
                "model": model,
                "format": output_format,
                "elapsed_seconds": elapsed_seconds,
                "response": response_text,
                "ollama": ollama_result,
                "files": {
                    "request_json": str(request_json_path),
                    "response_txt": str(response_txt_path),
                    "result_json": str(result_json_path),
                },
            }

            result_json_path.write_text(
                json.dumps(final_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return final_result

        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama server: {self.base_url}"
            ) from exc

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text
            raise RuntimeError(
                f"Ollama API returned HTTP {exc.response.status_code}: {error_text}"
            ) from exc

        finally:
            gc.collect()

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def get_examples(self) -> list[dict]:
        return [
            {
                "id": "generate",
                "title": "Generate Text",
                "description": "Generate a text response using a locally installed Ollama model.",
                "submit_mode": "json",
                "request": {
                    "task_type": self.task_type,
                    "priority": 5,
                    "payload": {
                        "model": self.default_model or "llama3.2:latest",
                        "prompt": "안녕하세요. 간단하게 자기소개를 해주세요.",
                        "system": "You are a helpful assistant.",
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512,
                        "format": "text",
                        "keep_alive": "5m",
                    },
                },
            }
        ]


    def get_models(self) -> list[dict]:
        if not self.enabled:
            return []

        try:
            with httpx.Client(timeout=2.0) as client:
                tags_response = client.get(f"{self.base_url}/api/tags")
                tags_response.raise_for_status()
                tags_data = tags_response.json()

                loaded_names = set()

                try:
                    ps_response = client.get(f"{self.base_url}/api/ps")
                    ps_response.raise_for_status()
                    ps_data = ps_response.json()

                    for item in ps_data.get("models", []):
                        model_name = (
                            item.get("name")
                            or item.get("model")
                        )

                        if model_name:
                            loaded_names.add(str(model_name))

                except Exception:
                    # /api/ps 조회 실패가 전체 모델 목록 capability를
                    # 없애지는 않도록 한다.
                    pass

        except Exception:
            return []

        models = []

        for item in tags_data.get("models", []):
            model_name = (
                item.get("name")
                or item.get("model")
            )

            if not model_name:
                continue

            model_name = str(model_name)

            details = item.get("details") or {}

            models.append(
                {
                    "id": model_name,
                    "display_name": model_name,
                    "supported": True,
                    "installed": True,
                    "loaded": model_name in loaded_names,

                    # 현재 LLMOllamaExecutor에는 /api/pull을 호출하는
                    # 자동 다운로드 로직이 없으므로 false.
                    "downloadable": False,

                    "size_bytes": item.get("size"),
                    "details": {
                        "format": details.get("format"),
                        "family": details.get("family"),
                        "parameter_size": details.get("parameter_size"),
                        "quantization_level": details.get("quantization_level"),
                    },
                    "features": {
                        "text_generation": True,
                    },
                }
            )

        return models