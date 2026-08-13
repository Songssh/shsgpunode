from pathlib import Path
import json
import gc

import whisper
import torch

from app.config import settings
from app.executors.base import BaseExecutor


def seconds_to_srt_time(seconds: float) -> str:
    milliseconds = int((seconds % 1) * 1000)
    total_seconds = int(seconds)

    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600

    return f"{h:02}:{m:02}:{s:02},{milliseconds:03}"


def write_srt(segments: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = seconds_to_srt_time(float(segment["start"]))
            end = seconds_to_srt_time(float(segment["end"]))
            text = segment.get("text", "").strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def cleanup_gpu_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_artifact_metadata(
    file_path: Path,
    artifact_path: str,
    content_type: str,
) -> dict:
    """
    Central Server / SHSAITools가 다운로드 버튼을 만들 때 사용할
    artifact metadata를 생성한다.

    artifact_path는 반드시 task work_dir 기준 상대 경로여야 한다.

    예:
    - output/result.txt
    - output/result.json
    - output/result.srt
    """

    size_bytes = None

    if file_path.exists() and file_path.is_file():
        size_bytes = file_path.stat().st_size

    return {
        "path": artifact_path,
        "name": file_path.name,
        "content_type": content_type,
        "size_bytes": size_bytes,
    }


class WhisperExecutor(BaseExecutor):
    task_type = "whisper"
    label = "Whisper Speech To Text"
    description = "Transcribe audio or video files using Whisper."
    submit_mode = "multipart"

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except Exception:
            return False

    def get_input_schema(self) -> dict:
        """
        Central Server가 Whisper 작업 입력 폼을 자동으로 만들 수 있도록
        입력 schema를 제공한다.

        주의:
        - Central Server에는 audio라는 file input으로 노출한다.
        - 실제 executor.run()에는 input_file 경로가 payload로 전달된다.
        - /api/worker/tasks multipart 처리 단계에서 audio 파일을 저장한 뒤
          payload["input_file"]로 변환할 예정이다.
        """
        return {
            "type": "object",
            "properties": {
                "audio": {
                    "type": "file",
                    "title": "Audio or Video File",
                    "accept": "audio/*,video/*",
                    "description": "Audio or video file to transcribe.",
                },
                "model": {
                    "type": "string",
                    "title": "Whisper Model",
                    "enum": [
                        "tiny",
                        "base",
                        "small",
                        "medium",
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    "default": "base",
                },
                "language": {
                    "type": "string",
                    "title": "Language",
                    "default": "auto",
                    "description": "Use auto for automatic language detection, or use language codes such as ko/en.",
                },
                "output_format": {
                    "type": "string",
                    "title": "Output Format",
                    "enum": ["text", "json", "srt", "all"],
                    "default": "all",
                },
            },
            "required": ["audio"],
        }

    def get_output_schema(self) -> dict:
        """
        Whisper 작업 결과 구조를 설명한다.

        실제 결과는 task.result에 저장된다.

        files:
        - 기존 호환용 실제 파일 시스템 경로

        output_files / artifacts:
        - Central Server와 SHSAITools가 다운로드에 사용할 상대 path metadata
        - 예: output/result.txt
        """
        artifact_item_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Task work_dir relative artifact path. Example: output/result.txt",
                },
                "name": {
                    "type": "string",
                    "description": "File name for download.",
                },
                "content_type": {
                    "type": "string",
                    "description": "MIME type.",
                },
                "size_bytes": {
                    "type": ["integer", "null"],
                    "description": "File size in bytes.",
                },
            },
        }

        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                },
                "language": {
                    "type": "string",
                },
                "model": {
                    "type": "string",
                },
                "requested_language": {
                    "type": "string",
                },
                "output_format": {
                    "type": "string",
                },
                "model_dir": {
                    "type": "string",
                },
                "files": {
                    "type": "object",
                    "description": "Backward-compatible absolute file paths on the Worker filesystem.",
                    "properties": {
                        "text": {
                            "type": "string",
                        },
                        "json": {
                            "type": "string",
                        },
                        "srt": {
                            "type": "string",
                        },
                    },
                },
                "output_files": {
                    "type": "array",
                    "description": "Downloadable artifact metadata using task-relative output/... paths.",
                    "items": artifact_item_schema,
                },
                "artifacts": {
                    "type": "array",
                    "description": "Alias of output_files for clients that use artifact terminology.",
                    "items": artifact_item_schema,
                },
            },
        }

    def get_required_resources(self) -> dict:
        """
        Whisper는 GPU 사용을 권장한다.
        모델 크기에 따라 요구 VRAM이 달라지지만 manifest에는 기본값을 제공한다.
        """
        return {
            "gpu": True,
            "min_vram_mb": 2048,
        }

    def estimate_requirements(self, payload: dict) -> dict:
        """
        payload에 들어온 Whisper model 이름을 기준으로 대략적인 VRAM 요구량을 추정한다.
        Central Server Scheduler가 나중에 참고할 수 있다.
        """
        model_name = payload.get("model", "base")

        rough_vram = {
            "tiny": 1024,
            "base": 1024,
            "small": 2048,
            "medium": 5120,
            "large": 10240,
            "large-v2": 10240,
            "large-v3": 10240,
        }

        min_vram_mb = rough_vram.get(model_name, 2048)

        return {
            "gpu": True,
            "min_vram_mb": min_vram_mb,
            "model": model_name,
        }

    def run(self, payload: dict, work_dir: Path) -> dict:
        input_file = payload.get("input_file")
        model_name = payload.get("model", "base")
        language = payload.get("language", "auto")
        output_format = payload.get("output_format", "all")

        if not input_file:
            raise ValueError("input_file is required.")

        if output_format not in ("text", "json", "srt", "all"):
            raise ValueError("output_format must be one of: text, json, srt, all.")

        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(settings.model_dir) / "whisper"
        model_dir.mkdir(parents=True, exist_ok=True)

        model = None

        try:
            model = whisper.load_model(
                model_name,
                download_root=str(model_dir),
            )

            options = {}

            if language and language != "auto":
                options["language"] = language

            result = model.transcribe(str(input_path), **options)

            saved_files = {}
            output_files = []

            if output_format in ("text", "all"):
                result_text_path = output_dir / "result.txt"

                with result_text_path.open("w", encoding="utf-8") as f:
                    f.write(result.get("text", "").strip())

                saved_files["text"] = str(result_text_path)
                output_files.append(
                    build_artifact_metadata(
                        file_path=result_text_path,
                        artifact_path="output/result.txt",
                        content_type="text/plain",
                    )
                )

            if output_format in ("json", "all"):
                result_json_path = output_dir / "result.json"

                with result_json_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                saved_files["json"] = str(result_json_path)
                output_files.append(
                    build_artifact_metadata(
                        file_path=result_json_path,
                        artifact_path="output/result.json",
                        content_type="application/json",
                    )
                )

            if output_format in ("srt", "all"):
                result_srt_path = output_dir / "result.srt"

                write_srt(result.get("segments", []), result_srt_path)

                saved_files["srt"] = str(result_srt_path)
                output_files.append(
                    build_artifact_metadata(
                        file_path=result_srt_path,
                        artifact_path="output/result.srt",
                        content_type="application/x-subrip",
                    )
                )

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language"),
                "model": model_name,
                "requested_language": language,
                "output_format": output_format,
                "model_dir": str(model_dir),

                # 기존 호환용: Worker 내부 실제 파일 경로
                "files": saved_files,

                # Central Server / SHSAITools용: 다운로드 가능한 상대 artifact path
                "output_files": output_files,
                "artifacts": output_files,
            }

        finally:
            if model is not None:
                del model

            cleanup_gpu_memory()
            
    def get_file_inputs(self) -> list[dict]:
        return [
            {
                "name": "uploaded_audio",
                "label": "Audio or Video File",
                "payload_key": "input_file",
                "required": True,
                "accept": [
                    "audio/*",
                    "video/*",
                ],
                "max_files": 1,
                "description": "Audio or video file to transcribe.",
            }
        ]


    def get_examples(self) -> list[dict]:
        return [
            {
                "id": "transcribe",
                "title": "Transcribe Audio",
                "description": "Transcribe an uploaded audio or video file with Whisper.",
                "submit_mode": "multipart",
                "request": {
                    "task_type": self.task_type,
                    "priority": 5,
                    "payload": {
                        "model": "base",
                        "language": "auto",
                        "output_format": "all",
                        "file_inputs": {
                            "input_file": "uploaded_audio"
                        },
                    },
                },
                "files": {
                    "uploaded_audio": "sample.wav",
                },
            }
        ]


    def get_models(self) -> list[dict]:
        model_dir = Path(settings.model_dir) / "whisper"

        model_names = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]

        models = []

        for model_name in model_names:
            model_path = model_dir / f"{model_name}.pt"

            models.append(
                {
                    "id": model_name,
                    "display_name": f"Whisper {model_name}",
                    "supported": True,
                    "installed": model_path.is_file(),

                    # 현재 WhisperExecutor는 매 작업마다 모델을 load 후
                    # 작업 종료 시 GPU에서 제거하므로 상주 loaded 상태를
                    # 유지하지 않는다.
                    "loaded": False,

                    # whisper.load_model(..., download_root=...)가
                    # 필요할 경우 모델을 자동 다운로드한다.
                    "downloadable": True,

                    "features": {
                        "transcription": True,
                        "language_detection": True,
                        "srt_output": True,
                    },
                }
            )

        return models