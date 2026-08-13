import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.auth import verify_internal_api_key
from app.core.task_manager import (
    cancel_task,
    create_and_queue_task,
    get_task,
    get_task_artifact_file_path,
    get_task_output_file_path,
    guess_content_type,
    list_task_output_files,
    update_task_priority,
)
from app.core.workdir import save_upload_file
from app.executors.registry import executor_registry


router = APIRouter(
    prefix="/api/worker",
    tags=["worker-tasks"],
)


class WorkerTaskPriorityUpdateRequest(BaseModel):
    priority: int = Field(..., description="New priority for a queued task")


def _is_json_request(content_type: str) -> bool:
    return "application/json" in content_type.lower()


def _is_multipart_request(content_type: str) -> bool:
    return "multipart/form-data" in content_type.lower()


def _load_task_submit_body(raw_data: Any) -> dict:
    """
    JSON body 또는 multipart metadata_json 값을 공통 task submit dict로 정규화한다.
    """

    if not isinstance(raw_data, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task submit body must be a JSON object.",
        )

    task_type = raw_data.get("task_type")

    if not task_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="task_type is required.",
        )

    payload = raw_data.get("payload") or {}
    metadata = raw_data.get("metadata") or {}
    priority = raw_data.get("priority", 5)

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload must be an object.",
        )

    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be an object.",
        )

    try:
        priority = int(priority)
    except (TypeError, ValueError):
        priority = 5

    return {
        "task_type": str(task_type),
        "priority": priority,
        "payload": payload,
        "metadata": metadata,
    }


async def _parse_json_task_request(request: Request) -> dict:
    """
    application/json 요청을 파싱한다.

    예:

    {
      "task_type": "llm_generate",
      "priority": 5,
      "payload": {
        "model": "llama3.2:latest",
        "prompt": "hello"
      },
      "metadata": {
        "central_task_id": "central_task_001"
      }
    }
    """

    try:
        raw_data = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {str(e)}",
        )

    return _load_task_submit_body(raw_data)


async def _save_multipart_files_to_payload(
    payload: dict,
    form,
) -> dict:
    """
    multipart form에 포함된 파일을 저장하고 payload에 실제 파일 경로를 넣는다.

    표준 형식:

    metadata_json:
    {
      "task_type": "whisper",
      "priority": 5,
      "payload": {
        "model": "base",
        "language": "auto",
        "output_format": "all",
        "file_inputs": {
          "input_file": "uploaded_audio"
        }
      },
      "metadata": {
        "central_task_id": "central_task_whisper_001"
      }
    }

    uploaded_audio:
    sample.mp3

    의미:
    - multipart form field "uploaded_audio" 파일을 저장한다.
    - 저장된 경로를 payload["input_file"]에 넣는다.

    주의:
    - file_map은 더 이상 지원하지 않는다.
    - 모든 파일 매핑은 file_inputs로 통일한다.
    """

    payload = dict(payload)

    file_inputs = payload.pop("file_inputs", None)

    if file_inputs is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Multipart task requires payload.file_inputs. "
                "Example: {'file_inputs': {'input_file': 'uploaded_audio'}}"
            ),
        )

    if not isinstance(file_inputs, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload.file_inputs must be an object.",
        )

    if not file_inputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload.file_inputs must not be empty.",
        )

    uploaded_files: dict[str, dict] = {}

    for payload_key, form_field_name in file_inputs.items():
        payload_key = str(payload_key)
        form_field_name = str(form_field_name)

        form_value = form.get(form_field_name)

        if form_value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing multipart file field: {form_field_name}",
            )

        if not hasattr(form_value, "read") or not hasattr(form_value, "filename"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Form field is not a file: {form_field_name}",
            )

        file_bytes = await form_value.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file is empty: {form_field_name}",
            )

        saved_path = save_upload_file(
            file_bytes,
            form_value.filename or form_field_name,
        )

        payload[payload_key] = str(saved_path)
        uploaded_files[payload_key] = {
            "form_field": form_field_name,
            "filename": form_value.filename,
            "saved_path": str(saved_path),
            "content_type": getattr(form_value, "content_type", None),
        }

    payload["_uploaded_files"] = uploaded_files

    return payload


async def _parse_multipart_task_request(request: Request) -> dict:
    """
    multipart/form-data 요청을 파싱한다.

    필수 form field:
    - metadata_json

    표준 예:

    metadata_json:
    {
      "task_type": "whisper",
      "priority": 5,
      "payload": {
        "model": "base",
        "language": "auto",
        "output_format": "all",
        "file_inputs": {
          "input_file": "uploaded_audio"
        }
      },
      "metadata": {
        "central_task_id": "central_task_whisper_001"
      }
    }

    uploaded_audio:
    sample.mp3
    """

    form = await request.form()

    metadata_json = form.get("metadata_json")

    if metadata_json is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="multipart request requires metadata_json field.",
        )

    if hasattr(metadata_json, "filename"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata_json must be a text field, not a file.",
        )

    try:
        raw_data = json.loads(str(metadata_json))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metadata_json: {str(e)}",
        )

    parsed = _load_task_submit_body(raw_data)

    parsed["payload"] = await _save_multipart_files_to_payload(
        payload=parsed["payload"],
        form=form,
    )

    return parsed


async def _parse_task_submit_request(request: Request) -> dict:
    """
    Content-Type에 따라 JSON 또는 multipart 요청을 파싱한다.
    """

    content_type = request.headers.get("content-type", "")

    if _is_json_request(content_type):
        return await _parse_json_task_request(request)

    if _is_multipart_request(content_type):
        return await _parse_multipart_task_request(request)

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            "Unsupported Content-Type. "
            "Use application/json or multipart/form-data."
        ),
    )

def _get_executor_submit_modes(executor) -> list[str]:
    """
    Executor가 지원하는 submit mode 목록을 반환한다.

    신규 executor:
        get_submit_modes() 사용

    기존 executor:
        submit_mode 단일 필드를 fallback으로 사용

    이렇게 해서 기존 Whisper/Ollama와 하위 호환을 유지한다.
    """

    if hasattr(executor, "get_submit_modes"):
        try:
            modes = executor.get_submit_modes()
        except Exception:
            modes = None

        if isinstance(modes, (list, tuple, set)):
            normalized = []

            for mode in modes:
                mode_text = str(mode).strip().lower()

                if mode_text in ("json", "multipart"):
                    if mode_text not in normalized:
                        normalized.append(mode_text)

            if normalized:
                return normalized

    legacy_mode = str(
        getattr(executor, "submit_mode", "json")
    ).strip().lower()

    if legacy_mode not in ("json", "multipart"):
        legacy_mode = "json"

    return [legacy_mode]


def _build_task_payload_for_response(task: dict) -> dict:
    """
    Central Server가 사용하기 좋은 task 응답 payload를 만든다.

    task_manager 내부 dict를 직접 수정하지 않기 위해 shallow copy를 사용한다.
    output_files는 조회 시점에 output 디렉토리를 스캔해서 붙인다.
    """

    task_payload = dict(task)

    output_files = list_task_output_files(task["id"])

    if output_files is None:
        output_files = []

    task_payload["output_files"] = output_files

    return task_payload


def _build_worker_task_response(task: dict) -> dict:
    """
    Central Server가 사용하기 좋은 응답 형태로 감싼다.
    """

    return {
        "ok": True,
        "task": _build_task_payload_for_response(task),
    }


@router.post("/tasks")
async def submit_worker_task(
    request: Request,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Generic Task Submit API.

    중앙 서버는 앞으로 task_type별 개별 API를 호출하지 않고,
    이 API 하나로 Worker Node에 작업을 제출한다.

    지원:
    - application/json
      - llm_generate
      - 텍스트 기반 이미지 생성
      - TTS 등 파일 입력이 필요 없는 작업

    - multipart/form-data
      - whisper
      - RVC
      - 이미지 편집/업스케일 등 파일 입력이 필요한 작업

    multipart 표준:
    - metadata_json 텍스트 필드
    - payload.file_inputs로 파일 매핑

    예:
    {
      "payload": {
        "file_inputs": {
          "input_file": "uploaded_audio"
        }
      }
    }
    """

    parsed = await _parse_task_submit_request(request)

    task_type = parsed["task_type"]
    priority = parsed["priority"]
    payload = parsed["payload"]
    metadata = parsed["metadata"]

    try:
        executor = executor_registry.get(task_type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not executor.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Executor is not available: {task_type}",
        )

    submit_modes = _get_executor_submit_modes(executor)
    content_type = request.headers.get("content-type", "")

    if _is_json_request(content_type):
        request_submit_mode = "json"
    elif _is_multipart_request(content_type):
        request_submit_mode = "multipart"
    else:
        request_submit_mode = None

    if request_submit_mode not in submit_modes:
        allowed_modes = ", ".join(submit_modes)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Executor '{task_type}' does not support this submit mode. "
                f"Allowed submit modes: {allowed_modes}."
            ),
        )

    try:
        task = create_and_queue_task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            metadata=metadata,
            source="central",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return _build_worker_task_response(task)


@router.get("/tasks/{task_id}")
async def worker_task_detail(
    task_id: str,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task Status API.

    Central Server는 작업 제출 후 이 API를 polling해서
    queued/running/completed/failed/cancelled 상태와 result를 동기화한다.
    """

    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    return _build_worker_task_response(task)


@router.get("/tasks/{task_id}/files")
async def worker_task_files(
    task_id: str,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task Output Files API.

    특정 task의 output 디렉토리에 있는 결과 파일 목록을 반환한다.
    """

    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    files = list_task_output_files(task_id)

    if files is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    return {
        "ok": True,
        "task_id": task_id,
        "files": files,
    }


@router.get("/tasks/{task_id}/artifacts")
async def download_worker_task_artifact(
    task_id: str,
    artifact_path: str = Query(
        ...,
        alias="path",
        description="Task work_dir relative artifact path. Only output/... paths are allowed.",
    ),
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task Artifact Download API.

    이 API는 path 기반 artifact 다운로드용이다.

    예:
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/result.txt
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/result.json
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/result.srt
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/audio.wav
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/video.mp4
    - GET /api/worker/tasks/{task_id}/artifacts?path=output/nested/archive.zip

    보안:
    - 외부 클라이언트가 직접 호출하지 않는다.
    - Central Server만 internal API key로 호출한다.
    - 실제 path 검증은 task_manager.get_task_artifact_file_path()에서 수행한다.
    - task work_dir/output 내부 파일만 다운로드 가능하다.
    """

    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    try:
        file_path = get_task_artifact_file_path(task_id, artifact_path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if file_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=guess_content_type(file_path),
    )


@router.get("/tasks/{task_id}/files/{filename}")
async def download_worker_task_file(
    task_id: str,
    filename: str,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task File Download API.

    보안:
    - filename에는 순수 파일명만 허용한다.
    - ../ 같은 path traversal은 task_manager에서 거부한다.
    - data/tasks/{task_id}/output 내부 파일만 다운로드 가능하다.

    참고:
    - 이 API는 기존 호환용이다.
    - 새 Central Server artifact proxy는 가능하면
      /api/worker/tasks/{task_id}/artifacts?path=output/result.txt 를 사용한다.
    """

    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    try:
        file_path = get_task_output_file_path(task_id, filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if file_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )

    files = list_task_output_files(task_id) or []
    file_info = next(
        (item for item in files if item["name"] == filename),
        None,
    )

    media_type = (
        file_info.get("content_type")
        if file_info is not None
        else None
    ) or guess_content_type(file_path)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )


@router.post("/tasks/{task_id}/cancel")
async def cancel_worker_task(
    task_id: str,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task Cancel API.

    MVP 규칙:
    - pending 상태: 취소 가능
    - queued 상태: 취소 가능
    - running 상태: 취소 불가
    - completed/failed/cancelled 상태: 취소 불가
    """

    try:
        task = cancel_task(task_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    return _build_worker_task_response(task)


@router.patch("/tasks/{task_id}/priority")
async def update_worker_task_priority(
    task_id: str,
    request: WorkerTaskPriorityUpdateRequest,
    _auth: bool = Depends(verify_internal_api_key),
):
    """
    Central Server 전용 Worker Task Priority Update API.

    queued 상태인 작업만 priority를 변경할 수 있다.
    running/completed/failed/cancelled 작업은 변경하지 않는다.
    """

    try:
        task = update_task_priority(task_id, request.priority)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )

    return _build_worker_task_response(task)