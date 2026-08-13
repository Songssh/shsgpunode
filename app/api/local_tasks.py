from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.core.workdir import save_upload_file
from app.core.task_manager import (
    create_and_queue_task,
    get_queue_status,
    get_task,
    list_tasks,
    update_task_priority,
)
from app.executors.registry import executor_registry


router = APIRouter(prefix="/api/local", tags=["local-tasks"])


@router.get("/executors")
def executors():
    return {
        "executors": executor_registry.list_executors(),
    }


@router.get("/queue")
def queue_status():
    return get_queue_status()


@router.get("/tasks")
def tasks():
    return {
        "tasks": list_tasks(),
    }


@router.get("/tasks/{task_id}")
def task_detail(task_id: str):
    task = get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return task


@router.patch("/tasks/{task_id}/priority")
def change_task_priority(
    task_id: str,
    priority: int = Form(...),
):
    try:
        task = update_task_priority(task_id, priority)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return task


@router.post("/tasks/whisper")
async def submit_whisper_task(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("auto"),
    output_format: str = Form("all"),
    priority: int = Form(5),
):
    file_bytes = await file.read()
    input_path = save_upload_file(file_bytes, file.filename or "audio")

    payload = {
        "input_file": str(input_path),
        "model": model,
        "language": language,
        "output_format": output_format,
    }

    task = create_and_queue_task(
        task_type="whisper",
        payload=payload,
        priority=priority,
    )

    return task


@router.post("/tasks/llm/generate")
async def submit_llm_generate_task(
    model: str = Form(...),
    prompt: str = Form(...),
    system: str | None = Form(None),
    temperature: float | None = Form(None),
    top_p: float | None = Form(None),
    max_tokens: int | None = Form(None),
    format: str = Form("text"),
    keep_alive: str = Form("5m"),
    priority: int = Form(5),
):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "format": format,
        "keep_alive": keep_alive,
    }

    payload = {key: value for key, value in payload.items() if value is not None}

    task = create_and_queue_task(
        task_type="llm_generate",
        payload=payload,
        priority=priority,
    )

    return task

@router.post("/tasks/tts")
async def submit_tts_task(
    mode: str = Form(...),
    model_id: str = Form(...),
    text: str = Form(...),
    language: str = Form("Auto"),

    # voice_clone
    reference_audio: UploadFile | None = File(None),
    ref_text: str | None = Form(None),
    x_vector_only_mode: bool = Form(False),

    # custom_voice / voice_design
    speaker: str | None = Form(None),
    instruct: str | None = Form(None),

    # generation
    max_new_tokens: int | None = Form(None),
    temperature: float | None = Form(None),
    top_p: float | None = Form(None),
    top_k: int | None = Form(None),
    repetition_penalty: float | None = Form(None),

    priority: int = Form(5),
):
    """
    Qwen3-TTS 로컬 테스트용 API.

    Swagger UI에서 다음 세 모드를 직접 시험할 수 있다.

    - voice_clone
    - custom_voice
    - voice_design

    모든 실제 검증과 모델 다운로드/로드/추론은
    Qwen3TTSExecutor가 담당한다.
    """

    normalized_mode = mode.strip().lower()

    if normalized_mode not in (
        "voice_clone",
        "custom_voice",
        "voice_design",
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "mode must be one of: "
                "voice_clone, custom_voice, voice_design"
            ),
        )

    payload = {
        "mode": normalized_mode,
        "model_id": model_id.strip(),
        "text": text,
        "language": language,
        "output_format": "wav",
        "x_vector_only_mode": x_vector_only_mode,
        "ref_text": ref_text,
        "speaker": speaker,
        "instruct": instruct,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }

    # None인 optional 필드는 executor로 보내지 않는다.
    payload = {
        key: value
        for key, value in payload.items()
        if value is not None
    }

    if normalized_mode == "voice_clone":
        if reference_audio is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "reference_audio is required "
                    "for voice_clone mode."
                ),
            )

        file_bytes = await reference_audio.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="reference_audio is empty.",
            )

        reference_path = save_upload_file(
            file_bytes,
            reference_audio.filename or "reference.wav",
        )

        payload["reference_audio"] = str(reference_path)

    task = create_and_queue_task(
        task_type="audio_tts",
        payload=payload,
        priority=priority,
    )

    return task