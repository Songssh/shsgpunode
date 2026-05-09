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