import asyncio
import heapq
import itertools
import mimetypes
import re
import threading
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from app.core.workdir import create_task_workdir
from app.executors.registry import executor_registry


TASKS: dict[str, dict] = {}

_TASK_QUEUE: list[tuple[int, int, int, str]] = []
_QUEUE_COUNTER = itertools.count()
_LOCK = threading.RLock()

_CURRENT_TASK_ID: str | None = None


TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_priority(priority: int | None) -> int:
    if priority is None:
        return 5

    try:
        return int(priority)
    except (TypeError, ValueError):
        return 5


def _normalize_metadata(metadata: dict | None) -> dict:
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        return {}

    return metadata


def _extract_central_task_id(metadata: dict) -> str | None:
    value = metadata.get("central_task_id")

    if value is None:
        return None

    return str(value)


def _extract_requested_by(metadata: dict) -> str | None:
    value = metadata.get("requested_by")

    if value is None:
        return None

    return str(value)


def _is_relative_to(path: Path, parent: Path) -> bool:
    """
    Python 버전 차이를 피하기 위한 안전한 relative_to 검사 helper.
    path가 parent 내부에 있으면 True를 반환한다.
    """

    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guess_content_type(file_path: Path) -> str:
    """
    파일 확장자를 기준으로 Content-Type을 추정한다.
    알 수 없으면 application/octet-stream으로 fallback한다.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".srt":
        return "application/x-subrip"

    content_type, _encoding = mimetypes.guess_type(str(file_path))

    return content_type or "application/octet-stream"


def create_task(
    task_type: str,
    payload: dict,
    priority: int = 5,
    metadata: dict | None = None,
    source: str = "local",
) -> dict:
    """
    Task 객체만 생성한다.
    실행은 하지 않는다.

    status 흐름:
    pending -> queued -> running -> completed
    pending -> queued -> running -> failed
    pending -> queued -> cancelled

    Args:
        task_type:
            executor task type.
            예: whisper, llm_generate

        payload:
            executor에 전달할 입력값.

        priority:
            높을수록 먼저 실행된다.
            같은 priority면 먼저 들어온 작업이 먼저 실행된다.

        metadata:
            Central Server 또는 호출자가 추가로 전달한 메타데이터.
            예:
            {
                "central_task_id": "central_task_001",
                "requested_by": "admin"
            }

        source:
            작업 출처.
            예:
            - "local"
            - "central"
            - "api"
    """

    # 알 수 없는 task_type이면 여기서 바로 에러 발생
    executor_registry.get(task_type)

    task_id = uuid4().hex
    work_dir = create_task_workdir(task_id)
    created_at = now_iso()

    metadata = _normalize_metadata(metadata)

    task = {
        "id": task_id,
        "task_type": task_type,
        "payload": payload,
        "priority": _normalize_priority(priority),
        "queue_sequence": None,
        "queue_version": 0,
        "status": "pending",
        "created_at": created_at,
        "queued_at": None,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "result": None,
        "work_dir": str(work_dir),

        # Central Server 연동용 필드
        "source": source,
        "metadata": metadata,
        "central_task_id": _extract_central_task_id(metadata),
        "requested_by": _extract_requested_by(metadata),
    }

    with _LOCK:
        TASKS[task_id] = task

    return task


def enqueue_task(task_id: str) -> dict:
    """
    Task를 priority queue에 넣는다.

    정렬 기준:
    1. priority 높은 작업 먼저
    2. priority가 같으면 먼저 들어온 작업 먼저

    heapq는 작은 값이 먼저 나오므로 -priority를 사용한다.
    """

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task["status"] not in ("pending", "queued"):
            raise ValueError(
                f"Only pending or queued tasks can be enqueued. "
                f"Current status: {task['status']}"
            )

        if task["queue_sequence"] is None:
            task["queue_sequence"] = next(_QUEUE_COUNTER)

        task["queue_version"] = int(task.get("queue_version", 0)) + 1
        task["status"] = "queued"

        if task["queued_at"] is None:
            task["queued_at"] = now_iso()

        heapq.heappush(
            _TASK_QUEUE,
            (
                -task["priority"],
                task["queue_sequence"],
                task["queue_version"],
                task_id,
            ),
        )

        return task


def create_and_queue_task(
    task_type: str,
    payload: dict,
    priority: int = 5,
    metadata: dict | None = None,
    source: str = "local",
) -> dict:
    """
    API 요청에서 주로 사용하는 함수.

    기존 create_and_run_task와 달리 즉시 실행하지 않고,
    작업을 생성한 뒤 queue에 넣고 바로 반환한다.
    """

    task = create_task(
        task_type=task_type,
        payload=payload,
        priority=priority,
        metadata=metadata,
        source=source,
    )

    return enqueue_task(task["id"])


def create_and_run_task(
    task_type: str,
    payload: dict,
    priority: int = 5,
    metadata: dict | None = None,
    source: str = "local",
) -> dict:
    """
    기존 호환용 함수.

    큐를 거치지 않고 즉시 실행한다.
    테스트용 또는 내부 디버그용으로 남겨둔다.
    일반 API에서는 create_and_queue_task를 사용하는 것을 권장한다.
    """

    task = create_task(
        task_type=task_type,
        payload=payload,
        priority=priority,
        metadata=metadata,
        source=source,
    )

    return run_task(task["id"])


def _pop_next_queued_task_id() -> str | None:
    """
    큐에서 다음 실행할 task_id를 꺼낸다.

    priority 변경으로 인해 오래된 heap entry가 남을 수 있으므로
    queue_version을 비교해서 stale entry를 무시한다.

    cancelled task는 status가 queued가 아니므로 자동으로 건너뛴다.
    """

    with _LOCK:
        while _TASK_QUEUE:
            _negative_priority, _sequence, queue_version, task_id = heapq.heappop(
                _TASK_QUEUE
            )

            task = TASKS.get(task_id)

            if task is None:
                continue

            if task["status"] != "queued":
                continue

            if task.get("queue_version") != queue_version:
                continue

            return task_id

    return None


def run_task(task_id: str) -> dict:
    """
    실제 executor를 실행한다.

    중요:
    - 이미 running 중인 작업을 중단하지 않는다.
    - queue worker가 이 함수를 하나씩 호출한다.
    """

    global _CURRENT_TASK_ID

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task["status"] in TERMINAL_STATUSES:
            return task

        if task["status"] == "running":
            return task

        task["status"] = "running"
        task["started_at"] = now_iso()
        task["completed_at"] = None
        task["error_message"] = None
        task["result"] = None

        _CURRENT_TASK_ID = task_id

        task_type = task["task_type"]
        payload = task["payload"]
        work_dir = Path(task["work_dir"])

    try:
        executor = executor_registry.get(task_type)
        result = executor.run(payload, work_dir)

        with _LOCK:
            task["status"] = "completed"
            task["completed_at"] = now_iso()
            task["result"] = result

    except Exception as e:
        with _LOCK:
            task["status"] = "failed"
            task["completed_at"] = now_iso()
            task["error_message"] = str(e)

    finally:
        with _LOCK:
            if _CURRENT_TASK_ID == task_id:
                _CURRENT_TASK_ID = None

    return task


async def queue_worker_loop(poll_interval: float = 0.2):
    """
    FastAPI lifespan에서 백그라운드로 실행되는 queue worker.

    동작:
    - 큐에서 작업 하나를 꺼냄
    - 해당 작업을 실행
    - 실행이 끝나면 다음 작업을 꺼냄

    즉, 현재 실행 중인 작업은 중간에 멈추지 않는다.
    """

    print("[SHS GPU NODE] Priority queue worker started.")

    try:
        while True:
            task_id = _pop_next_queued_task_id()

            if task_id is None:
                await asyncio.sleep(poll_interval)
                continue

            await asyncio.to_thread(run_task, task_id)

    except asyncio.CancelledError:
        print("[SHS GPU NODE] Priority queue worker stopped.")
        raise


def get_task(task_id: str) -> dict | None:
    with _LOCK:
        return TASKS.get(task_id)


def list_tasks() -> list[dict]:
    with _LOCK:
        return list(TASKS.values())


def get_running_tasks() -> list[dict]:
    """
    현재 running 상태인 작업 목록을 반환한다.

    현재 worker_count=1 구조에서는 보통 0개 또는 1개지만,
    나중에 멀티 worker 구조로 확장할 수 있으므로 list로 반환한다.
    """

    with _LOCK:
        return [
            task
            for task in TASKS.values()
            if task["status"] == "running"
        ]


def get_queue_status() -> dict:
    with _LOCK:
        queued_tasks = [
            {
                "id": task["id"],
                "task_type": task["task_type"],
                "priority": task["priority"],
                "queue_sequence": task["queue_sequence"],
                "status": task["status"],
                "created_at": task["created_at"],
                "queued_at": task["queued_at"],
                "source": task.get("source"),
                "central_task_id": task.get("central_task_id"),
                "requested_by": task.get("requested_by"),
            }
            for task in TASKS.values()
            if task["status"] == "queued"
        ]

        queued_tasks.sort(
            key=lambda item: (
                -item["priority"],
                item["queue_sequence"]
                if item["queue_sequence"] is not None
                else 999999999,
            )
        )

        current_task = TASKS.get(_CURRENT_TASK_ID) if _CURRENT_TASK_ID else None

        return {
            "current_task_id": _CURRENT_TASK_ID,
            "current_task": current_task,
            "queued_count": len(queued_tasks),
            "queued_task_ids": [task["id"] for task in queued_tasks],
            "queued_tasks": queued_tasks,
        }


def update_task_priority(task_id: str, priority: int) -> dict | None:
    """
    대기 중인 작업의 priority를 변경한다.

    running 중인 작업은 변경하지 않는다.
    completed / failed / cancelled 작업도 변경하지 않는다.
    """

    new_priority = _normalize_priority(priority)

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            return None

        if task["status"] != "queued":
            raise ValueError(
                f"Only queued tasks can have priority changed. "
                f"Current status: {task['status']}"
            )

        if task["queue_sequence"] is None:
            task["queue_sequence"] = next(_QUEUE_COUNTER)

        task["priority"] = new_priority
        task["queue_version"] = int(task.get("queue_version", 0)) + 1

        heapq.heappush(
            _TASK_QUEUE,
            (
                -task["priority"],
                task["queue_sequence"],
                task["queue_version"],
                task_id,
            ),
        )

        return task


def cancel_task(task_id: str) -> dict | None:
    """
    대기 중인 작업을 취소한다.

    MVP 규칙:
    - pending 상태: 취소 가능
    - queued 상태: 취소 가능
    - running 상태: 취소 불가
    - completed / failed / cancelled 상태: 취소 불가

    running 작업을 중단하는 기능은 나중에 executor별 cooperative cancel 구조로 확장한다.
    """

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            return None

        if task["status"] not in ("pending", "queued"):
            raise ValueError(
                f"Only pending or queued tasks can be cancelled. "
                f"Current status: {task['status']}"
            )

        task["status"] = "cancelled"
        task["completed_at"] = now_iso()
        task["error_message"] = "Task was cancelled before execution."

        # 오래된 heap entry를 무효화한다.
        task["queue_version"] = int(task.get("queue_version", 0)) + 1

        return task


def _get_work_dir_for_task(task: dict) -> Path:
    return Path(task["work_dir"])


def _get_output_dir_for_task(task: dict) -> Path:
    return _get_work_dir_for_task(task) / "output"


def _validate_artifact_relative_path(artifact_path: str) -> PurePosixPath:
    """
    task work_dir 기준 상대 artifact path를 검증한다.

    허용 예:
    - output/result.txt
    - output/result.json
    - output/result.srt
    - output/audio.wav
    - output/video.mp4
    - output/image.png
    - output/archive.zip

    차단 예:
    - ../secret
    - /etc/passwd
    - output/../../secret
    - C:\\Windows\\System32\\config
    - C:/Windows/System32/config
    - //server/share/file.txt
    - \\\\server\\share\\file.txt
    """

    if artifact_path is None:
        raise ValueError("artifact path is required.")

    path_text = str(artifact_path).strip()

    if not path_text:
        raise ValueError("artifact path is required.")

    if "\x00" in path_text:
        raise ValueError("artifact path contains invalid characters.")

    # Windows backslash/UNC 경로 차단
    if "\\" in path_text:
        raise ValueError("Backslash paths are not allowed.")

    # Windows drive path 차단: C:/..., C:\... 등
    if re.match(r"^[a-zA-Z]:", path_text):
        raise ValueError("Windows drive paths are not allowed.")

    # POSIX absolute 또는 UNC 스타일 //server/share 차단
    if path_text.startswith("/"):
        raise ValueError("Absolute artifact paths are not allowed.")

    if path_text.startswith("//"):
        raise ValueError("UNC paths are not allowed.")

    relative_path = PurePosixPath(path_text)

    if relative_path.is_absolute():
        raise ValueError("Absolute artifact paths are not allowed.")

    if ".." in relative_path.parts:
        raise ValueError("Path traversal is not allowed.")

    if not relative_path.parts:
        raise ValueError("artifact path is required.")

    # 현재 정책: task work_dir 전체가 아니라 output/ 아래 artifact만 허용한다.
    if relative_path.parts[0] != "output":
        raise ValueError("Only output/ artifact paths are allowed.")

    if len(relative_path.parts) < 2:
        raise ValueError("Artifact path must point to a file under output/.")

    return relative_path


def list_task_output_files(task_id: str) -> list[dict] | None:
    """
    특정 task의 output 디렉토리 안에 있는 파일 목록을 반환한다.

    Central Server의 files API에서 사용한다.

    기존에는 output 디렉토리 바로 아래 파일만 나열했지만,
    artifact API와 맞추기 위해 output 하위 파일을 재귀적으로 나열한다.
    """

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            return None

        output_dir = _get_output_dir_for_task(task)

    if not output_dir.exists() or not output_dir.is_dir():
        return []

    output_dir = output_dir.resolve()

    files = []

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue

        resolved_path = path.resolve()

        if not _is_relative_to(resolved_path, output_dir):
            continue

        relative_path = resolved_path.relative_to(output_dir)
        relative_path_text = relative_path.as_posix()

        files.append(
            {
                "name": path.name,
                "path": f"output/{relative_path_text}",
                "size_bytes": path.stat().st_size,
                "content_type": guess_content_type(path),
            }
        )

    files.sort(key=lambda item: item["path"])

    return files


def get_task_output_file_path(task_id: str, filename: str) -> Path | None:
    """
    특정 task의 output 파일 경로를 반환한다.

    기존 호환용 함수다.

    보안:
    - filename에는 순수 파일명만 허용한다.
    - ../ 같은 path traversal을 방지한다.
    - output 디렉토리 바로 아래 파일만 반환한다.

    path 기반 artifact 다운로드는 get_task_artifact_file_path()를 사용한다.
    """

    if not filename:
        raise ValueError("filename is required.")

    safe_name = Path(filename).name

    if safe_name != filename:
        raise ValueError("Invalid filename.")

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            return None

        output_dir = _get_output_dir_for_task(task)

    output_dir = output_dir.resolve()
    file_path = (output_dir / safe_name).resolve()

    if not _is_relative_to(file_path, output_dir):
        raise ValueError("Invalid file path.")

    if not file_path.exists() or not file_path.is_file():
        return None

    return file_path


def get_task_artifact_file_path(
    task_id: str,
    artifact_path: str,
) -> Path | None:
    """
    path 기반 artifact 다운로드용 파일 경로를 반환한다.

    Central Server가 호출할 Worker endpoint에서 사용한다.

    입력 예:
    - output/result.txt
    - output/result.json
    - output/result.srt
    - output/audio.wav
    - output/video.mp4
    - output/nested/archive.zip

    반환:
    - task가 없으면 None
    - 파일이 없으면 None
    - path가 위험하면 ValueError
    - 파일이 안전하게 존재하면 resolved Path
    """

    relative_path = _validate_artifact_relative_path(artifact_path)

    with _LOCK:
        task = TASKS.get(task_id)

        if task is None:
            return None

        task_work_dir = _get_work_dir_for_task(task)

    task_work_dir = task_work_dir.resolve()
    output_dir = (task_work_dir / "output").resolve()

    file_path = task_work_dir.joinpath(*relative_path.parts).resolve()

    if not _is_relative_to(file_path, task_work_dir):
        raise ValueError("Artifact path escapes task work directory.")

    if not _is_relative_to(file_path, output_dir):
        raise ValueError("Only output/ artifact paths are allowed.")

    if not file_path.exists() or not file_path.is_file():
        return None

    return file_path