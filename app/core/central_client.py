import asyncio
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.config import settings
from app.core.capability_manifest import build_capability_manifest
from app.core.task_manager import get_queue_status, get_running_tasks
from app.hardware.gpu_info import get_gpu_info
from app.hardware.system_info import get_system_info


def now_iso() -> str:
    tz = ZoneInfo(settings.app_timezone)
    return datetime.now(tz).isoformat()


def get_setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def get_node_id() -> str:
    return str(get_setting("node_id", "auto"))


def get_node_name() -> str:
    return str(get_setting("node_name", "shs-gpu-node-01"))


def get_worker_version() -> str:
    return str(get_setting("worker_version", "0.1.0"))


def get_worker_protocol_version() -> str:
    return str(get_setting("worker_protocol_version", "1.0"))


def get_node_public_base_url() -> str:
    default_url = f"http://127.0.0.1:{get_setting('port', 8100)}"
    return str(get_setting("node_public_base_url", default_url)).rstrip("/")


def get_central_base_url() -> str:
    """
    신규 설정 central_base_url을 우선 사용한다.
    기존 호환을 위해 central_server_url도 fallback으로 유지한다.
    """

    central_base_url = get_setting("central_base_url", None)

    if central_base_url:
        return str(central_base_url).rstrip("/")

    return str(get_setting("central_server_url", "http://central-gpu.shs")).rstrip("/")


def get_register_path() -> str:
    return str(get_setting("central_register_path", "/api/nodes/register"))


def get_heartbeat_path() -> str:
    return str(
        get_setting(
            "central_heartbeat_path",
            "/api/nodes/{node_id}/heartbeat",
        )
    )


def get_heartbeat_interval_seconds() -> int:
    value = get_setting("central_heartbeat_interval_seconds", 10)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 10


def get_register_retry_seconds() -> int:
    value = get_setting("central_register_retry_seconds", 10)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 10


def build_url(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/")
    path = path.lstrip("/")
    return f"{base_url}/{path}"


def render_path(path: str, **kwargs) -> str:
    """
    /api/nodes/{node_id}/heartbeat 같은 path template을 실제 path로 변환한다.
    """

    return path.format(**kwargs)


def get_internal_auth_headers() -> dict[str, str]:
    """
    Central Server에 register/heartbeat를 보낼 때 사용할 내부 인증 헤더.

    기본:
    x-shs-internal-api-key: change-this-secret
    """

    header_name = str(
        get_setting(
            "shs_internal_api_key_header",
            "x-shs-internal-api-key",
        )
    )

    api_key = str(
        get_setting(
            "shs_internal_api_key",
            get_setting("central_api_key", "change-this-secret"),
        )
    )

    return {
        header_name: api_key,
    }


def build_queue_payload() -> dict:
    """
    현재 Worker Node의 queue 상태를 heartbeat/register payload에 넣기 좋게 정리한다.
    """

    queue_status = get_queue_status()

    return {
        "enabled": True,
        "priority": True,
        "preemptive": False,
        "worker_count": 1,
        "current_task_id": queue_status.get("current_task_id"),
        "queued_count": queue_status.get("queued_count", 0),
        "queued_task_ids": queue_status.get("queued_task_ids", []),
        "queued_tasks": queue_status.get("queued_tasks", []),
    }


def build_running_tasks_payload() -> list[dict]:
    """
    heartbeat에 넣을 running task 요약 목록.
    """

    running_tasks = []

    for task in get_running_tasks():
        running_tasks.append(
            {
                "id": task.get("id"),
                "task_type": task.get("task_type"),
                "status": task.get("status"),
                "priority": task.get("priority"),
                "started_at": task.get("started_at"),
                "source": task.get("source"),
                "central_task_id": task.get("central_task_id"),
                "requested_by": task.get("requested_by"),
            }
        )

    return running_tasks


def build_register_payload() -> dict:
    """
    Worker Node가 Central Server에 등록할 때 보내는 payload.

    register에는 비교적 전체 정보를 넣는다.
    """

    return {
        "node_id": get_node_id(),
        "node_name": get_node_name(),
        "base_url": get_node_public_base_url(),
        "host": get_setting("host", "0.0.0.0"),
        "port": get_setting("port", 8100),
        "version": get_worker_version(),
        "protocol_version": get_worker_protocol_version(),
        "status": "online",
        "registered_at": now_iso(),
        "system": get_system_info(),
        "gpus": get_gpu_info(),
        "queue": build_queue_payload(),
        "running_tasks": build_running_tasks_payload(),
        "capabilities": build_capability_manifest(),
    }


def build_heartbeat_payload() -> dict:
    """
    Worker Node가 Central Server에 주기적으로 보내는 heartbeat payload.

    heartbeat에는 매번 바뀌는 상태를 중심으로 넣는다.
    초기 구현에서는 capabilities도 포함한다.
    """

    return {
        "node_id": get_node_id(),
        "node_name": get_node_name(),
        "base_url": get_node_public_base_url(),
        "status": "online",
        "version": get_worker_version(),
        "protocol_version": get_worker_protocol_version(),
        "timestamp": now_iso(),
        "system": get_system_info(),
        "gpus": get_gpu_info(),
        "queue": build_queue_payload(),
        "running_tasks": build_running_tasks_payload(),
        "capabilities": build_capability_manifest(),
    }


async def register_to_central() -> dict | None:
    """
    Central Server에 Worker Node를 등록한다.

    실패해도 예외를 밖으로 던지지 않는다.
    Worker Node는 Central Server 없이도 단독 실행 가능해야 한다.
    """

    if not settings.central_enabled:
        print("[SHS GPU NODE] Central register skipped: central_enabled=false")
        return None

    base_url = get_central_base_url()
    register_url = build_url(base_url, get_register_path())
    headers = get_internal_auth_headers()
    payload = build_register_payload()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                register_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        try:
            result = response.json()
        except ValueError:
            result = {
                "ok": True,
                "raw": response.text,
            }

        print("[SHS GPU NODE] Registered to central server.")
        print(f"[SHS GPU NODE] Central register URL: {register_url}")

        return result

    except Exception as e:
        print("[SHS GPU NODE] Failed to register to central server.")
        print(f"[SHS GPU NODE] Central register URL: {register_url}")
        print(str(e))
        return None


async def send_heartbeat() -> dict | None:
    """
    Central Server에 heartbeat를 보낸다.

    실패해도 예외를 밖으로 던지지 않는다.
    """

    if not settings.central_enabled:
        return None

    node_id = get_node_id()
    base_url = get_central_base_url()
    heartbeat_path = render_path(
        get_heartbeat_path(),
        node_id=node_id,
    )
    heartbeat_url = build_url(base_url, heartbeat_path)
    headers = get_internal_auth_headers()
    payload = build_heartbeat_payload()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                heartbeat_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        try:
            result = response.json()
        except ValueError:
            result = {
                "ok": True,
                "raw": response.text,
            }

        return result

    except Exception as e:
        print("[SHS GPU NODE] Failed to send heartbeat to central server.")
        print(f"[SHS GPU NODE] Central heartbeat URL: {heartbeat_url}")
        print(str(e))
        return None


async def central_heartbeat_loop():
    """
    Central Server heartbeat loop.

    동작:
    1. 시작 시 register 시도
    2. 주기적으로 heartbeat 전송
    3. heartbeat 응답에서 re_register=true가 오면 register 재시도
    4. 실패해도 Worker Node는 계속 실행
    """

    if not settings.central_enabled:
        print("[SHS GPU NODE] Central heartbeat loop disabled.")
        return

    print("[SHS GPU NODE] Central heartbeat loop started.")

    register_retry_seconds = get_register_retry_seconds()
    heartbeat_interval_seconds = get_heartbeat_interval_seconds()

    register_result = await register_to_central()

    if register_result is None:
        print(
            "[SHS GPU NODE] Initial central registration failed. "
            "Worker will keep running and retry through heartbeat loop."
        )
        await asyncio.sleep(register_retry_seconds)

    try:
        while True:
            result = await send_heartbeat()

            if result is not None:
                re_register = bool(result.get("re_register", False))

                next_interval = result.get(
                    "heartbeat_interval_seconds",
                    heartbeat_interval_seconds,
                )

                try:
                    heartbeat_interval_seconds = int(next_interval)
                except (TypeError, ValueError):
                    heartbeat_interval_seconds = get_heartbeat_interval_seconds()

                if re_register:
                    print("[SHS GPU NODE] Central requested re-register.")
                    await register_to_central()

            await asyncio.sleep(heartbeat_interval_seconds)

    except asyncio.CancelledError:
        print("[SHS GPU NODE] Central heartbeat loop stopped.")
        raise