import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.config import settings
from app.hardware.system_info import get_system_info
from app.hardware.gpu_info import get_gpu_info

from app.api.local_tasks import router as local_tasks_router
from app.api.worker_tasks import router as worker_tasks_router
from app.core.capability_manifest import build_capability_manifest
from app.core.central_client import central_heartbeat_loop
from app.core.task_manager import queue_worker_loop


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_setting(name: str, default):
    return getattr(settings, name, default)


def build_node_info_payload() -> dict:
    """
    /api/node/info에서 반환할 최신 Node 정보를 생성한다.

    조회 시점 기준으로 system/gpu/capabilities를 다시 생성한다.
    """

    protocol_version = get_setting("worker_protocol_version", "1.0")
    worker_version = get_setting("worker_version", "0.1.0")
    node_public_base_url = get_setting(
        "node_public_base_url",
        f"http://127.0.0.1:{settings.port}",
    )

    return {
        "node_id": settings.node_id,
        "node_name": settings.node_name,
        "host": settings.host,
        "port": settings.port,
        "base_url": node_public_base_url,
        "version": worker_version,
        "protocol_version": protocol_version,
        "status": "online",
        "timestamp": now_iso(),
        "system": get_system_info(),
        "gpus": get_gpu_info(),
        "capabilities": build_capability_manifest(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    node_info_payload = build_node_info_payload()
    app.state.node_info = node_info_payload

    print("[SHS GPU NODE] Node started.")
    print(f"[SHS GPU NODE] Node ID: {settings.node_id}")
    print(f"[SHS GPU NODE] Node name: {settings.node_name}")
    print(f"[SHS GPU NODE] Public base URL: {node_info_payload['base_url']}")
    print(f"[SHS GPU NODE] GPUs found: {len(node_info_payload['gpus'])}")

    background_tasks: list[asyncio.Task] = []

    queue_worker_task = asyncio.create_task(queue_worker_loop())
    app.state.queue_worker_task = queue_worker_task
    background_tasks.append(queue_worker_task)

    if settings.central_enabled:
        central_task = asyncio.create_task(central_heartbeat_loop())
        app.state.central_heartbeat_task = central_task
        background_tasks.append(central_task)
        print("[SHS GPU NODE] Central heartbeat enabled.")
    else:
        app.state.central_heartbeat_task = None
        print("[SHS GPU NODE] Central server integration disabled.")

    try:
        yield

    finally:
        print("[SHS GPU NODE] Node shutting down.")

        for task in background_tasks:
            task.cancel()

        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)


@app.get("/")
def index():
    return {
        "message": "SHS GPU Worker Node is running",
        "node_id": settings.node_id,
        "node_name": settings.node_name,
    }


# Local test/debug API
app.include_router(local_tasks_router)

# Central Server facing generic Worker API
app.include_router(worker_tasks_router)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "node_id": settings.node_id,
        "node_name": settings.node_name,
    }


@app.get("/api/node/info")
def node_info():
    return build_node_info_payload()


@app.get("/api/gpu")
def gpu_status():
    return {
        "gpus": get_gpu_info(),
    }


@app.get("/api/capabilities")
def capabilities():
    return build_capability_manifest()