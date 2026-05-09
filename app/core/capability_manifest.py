from app.config import settings
from app.executors.registry import executor_registry


def get_worker_protocol_version() -> str:
    return getattr(settings, "worker_protocol_version", "1.0")


def build_queue_capability() -> dict:
    """
    Worker Node의 queue 처리 정책을 설명한다.

    현재 Worker Node는:
    - priority queue 사용
    - 비선점형 실행
    - worker_count = 1
    """

    return {
        "enabled": True,
        "priority": True,
        "preemptive": False,
        "worker_count": 1,
    }


def build_capability_manifest() -> dict:
    """
    Central Server와 외부 관리 도구에 제공할 Worker capability manifest를 생성한다.

    Central Server는 이 manifest를 보고 다음을 판단한다.

    1. 이 Worker가 어떤 task_type을 지원하는가?
    2. 각 task_type은 어떤 입력 schema를 요구하는가?
    3. 파일 업로드가 필요한 작업인가?
    4. GPU가 필요한 작업인가?
    5. 중앙 서버가 어떤 endpoint로 작업을 제출해야 하는가?

    중요한 점:
    - Central Server는 llm_generate, whisper 같은 task별 API를 하드코딩하지 않는다.
    - Worker가 제공하는 manifest를 읽고 generic task API로 작업을 제출한다.
    """

    return {
        "protocol_version": get_worker_protocol_version(),
        "task_types": executor_registry.get_task_types(),
        "task_submit_endpoint": "/api/worker/tasks",
        "task_status_endpoint": "/api/worker/tasks/{task_id}",
        "task_files_endpoint": "/api/worker/tasks/{task_id}/files",
        "executors": executor_registry.list_manifests(),
        "queue": build_queue_capability(),
    }


def build_register_capabilities() -> dict:
    """
    Central Server register payload에 넣을 capabilities.

    현재는 build_capability_manifest()와 동일하게 사용한다.
    나중에 register 전용 필드가 필요해지면 여기에서 확장하면 된다.
    """

    return build_capability_manifest()


def build_heartbeat_capabilities() -> dict:
    """
    Central Server heartbeat payload에 넣을 capabilities.

    초기 구현에서는 heartbeat에도 전체 capabilities를 포함한다.
    이렇게 하면 Worker Node에 새 executor가 추가되어 재시작되었을 때
    Central Server가 heartbeat만 보고도 새 기능을 인식할 수 있다.
    """

    return build_capability_manifest()