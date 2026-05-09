from app.executors.whisper_executor import WhisperExecutor
from app.executors.llm_ollama_executor import LLMOllamaExecutor


class ExecutorRegistry:
    def __init__(self):
        self.executors = {}
        self.register(WhisperExecutor())
        self.register(LLMOllamaExecutor())

    def register(self, executor):
        self.executors[executor.task_type] = executor

    def get(self, task_type: str):
        executor = self.executors.get(task_type)

        if executor is None:
            raise ValueError(f"Unknown task_type: {task_type}")

        return executor

    def list_executors(self) -> list[dict]:
        """
        로컬 테스트용 간단 executor 목록.

        기존 /api/local/executors 또는 간단한 상태 표시에서 사용한다.
        """
        return [
            {
                "task_type": executor.task_type,
                "label": executor.label,
                "available": executor.is_available(),
            }
            for executor in self.executors.values()
        ]

    def list_manifests(self) -> list[dict]:
        """
        Central Server에 제공할 executor capability manifest 목록.

        각 executor가 get_manifest()를 구현하고 있으면 그 값을 사용한다.
        만약 아직 get_manifest()가 없는 오래된 executor가 있으면
        기본 fallback manifest를 반환해서 서버가 죽지 않도록 한다.
        """
        manifests = []

        for executor in self.executors.values():
            if hasattr(executor, "get_manifest"):
                manifests.append(executor.get_manifest())
            else:
                manifests.append(
                    {
                        "task_type": executor.task_type,
                        "label": getattr(executor, "label", executor.task_type),
                        "description": getattr(executor, "description", ""),
                        "available": executor.is_available(),
                        "enabled": True,
                        "submit_mode": getattr(executor, "submit_mode", "json"),
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {},
                        },
                        "required_resources": {
                            "gpu": False,
                            "min_vram_mb": 0,
                        },
                    }
                )

        return manifests

    def get_task_types(self) -> list[str]:
        return list(self.executors.keys())

    def has_task_type(self, task_type: str) -> bool:
        return task_type in self.executors


executor_registry = ExecutorRegistry()