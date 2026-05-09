from abc import ABC, abstractmethod
from pathlib import Path


class BaseExecutor(ABC):
    task_type = "base"
    label = "Base Executor"
    description = ""

    # submit_mode:
    # - "json": 일반 JSON payload로 제출 가능한 작업
    # - "multipart": 파일 업로드가 필요한 작업
    submit_mode = "json"

    def is_available(self) -> bool:
        """
        현재 executor가 실행 가능한 상태인지 반환한다.

        예:
        - Ollama 서버가 켜져 있으면 True
        - Whisper 패키지가 설치되어 있으면 True
        - 필요한 모델/외부 프로그램이 없으면 False
        """
        return True

    def estimate_requirements(self, payload: dict) -> dict:
        """
        특정 payload를 실행하기 위해 필요한 리소스를 추정한다.

        Central Server Scheduler가 나중에 참고할 수 있다.
        현재는 기본값만 제공한다.
        """
        return {
            "gpu": False,
            "min_vram_mb": 0,
        }

    def get_input_schema(self) -> dict:
        """
        Central Server가 동적 입력 폼을 만들 수 있도록
        executor의 입력 schema를 반환한다.

        각 executor는 필요하면 이 메서드를 override한다.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def get_output_schema(self) -> dict:
        """
        executor 결과 구조를 설명한다.

        각 executor는 필요하면 이 메서드를 override한다.
        """
        return {
            "type": "object",
            "properties": {},
        }

    def get_required_resources(self) -> dict:
        """
        executor 실행에 필요한 기본 리소스 요구사항을 반환한다.

        각 executor는 필요하면 이 메서드를 override한다.
        """
        return {
            "gpu": False,
            "min_vram_mb": 0,
        }

    def get_manifest(self) -> dict:
        """
        Central Server에 제공할 executor capability manifest를 반환한다.

        새로운 executor를 추가할 때 Central Server 코드를 수정하지 않기 위해,
        각 executor가 자신의 task_type, 입력 schema, 리소스 요구사항을
        스스로 설명하도록 한다.
        """
        return {
            "task_type": self.task_type,
            "label": self.label,
            "description": self.description,
            "available": self.is_available(),
            "enabled": True,
            "submit_mode": self.submit_mode,
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
            "required_resources": self.get_required_resources(),
        }

    @abstractmethod
    def run(self, payload: dict, work_dir: Path) -> dict:
        """
        실제 작업 실행 메서드.

        Args:
            payload:
                작업 입력값.
            work_dir:
                data/tasks/{task_id} 작업 디렉토리.

        Returns:
            Task result에 저장될 dict.
        """
        pass