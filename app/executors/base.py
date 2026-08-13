from abc import ABC, abstractmethod
from pathlib import Path


class BaseExecutor(ABC):
    # ------------------------------------------------------------------
    # Executor identity
    # ------------------------------------------------------------------

    # 실제 작업 종류.
    #
    # 예:
    # - whisper
    # - llm_generate
    # - audio_tts
    task_type = "base"

    # Executor 구현체 자체의 ID.
    #
    # 기존 executor와의 하위 호환을 위해 None이면 task_type을 사용한다.
    #
    # 미래 예:
    # task_type = "audio_tts"
    # executor_id = "qwen3_tts"
    executor_id = None

    label = "Base Executor"
    description = ""

    # Executor 구현 버전.
    #
    # Worker 전체 버전과 별개로,
    # 특정 executor manifest/동작의 버전을 표현한다.
    version = "1.0.0"

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    # 기존 호환 필드.
    #
    # - "json": 일반 JSON payload로 제출 가능한 작업
    # - "multipart": 파일 업로드가 필요한 작업
    #
    # 기존 Central Server와의 호환을 위해 유지한다.
    submit_mode = "json"

    # ------------------------------------------------------------------
    # Runtime availability
    # ------------------------------------------------------------------

    enabled = True

    def is_enabled(self) -> bool:
        """
        설정상 executor가 활성화되어 있는지 반환한다.

        is_available()과 의미가 다르다.

        enabled:
            관리자가 이 executor 사용을 허용했는가?

        available:
            현재 실행 환경에서 실제 실행 가능한가?

        예:
            QWEN3_TTS_ENABLED=false
            -> enabled=False

            QWEN3_TTS_ENABLED=true지만 필요한 Python package가 없음
            -> enabled=True, available=False
        """
        return bool(self.enabled)

    def is_available(self) -> bool:
        """
        현재 executor가 실행 가능한 상태인지 반환한다.

        이 함수는 capability heartbeat에서도 호출될 수 있으므로
        모델을 실제로 로드하거나 다운로드하는 등의 비싼 작업을
        수행하면 안 된다.

        예:
        - Ollama 서버가 켜져 있으면 True
        - 필요한 Python package/runtime이 있으면 True
        - 실행 환경 자체가 준비되지 않았으면 False

        특정 모델이 설치되어 있지 않다는 이유만으로 executor 전체를
        unavailable로 만들지는 않는다.
        """
        return True

    # ------------------------------------------------------------------
    # Scheduler hints
    # ------------------------------------------------------------------

    def estimate_requirements(self, payload: dict) -> dict:
        """
        특정 payload를 실행하기 위해 필요한 리소스를 추정한다.

        Central Server Scheduler가 참고할 수 있다.

        payload에 따라 요구 VRAM 등이 달라질 수 있는 executor는
        이 메서드를 override한다.
        """
        return {
            "gpu": False,
            "min_vram_mb": 0,
        }

    def get_required_resources(self) -> dict:
        """
        executor의 기본 리소스 요구사항을 반환한다.

        특정 payload에 따른 동적 추정은 estimate_requirements()를 사용한다.
        """
        return {
            "gpu": False,
            "min_vram_mb": 0,
        }

    # ------------------------------------------------------------------
    # Self-describing API contract
    # ------------------------------------------------------------------

    def get_submit_modes(self) -> list[str]:
        """
        이 executor가 지원하는 제출 방식을 반환한다.

        기존 submit_mode 단일 필드와의 하위 호환을 위해
        기본값은 [submit_mode]이다.

        예:
            ["json"]
            ["multipart"]

        미래 executor는 필요하면 여러 mode를 반환할 수 있다.
        """
        return [self.submit_mode]

    def get_input_schema(self) -> dict:
        """
        Central Server / 관리 UI / 클라이언트가 executor 입력 구조를
        이해할 수 있도록 schema를 반환한다.

        각 executor는 필요하면 override한다.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def get_file_inputs(self) -> list[dict]:
        """
        multipart 작업에서 사용할 named file input들을 설명한다.

        기본값은 파일 입력 없음.

        미래 예:

        [
            {
                "name": "reference_audio",
                "label": "Reference Audio",
                "required": True,
                "accept": ["audio/wav", "audio/mpeg"]
            }
        ]

        조건부 required 등 복잡한 규칙이 필요하면
        executor가 추가 metadata를 자유롭게 제공할 수 있다.
        """
        return []

    def get_output_schema(self) -> dict:
        """
        executor 결과 구조를 설명한다.

        각 executor는 필요하면 override한다.
        """
        return {
            "type": "object",
            "properties": {},
        }

    def get_examples(self) -> list[dict]:
        """
        Central Server 관리자 페이지나 다른 SHS 서비스 개발자가
        실제 요청 방법을 확인할 수 있도록 예제 요청을 반환한다.

        기본값은 빈 목록이다.

        Executor별로 JSON 또는 multipart 예제를 제공할 수 있다.
        """
        return []

    # ------------------------------------------------------------------
    # Model capability
    # ------------------------------------------------------------------

    def get_models(self) -> list[dict]:
        """
        이 executor가 처리할 수 있는 모델과 현재 runtime 상태를 반환한다.

        기본값은 모델 개념이 없는 executor를 위해 빈 목록이다.

        모델 기반 executor는 필요하면 다음과 같은 정보를 제공할 수 있다.

        [
            {
                "id": "example/model",
                "supported": True,
                "installed": False,
                "loaded": False,
                "downloadable": True
            }
        ]

        중요한 원칙:
        - supported와 installed는 다른 개념이다.
        - 설치되어 있지 않아도 Node가 다운로드 후 실행할 수 있다면
          supported=True, downloadable=True가 될 수 있다.
        - 절대 로컬 모델 경로나 인증 정보를 manifest에 노출하지 않는다.
        """
        return []

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def get_manifest(self) -> dict:
        """
        Central Server에 제공할 executor capability manifest를 반환한다.

        목표:
        새로운 executor를 Node에 추가했을 때 Central Server가
        해당 executor를 하드코딩해서 이해할 필요가 없도록 한다.

        Central Server는 이 manifest를 저장/집계하여:
        - 사용 가능한 Executor 목록
        - 입력 schema
        - 파일 입력
        - 모델 상태
        - 예제 요청
        - 관리 페이지 문서
        등에 사용할 수 있다.

        기존 manifest 필드는 하위 호환을 위해 유지한다.
        """
        executor_id = self.executor_id or self.task_type
        submit_modes = self.get_submit_modes()

        return {
            # identity
            "executor_id": executor_id,
            "task_type": self.task_type,
            "label": self.label,
            "description": self.description,
            "version": self.version,

            # runtime
            "available": self.is_available(),
            "enabled": self.is_enabled(),

            # submission
            #
            # submit_mode:
            #   기존 Central Server 호환용
            #
            # submit_modes:
            #   새로운 self-describing manifest용
            "submit_mode": self.submit_mode,
            "submit_modes": submit_modes,

            # contracts
            "input_schema": self.get_input_schema(),
            "file_inputs": self.get_file_inputs(),
            "output_schema": self.get_output_schema(),

            # documentation
            "examples": self.get_examples(),

            # resources / models
            "required_resources": self.get_required_resources(),
            "models": self.get_models(),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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