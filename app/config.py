from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------
    # App
    # ------------------------------------------------------------
    debug: bool = False
    app_name: str = "SHS GPU Worker Node"

    # Worker software/protocol version
    worker_version: str = "0.1.0"
    worker_protocol_version: str = "1.0"

    # Timezone for timestamp reporting
    app_timezone: str = "UTC" 

    # ------------------------------------------------------------
    # Node
    # ------------------------------------------------------------
    node_id: str = "auto"
    node_name: str = "shs-gpu-node-01"

    host: str = "0.0.0.0"
    port: int = 8100

    # Central Server가 이 Worker Node에 접근할 때 사용할 공개 주소.
    # 예:
    # NODE_PUBLIC_BASE_URL=http://192.168.0.99:8100
    node_public_base_url: str = "http://127.0.0.1:8100"

    # ------------------------------------------------------------
    # Central Server
    # ------------------------------------------------------------
    central_enabled: bool = False

    # 기존 호환용 설정.
    # 기존 app.agent.client.CentralServerClient가 이 값을 쓸 수 있으므로 유지한다.
    central_server_url: str = "http://central-gpu.shs"
    central_api_key: str = "change-me"

    # 신규 중앙 서버 연동용 설정.
    # 앞으로는 central_base_url을 기준으로 register/heartbeat URL을 조립한다.
    central_base_url: str = "http://central-gpu.shs"
    central_register_path: str = "/api/nodes/register"
    central_heartbeat_path: str = "/api/nodes/{node_id}/heartbeat"
    central_heartbeat_interval_seconds: int = 10
    central_register_retry_seconds: int = 10

    # ------------------------------------------------------------
    # Internal API Auth
    # ------------------------------------------------------------
    # Central Server -> Worker Node /api/worker/* 요청 검증에 사용.
    shs_internal_api_key: str = "change-this-secret"
    shs_internal_api_key_header: str = "x-shs-internal-api-key"

    # ------------------------------------------------------------
    # Model Paths
    # ------------------------------------------------------------
    model_dir: str = "./data/models"

    # ------------------------------------------------------------
    # LLM / Ollama
    # ------------------------------------------------------------
    ollama_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 300
    default_llm_model: str = "llama3.2:latest"

    # ------------------------------------------------------------
    # TTS / Qwen3-TTS
    # ------------------------------------------------------------
    qwen3_tts_enabled: bool = True

    # 모델이 로컬에 없을 때 자동 다운로드할지 여부.
    qwen3_tts_auto_download: bool = True

    # 실제 모델 저장 위치는:
    # {model_dir}/qwen3_tts/
    #
    # 예:
    # ./data/models/qwen3_tts/Qwen3-TTS-12Hz-0.6B-Base/
    #
    # 별도 absolute path 설정을 두지 않고 기존 model_dir 정책을 따른다.

    # 기본 실행 장치.
    # 예:
    # cuda:0
    # cpu
    qwen3_tts_device: str = "cuda:0"

    # 모델 dtype.
    # 지원 예정:
    # auto
    # float16
    # bfloat16
    # float32
    qwen3_tts_dtype: str = "bfloat16"

    # Attention 구현.
    #
    # auto:
    #   별도 attn_implementation을 강제하지 않는다.
    #
    # sdpa:
    #   PyTorch SDPA 사용
    #
    # flash_attention_2:
    #   사용자가 별도로 FlashAttention 2를 설치한 경우
    qwen3_tts_attention: str = "auto"

    # 동시에 GPU 메모리에 유지할 Qwen3-TTS 모델 수.
    # 현재 Worker 자체가 worker_count=1이므로 첫 버전은 1이 가장 안전하다.
    qwen3_tts_max_loaded_models: int = 1

    # ------------------------------------------------------------
    # Pydantic Settings
    # ------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()