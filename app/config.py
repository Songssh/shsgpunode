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
    # Pydantic Settings
    # ------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()