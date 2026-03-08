from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required
    gemini_api_key: str = ""

    # Database
    database_url: str = "postgresql://evaluser:evalpass@localhost:5432/evaldb"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # App
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Evaluation
    llm_judge_model: str = "gemini-2.0-flash-lite"
    sync_evaluation: bool = False  # True on Vercel (no Celery workers)
    latency_threshold_ms: int = 1000
    min_annotation_agreement: float = 0.6
    auto_suggest_after_n_evals: int = 100


settings = Settings()
