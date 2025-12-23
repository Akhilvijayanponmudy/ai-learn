from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Week7 FastAPI Service"
    ENV: str = "dev"

    # Rate limits
    RATE_LIMIT_CLASSIFY: str = "30/minute"
    RATE_LIMIT_SEARCH: str = "60/minute"
    RATE_LIMIT_RAG: str = "20/minute"

    # Search defaults
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
