"""
Application configuration via Pydantic BaseSettings.

All settings are read from environment variables or from a .env file
at project root.  Defaults are safe for local development only.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Pramana"
    environment: str = "development"
    log_level: str = "INFO"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    database_url: str = "sqlite+aiosqlite:///./pramana.db"
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def get_settings() -> Settings:
    """Return the application settings singleton."""
    # TODO: implement
    return settings
