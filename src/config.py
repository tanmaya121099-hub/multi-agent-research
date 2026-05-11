from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    planner_model: str = Field(default="gpt-4o-mini")
    researcher_model: str = Field(default="gpt-4o-mini")
    critic_model: str = Field(default="gpt-4o-mini")
    writer_model: str = Field(default="claude-sonnet-4-6")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="multi-agent-research")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_short_ttl: int = Field(default=3600)
    redis_working_ttl: int = Field(default=86400)

    # Postgres
    postgres_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/research")
    postgres_pool_size: int = Field(default=10)

    # Tavily
    tavily_api_key: str = Field(default="")

    # Agent behaviour
    max_critic_retries: int = Field(default=2)
    max_subtasks: int = Field(default=5)

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    metrics_port: int = Field(default=9091)


settings = Settings()
