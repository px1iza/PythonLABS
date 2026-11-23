from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    index_path: Path = Path("index_v4.bin")

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )
