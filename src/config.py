from pydantic import field_validator
from pydantic_settings import BaseSettings,SettingsConfigDict
from enum import IntEnum


def _clean_env_bool(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    quote = value[0] if value[0] in {"'", '"'} else ""
    if quote:
        end = value.find(quote, 1)
        value = value[1:end] if end != -1 else value[1:]
    else:
        value = value.split("#", 1)[0].strip()
    return value.lower() in {"1", "true", "yes", "y", "on"}

class SequenceLength(IntEnum):

    K_8 = 8 * 1024
    K_16 = 16 * 1024
    K_32 = 32 * 1024
    K_64 = 64 * 1024
    K_128 = 128 * 1024
    K_256 = 256 * 1024
    K_512 = 512 * 1024

    M_1 = 1 * 1024 * 1024
    M_2 = 2 * 1024 * 1024

class Settings(BaseSettings):
    MODEL_NAME: str = "all_folds"
    SEQUENCE_LEN: int = SequenceLength.K_8
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    DEVICE: str = "cuda:0"
    HF_TOKEN: str = ""
    BATCH_SIZE: int = 1
    WINDOW_SWEEP: bool = False

    @field_validator("WINDOW_SWEEP", mode="before")
    @classmethod
    def parse_window_sweep(cls, value):
        return _clean_env_bool(value)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ALPHAGENOME_",
        extra="ignore"  # Ignora chaves no .env que não estejam mapeadas na classe
    )


settings = Settings()
