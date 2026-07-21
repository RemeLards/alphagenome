from pydantic_settings import BaseSettings,SettingsConfigDict
from enum import IntEnum

class SequenceLength(IntEnum):

    KB_8 = 8 * 1024
    KB_16 = 16 * 1024
    KB_32 = 32 * 1024
    KB_64 = 64 * 1024
    KB_128 = 128 * 1024
    KB_256 = 256 * 1024
    KB_512 = 512 * 1024

    MB_1 = 1 * 1024 * 1024
    MB_2 = 2 * 1024 * 1024

class Settings(BaseSettings):
    model_name: str = "all_folds"
    sequence_len: int = SequenceLength.KB_8
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    device: str = "cuda:0"
    hf_token: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ALPHAGENOME_",
        extra="ignore"  # Ignora chaves no .env que não estejam mapeadas na classe
    )


settings = Settings()
