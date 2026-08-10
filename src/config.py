from pydantic_settings import BaseSettings,SettingsConfigDict
from enum import IntEnum

class SequenceLength(IntEnum):

    KB_8 = 8 * 1000
    KB_16 = 16 * 1000
    KB_32 = 32 * 1000
    KB_64 = 64 * 1000
    KB_128 = 128 * 1000
    KB_256 = 256 * 1000
    KB_512 = 512 * 1000

    MB_1 = 1 * 1000 * 1000
    MB_2 = 2 * 1000 * 1000

class Settings(BaseSettings):
    MODEL_NAME: str = "all_folds"
    SEQUENCE_LEN: int = SequenceLength.KB_8
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    DEVICE: str = "cuda:0"
    HF_TOKEN: str = ""
    BATCH_SIZE: int = 1
    WINDOW_SWEEP: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ALPHAGENOME_",
        extra="ignore"  # Ignora chaves no .env que não estejam mapeadas na classe
    )


settings = Settings()
