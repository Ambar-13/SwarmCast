from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    allow_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
    ]
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB
    default_n_population: int = 1000
    default_num_rounds: int = 16

    model_config = {"env_prefix": "POLICYLAB_"}


settings = Settings()
