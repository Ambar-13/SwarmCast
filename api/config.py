import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve .env path relative to this file so it works regardless of cwd
_ENV_FILE = Path(__file__).parent / ".env"


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

    # OpenAI — required for swarm elicitation and run_llm_strategic
    # Priority: POLICYLAB_OPENAI_API_KEY env var > OPENAI_API_KEY env var > api/.env file
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="POLICYLAB_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_openai_key(self) -> str | None:
        """Check all sources at call time (not just at startup).

        Priority order:
          1. POLICYLAB_OPENAI_API_KEY (captured at startup by pydantic-settings)
          2. OPENAI_API_KEY env var (read live — picks up changes without restart)
          3. OPENAI_API_KEY in api/.env file (read live)
        """
        # Live read from environment (catches vars set after process start)
        live_policylab = os.environ.get("POLICYLAB_OPENAI_API_KEY")
        live_openai    = os.environ.get("OPENAI_API_KEY")

        # Also check .env file directly so it works even without env export
        dotenv_key: str | None = None
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("OPENAI_API_KEY", "POLICYLAB_OPENAI_API_KEY") and v:
                    dotenv_key = v
                    break

        return live_policylab or self.openai_api_key or live_openai or dotenv_key


settings = Settings()
