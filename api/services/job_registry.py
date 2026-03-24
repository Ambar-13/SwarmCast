"""In-memory async job store for background tasks (evidence pack)."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any


class JobState:
    def __init__(self) -> None:
        self.status: str = "queued"
        self.progress: float = 0.0
        self.result: Any | None = None
        self.error: str | None = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = asyncio.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobState()
        return job_id

    async def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)


registry = JobRegistry()
