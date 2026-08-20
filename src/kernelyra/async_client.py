from __future__ import annotations

import asyncio
from typing import Any

from .client import DaemonClient
from .models import RunConfig


class AsyncKernelyraClient:
    """Async facade over the dependency-free remote client.

    Network work runs outside the event loop. Retries remain limited to idempotent
    GET/HEAD requests by :class:`DaemonClient`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._client = DaemonClient(*args, **kwargs)

    async def request(self, method: str, path: str, body: Any | None = None) -> Any:
        return await asyncio.to_thread(self._client.request, method, path, body)

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.health)

    async def capabilities(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.capabilities)

    async def list_datasets(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._client.list_datasets)

    async def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.get_dataset, dataset_id)

    async def add_dataset(self, path: str, target: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.add_dataset, path, target)

    async def remove_dataset(self, dataset_id: str) -> None:
        await asyncio.to_thread(self._client.remove_dataset, dataset_id)

    async def list_runs(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._client.list_runs)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.get_run, run_id)

    async def create_run(self, config: RunConfig) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.create_run, config)

    async def command(self, run_id: str, command: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.command, run_id, command)

    async def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.get_run_metrics, run_id)

    async def get_run_logs(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._client.get_run_logs, run_id, limit)

    async def export_run(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._client.export_run, run_id)

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)

    async def __aenter__(self) -> AsyncKernelyraClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
