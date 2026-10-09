from __future__ import annotations

import asyncio

import pytest

from embedserve.scheduler import (
    CoordinatorState,
    ModelBusy,
    ModelCoordinator,
    ModelLoadError,
)


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Runtime:
    def __init__(self, loaded: str | None = "nomic") -> None:
        self.resident = {loaded} if loaded else set()
        self.events: list[tuple[str, str]] = []
        self.fail_load: set[str] = set()
        self.load_started: asyncio.Event | None = None
        self.allow_load: asyncio.Event | None = None

    async def load(self, model: str) -> None:
        self.events.append(("load", model))
        if self.load_started is not None:
            self.load_started.set()
        if self.allow_load is not None:
            await self.allow_load.wait()
        if model in self.fail_load:
            raise RuntimeError("synthetic load failure")
        self.resident.add(model)
        assert len(self.resident) == 1

    async def unload(self, model: str) -> None:
        self.events.append(("unload", model))
        self.resident.remove(model)


def coordinator(
    runtime: Runtime, clock: Clock | None = None, minimum_residency: float = 0
) -> ModelCoordinator:
    return ModelCoordinator(
        loaded_model=next(iter(runtime.resident), None),
        load_model=runtime.load,
        unload_model=runtime.unload,
        intent_ttl_seconds=10,
        reservation_ttl_seconds=5,
        minimum_residency_seconds=minimum_residency,
        retry_after_ms=1500,
        monotonic=clock or Clock(),
    )


@pytest.mark.asyncio
async def test_same_model_requests_share_residency_without_switching() -> None:
    runtime = Runtime()
    manager = coordinator(runtime)
    first = await manager.acquire("nomic")
    second = await manager.acquire("nomic")
    assert (await manager.snapshot()).active_requests == 2
    assert runtime.events == []
    await first.release()
    await second.release()
    assert (await manager.snapshot()).state == CoordinatorState.READY


@pytest.mark.asyncio
async def test_idle_alternate_request_reserves_then_retry_claims() -> None:
    runtime = Runtime()
    manager = coordinator(runtime)
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    admission = await manager.acquire("e5")
    await admission.release()
    assert runtime.events == [("unload", "nomic"), ("load", "e5")]


@pytest.mark.asyncio
async def test_alternate_intent_blocks_old_arrivals_and_switches_once() -> None:
    runtime = Runtime()
    manager = coordinator(runtime)
    active = await manager.acquire("nomic")

    with pytest.raises(ModelBusy) as alternate:
        await manager.acquire("e5")
    assert alternate.value.retry_after_seconds == 2
    assert alternate.value.payload()["error"] == {
        "code": "MODEL_BUSY",
        "message": "Requested model is waiting for the active model to finish.",
        "retryable": True,
        "loaded_model": "nomic",
        "requested_model": "e5",
        "state": CoordinatorState.SWITCH_RESERVED,
        "retry_after_ms": 1500,
    }
    with pytest.raises(ModelBusy):
        await manager.acquire("nomic")

    await active.release()
    admission = await manager.acquire("e5")
    assert runtime.events == [("unload", "nomic"), ("load", "e5")]
    assert runtime.resident == {"e5"}
    await admission.release()


@pytest.mark.asyncio
async def test_continuous_old_model_traffic_cannot_starve_waiter() -> None:
    runtime = Runtime()
    manager = coordinator(runtime)
    active = await manager.acquire("nomic")
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    for _ in range(50):
        with pytest.raises(ModelBusy):
            await manager.acquire("nomic")
    await active.release()
    alternate = await manager.acquire("e5")
    await alternate.release()
    assert runtime.resident == {"e5"}


@pytest.mark.asyncio
async def test_reverse_handoff() -> None:
    runtime = Runtime("e5")
    manager = coordinator(runtime)
    active = await manager.acquire("e5")
    with pytest.raises(ModelBusy):
        await manager.acquire("nomic")
    await active.release()
    nomic = await manager.acquire("nomic")
    await nomic.release()
    assert runtime.events == [("unload", "e5"), ("load", "nomic")]


@pytest.mark.asyncio
async def test_minimum_residency_prevents_model_thrashing() -> None:
    runtime = Runtime()
    clock = Clock()
    manager = coordinator(runtime, clock, minimum_residency=5)
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    e5 = await manager.acquire("e5")
    await e5.release()

    with pytest.raises(ModelBusy):
        await manager.acquire("nomic")
    assert (await manager.snapshot()).reserved_model is None
    clock.advance(5)
    with pytest.raises(ModelBusy):
        await manager.acquire("nomic")
    nomic = await manager.acquire("nomic")
    await nomic.release()
    assert runtime.events == [
        ("unload", "nomic"),
        ("load", "e5"),
        ("unload", "e5"),
        ("load", "nomic"),
    ]


@pytest.mark.asyncio
async def test_abandoned_reservation_expires() -> None:
    runtime = Runtime()
    clock = Clock()
    manager = coordinator(runtime, clock)
    active = await manager.acquire("nomic")
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    clock.advance(6)
    resumed = await manager.acquire("nomic")
    await resumed.release()
    await active.release()
    clock.advance(5)
    snapshot = await manager.snapshot()
    assert snapshot.reserved_model is None
    assert snapshot.pending_model is None
    assert runtime.events == []


@pytest.mark.asyncio
async def test_cancelled_claimant_does_not_wedge_switch() -> None:
    runtime = Runtime()
    runtime.load_started = asyncio.Event()
    runtime.allow_load = asyncio.Event()
    manager = coordinator(runtime)
    active = await manager.acquire("nomic")
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    await active.release()

    claimant = asyncio.create_task(manager.acquire("e5"))
    await runtime.load_started.wait()
    claimant.cancel()
    with pytest.raises(asyncio.CancelledError):
        await claimant
    runtime.allow_load.set()
    for _ in range(20):
        if (await manager.snapshot()).loaded_model == "e5":
            break
        await asyncio.sleep(0)
    snapshot = await manager.snapshot()
    assert snapshot.loaded_model == "e5"
    assert snapshot.active_requests == 0
    assert not snapshot.switch_in_progress
    retry = await manager.acquire("e5")
    await retry.release()


@pytest.mark.asyncio
async def test_claimant_timeout_during_loading_does_not_wedge_switch() -> None:
    runtime = Runtime()
    runtime.load_started = asyncio.Event()
    runtime.allow_load = asyncio.Event()
    manager = coordinator(runtime)
    active = await manager.acquire("nomic")
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    await active.release()

    async def claim() -> None:
        async with asyncio.timeout(0.01):
            await manager.acquire("e5")

    task = asyncio.create_task(claim())
    await runtime.load_started.wait()
    with pytest.raises(TimeoutError):
        await task
    runtime.allow_load.set()
    for _ in range(20):
        snapshot = await manager.snapshot()
        if snapshot.loaded_model == "e5" and not snapshot.switch_in_progress:
            break
        await asyncio.sleep(0)
    snapshot = await manager.snapshot()
    assert snapshot.loaded_model == "e5"
    assert snapshot.active_requests == 0
    assert snapshot.reserved_model is None


@pytest.mark.asyncio
async def test_load_failure_restores_known_good_model() -> None:
    runtime = Runtime()
    runtime.fail_load.add("e5")
    manager = coordinator(runtime)
    active = await manager.acquire("nomic")
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    await active.release()

    with pytest.raises(ModelLoadError) as raised:
        await manager.acquire("e5")
    assert raised.value.failure.retryable
    assert raised.value.failure.restored_model == "nomic"
    snapshot = await manager.snapshot()
    assert snapshot.state == CoordinatorState.READY
    assert snapshot.loaded_model == "nomic"
    assert runtime.resident == {"nomic"}
    assert runtime.events == [
        ("unload", "nomic"),
        ("load", "e5"),
        ("load", "nomic"),
    ]

    runtime.fail_load.clear()
    with pytest.raises(ModelBusy):
        await manager.acquire("e5")
    retry = await manager.acquire("e5")
    await retry.release()
    assert runtime.resident == {"e5"}


@pytest.mark.asyncio
async def test_cancellation_releases_active_counter() -> None:
    runtime = Runtime()
    manager = coordinator(runtime)
    entered = asyncio.Event()

    async def request() -> None:
        async with await manager.acquire("nomic"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = await manager.snapshot()
    assert snapshot.active_requests == 0
    assert snapshot.state == CoordinatorState.READY


@pytest.mark.asyncio
async def test_restart_drops_abandoned_reservation() -> None:
    runtime = Runtime()
    before_restart = coordinator(runtime)
    active = await before_restart.acquire("nomic")
    with pytest.raises(ModelBusy):
        await before_restart.acquire("e5")
    await active.release()
    assert (await before_restart.snapshot()).reserved_model == "e5"

    after_restart = coordinator(runtime)
    snapshot = await after_restart.snapshot()
    assert snapshot.state == CoordinatorState.READY
    assert snapshot.loaded_model == "nomic"
    assert snapshot.reserved_model is None
    nomic = await after_restart.acquire("nomic")
    await nomic.release()
