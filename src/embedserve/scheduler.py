"""Starvation-safe admission and handoff for one GPU-resident model."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType


class CoordinatorState(StrEnum):
    EMPTY = "EMPTY"
    LOADING = "LOADING"
    READY = "READY"
    SERVING = "SERVING"
    SWITCH_RESERVED = "SWITCH_RESERVED"
    UNLOADING = "UNLOADING"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class ModelFailure:
    model: str
    error_type: str
    retryable: bool
    restored_model: str | None


@dataclass(frozen=True)
class CoordinatorSnapshot:
    state: CoordinatorState
    loaded_model: str | None
    active_requests: int
    pending_model: str | None
    intent_expires_at: float | None
    reserved_model: str | None
    reservation_expires_at: float | None
    switch_in_progress: bool
    last_activity: float | None
    last_successful_switch: float | None
    switch_protected_until: float | None
    model_load_failure: ModelFailure | None


class ModelBusy(RuntimeError):
    def __init__(
        self,
        *,
        loaded_model: str | None,
        requested_model: str,
        state: CoordinatorState,
        retry_after_ms: int,
    ) -> None:
        super().__init__("requested model is waiting for the active model to finish")
        self.loaded_model = loaded_model
        self.requested_model = requested_model
        self.state = state
        self.retry_after_ms = retry_after_ms

    @property
    def retry_after_seconds(self) -> int:
        return max(1, math.ceil(self.retry_after_ms / 1000))

    def payload(self) -> dict[str, object]:
        return {
            "error": {
                "code": "MODEL_BUSY",
                "message": "Requested model is waiting for the active model to finish.",
                "retryable": True,
                "loaded_model": self.loaded_model,
                "requested_model": self.requested_model,
                "state": self.state,
                "retry_after_ms": self.retry_after_ms,
            }
        }


class ModelLoadError(RuntimeError):
    def __init__(self, failure: ModelFailure) -> None:
        super().__init__(f"model {failure.model} could not be loaded")
        self.failure = failure


@dataclass(frozen=True)
class _Reservation:
    model: str
    expires_at: float


@dataclass(frozen=True)
class _SwitchOutcome:
    failure: ModelFailure | None = None


class Admission:
    def __init__(self, coordinator: ModelCoordinator, model: str) -> None:
        self._coordinator = coordinator
        self.model = model
        self._released = False

    async def __aenter__(self) -> Admission:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._coordinator._release(self.model)


class ModelCoordinator:
    def __init__(
        self,
        *,
        loaded_model: str | None,
        load_model: Callable[[str], Awaitable[None]],
        unload_model: Callable[[str], Awaitable[None]],
        intent_ttl_seconds: float = 10.0,
        reservation_ttl_seconds: float = 5.0,
        minimum_residency_seconds: float = 5.0,
        retry_after_ms: int = 1500,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if intent_ttl_seconds <= 0 or reservation_ttl_seconds <= 0:
            raise ValueError("intent and reservation TTLs must be positive")
        if minimum_residency_seconds < 0:
            raise ValueError("minimum residency must not be negative")
        if retry_after_ms <= 0:
            raise ValueError("retry_after_ms must be positive")
        self._load_model = load_model
        self._unload_model = unload_model
        self._intent_ttl = intent_ttl_seconds
        self._reservation_ttl = reservation_ttl_seconds
        self._minimum_residency = minimum_residency_seconds
        self._retry_after_ms = retry_after_ms
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._state = CoordinatorState.READY if loaded_model else CoordinatorState.EMPTY
        self._loaded_model = loaded_model
        self._active_requests = 0
        self._pending_model: str | None = None
        self._intent_expires_at: float | None = None
        self._reservation: _Reservation | None = None
        self._switch_task: asyncio.Task[_SwitchOutcome] | None = None
        self._last_activity: float | None = None
        self._last_successful_switch: float | None = None
        self._model_load_failure: ModelFailure | None = None

    async def acquire(self, requested_model: str) -> Admission:
        while True:
            switch_task: asyncio.Task[_SwitchOutcome] | None = None
            async with self._lock:
                now = self._monotonic()
                self._expire(now)
                if self._switch_task is not None:
                    raise self._busy(requested_model)

                reservation = self._reservation
                if self._loaded_model == requested_model:
                    if reservation is not None and reservation.model != requested_model:
                        raise self._busy(requested_model)
                    self._active_requests += 1
                    self._last_activity = now
                    self._state = CoordinatorState.SERVING
                    return Admission(self, requested_model)

                self._pending_model = requested_model
                self._intent_expires_at = now + self._intent_ttl
                protected_until = self._protected_until()
                if protected_until is not None and now < protected_until:
                    raise self._busy(requested_model)
                created_reservation = reservation is None or reservation.model != requested_model
                if created_reservation:
                    self._reservation = _Reservation(
                        requested_model, now + self._reservation_ttl
                    )
                else:
                    assert reservation is not None
                    self._reservation = _Reservation(
                        reservation.model, now + self._reservation_ttl
                    )
                self._state = CoordinatorState.SWITCH_RESERVED

                if self._active_requests > 0 or created_reservation:
                    raise self._busy(requested_model)

                old_model = self._loaded_model
                self._state = (
                    CoordinatorState.UNLOADING if old_model else CoordinatorState.LOADING
                )
                switch_task = asyncio.create_task(
                    self._switch(old_model, requested_model),
                    name=f"embedserve-switch-{requested_model}",
                )
                self._switch_task = switch_task

            outcome = await asyncio.shield(switch_task)
            if outcome.failure is not None:
                raise ModelLoadError(outcome.failure)

    async def snapshot(self) -> CoordinatorSnapshot:
        async with self._lock:
            now = self._monotonic()
            self._expire(now)
            reservation = self._reservation
            return CoordinatorSnapshot(
                state=self._state,
                loaded_model=self._loaded_model,
                active_requests=self._active_requests,
                pending_model=self._pending_model,
                intent_expires_at=self._intent_expires_at,
                reserved_model=reservation.model if reservation else None,
                reservation_expires_at=reservation.expires_at if reservation else None,
                switch_in_progress=self._switch_task is not None,
                last_activity=self._last_activity,
                last_successful_switch=self._last_successful_switch,
                switch_protected_until=self._protected_until(),
                model_load_failure=self._model_load_failure,
            )

    async def _release(self, model: str) -> None:
        async with self._lock:
            if self._active_requests <= 0 or self._loaded_model != model:
                raise RuntimeError("model admission counter is inconsistent")
            self._active_requests -= 1
            self._last_activity = self._monotonic()
            if self._active_requests == 0:
                self._state = (
                    CoordinatorState.SWITCH_RESERVED
                    if self._reservation is not None
                    else CoordinatorState.READY
                )

    async def _switch(self, old_model: str | None, requested_model: str) -> _SwitchOutcome:
        old_unloaded = False
        current = asyncio.current_task()
        try:
            if old_model is not None:
                await self._unload_model(old_model)
                old_unloaded = True
                async with self._lock:
                    self._loaded_model = None
                    self._state = CoordinatorState.LOADING
            await self._load_model(requested_model)
        except asyncio.CancelledError:
            await self._recover(old_model, requested_model, old_unloaded, "CancelledError")
            raise
        except Exception as error:
            failure = await self._recover(
                old_model, requested_model, old_unloaded, type(error).__name__
            )
            return _SwitchOutcome(failure)
        else:
            async with self._lock:
                now = self._monotonic()
                self._loaded_model = requested_model
                self._state = CoordinatorState.READY
                self._reservation = None
                self._pending_model = None
                self._intent_expires_at = None
                self._last_activity = now
                self._last_successful_switch = now
                self._model_load_failure = None
            return _SwitchOutcome()
        finally:
            async with self._lock:
                if self._switch_task is current:
                    self._switch_task = None

    async def _recover(
        self,
        old_model: str | None,
        requested_model: str,
        old_unloaded: bool,
        error_type: str,
    ) -> ModelFailure:
        restored_model: str | None = old_model if not old_unloaded else None
        if old_unloaded and old_model is not None:
            try:
                await self._load_model(old_model)
            except Exception:
                restored_model = None
            else:
                restored_model = old_model
        failure = ModelFailure(
            model=requested_model,
            error_type=error_type,
            retryable=True,
            restored_model=restored_model,
        )
        async with self._lock:
            self._loaded_model = restored_model
            self._state = (
                CoordinatorState.READY if restored_model else CoordinatorState.DEGRADED
            )
            self._reservation = None
            self._pending_model = None
            self._intent_expires_at = None
            self._last_activity = self._monotonic()
            self._model_load_failure = failure
        return failure

    def _expire(self, now: float) -> None:
        if self._switch_task is not None:
            return
        if self._intent_expires_at is not None and self._intent_expires_at <= now:
            self._pending_model = None
            self._intent_expires_at = None
        if self._reservation is not None and self._reservation.expires_at <= now:
            self._reservation = None
        if self._reservation is None and self._state == CoordinatorState.SWITCH_RESERVED:
            self._state = (
                CoordinatorState.SERVING
                if self._active_requests
                else CoordinatorState.READY
                if self._loaded_model
                else CoordinatorState.EMPTY
            )

    def _busy(self, requested_model: str) -> ModelBusy:
        return ModelBusy(
            loaded_model=self._loaded_model,
            requested_model=requested_model,
            state=self._state,
            retry_after_ms=self._retry_after_ms,
        )

    def _protected_until(self) -> float | None:
        if self._last_successful_switch is None:
            return None
        return self._last_successful_switch + self._minimum_residency
